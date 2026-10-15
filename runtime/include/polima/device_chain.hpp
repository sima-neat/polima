// On-device graph programs (`run_device_chain`).
//
// Generalises SharedRunnerChain (one input, one output, ping-pong) and
// DiTRunnerChain (four persistent bindings) to what GR00T's backbone needs:
//
//   * plan-wide named BF16 device buffers ("device_buffers" in plan.json) that
//     survive across steps, so one chain can leave its result where a later
//     chain reads it (vision -> language input slots, LM -> DiT backbone);
//   * graph bindings to sub-spans of a buffer, so a producer writes straight
//     into the hidden-state part of the consumer's packed input while the
//     consumer's other inputs (cos/sin/masks) stay resident behind it;
//   * several downloads (deepstack side outputs, a layer whose output the host
//     must still modify) and uploads, optionally cached across requests.
//
// The bytes each graph reads are exactly the bytes the per-graph Runner path
// would upload: BF16 -> float32 -> BF16 is the identity for finite values, and
// every intermediate here is a graph output that the Runner would have decoded
// and re-encoded unchanged. Host arithmetic (Euler adds, pixel-unshuffle) stays
// on the host. INT8 graphs are refused: their boundary is a per-graph
// requantisation, not a copy.
//
// Static checks at plan load (DeviceProgramChecker) guarantee that every device
// span a graph or download reads was written earlier in the same request, so
// nothing from a previous observation can leak in, and that cached upload
// regions are never written by anything else.
//
// Host boundary work uses the shared kernels (kernels.hpp): uploads convert
// with the NEON/worker-pool bf16 encode straight into the device mapping;
// downloads decode and check finiteness in one pass. Graph calls go to the MLA
// as queued submissions split so that no ELF occurs twice in one (see
// queue_segments in runner.hpp; POLIMA_DISABLE_MLA_QUEUE=1 runs them one by
// one). Diagnostics: POLIMA_DEVICE_NO_CACHE=1 re-uploads cached regions every
// request; POLIMA_DEVICE_VERIFY_CACHE=1 re-reads them after every program.
#pragma once

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "polima/bf16.hpp"
#include "polima/runner.hpp"
#include "polima/sidecar.hpp"

namespace polima {

// What a device program needs to know about one compiled graph (from bundle.json).
struct DeviceGraph {
  std::string name;
  std::filesystem::path elf;
  size_t input_elements = 0;
  size_t output_elements = 0;
  // Separate MLA input bindings, in compiler order. A packed-input ELF has one.
  std::vector<size_t> bindings;
  std::string external_layout = "compiler";
  bool plain_output = true;
  bool int8 = false;
};

struct DeviceSpan {
  size_t buffer = 0;
  size_t offset = 0;  // elements
  size_t count = 0;   // elements
};

// Plan-wide BF16 device buffers plus the upload cache that describes what a
// cached region currently holds.
class DeviceArena {
 public:
  DeviceArena() = default;
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  void declare(const std::string& name, size_t elements) {
    if (!elements) throw std::runtime_error("device buffer '" + name + "' is empty");
    if (names_.count(name)) throw std::runtime_error("device buffer '" + name + "' declared twice");
    names_[name] = buffers_.size();
    buffers_.push_back(std::make_unique<Buffer>("dev_" + name, elements));
    labels_.push_back(name);
  }
  bool has(const std::string& name) const { return names_.count(name) != 0; }
  size_t index(const std::string& name) const {
    auto found = names_.find(name);
    if (found == names_.end()) throw std::runtime_error("undeclared device buffer '" + name + "'");
    return found->second;
  }
  size_t count() const { return buffers_.size(); }
  size_t elements(size_t index) const { return buffers_.at(index)->elements; }
  const std::string& label(size_t index) const { return labels_.at(index); }
  llima::MLABuffer& storage(size_t index) { return buffers_.at(index)->storage; }
  uint16_t* data(size_t index) {
    return static_cast<uint16_t*>(buffers_.at(index)->storage.get_virtual_addr());
  }

  // What a cached upload region currently holds. Keyed by the exact span.
  struct Cached {
    bool valid = false;
    std::string sidecar;          // sidecar source, or empty for a host buffer
    size_t source_offset = 0;     // effective sidecar offset (task selection)
    std::vector<float> snapshot;  // host-buffer source: the values last uploaded
    std::vector<uint16_t> device;  // POLIMA_DEVICE_VERIFY_CACHE: the bytes uploaded
  };
  Cached& cached(const DeviceSpan& span) {
    return cache_[std::make_tuple(span.buffer, span.offset, span.count)];
  }

  // Diagnostics: re-read every cached region and report the first one whose
  // device bytes no longer match what was uploaded (something wrote outside
  // its declared binding).
  void verify_cached(const std::string& after) {
    for (auto& [key, entry] : cache_) {
      if (!entry.valid || entry.device.empty()) continue;
      const auto [buffer, offset, count] = key;
      detail::invalidate_range(storage(buffer), offset * 2, count * 2);
      const uint16_t* now = data(buffer) + offset;
      for (size_t i = 0; i < count; ++i)
        if (now[i] != entry.device[i]) {
          size_t changed = 0;
          for (size_t j = 0; j < count; ++j) changed += now[j] != entry.device[j];
          throw std::runtime_error("cached device region " + label(buffer) + "[" +
                                   std::to_string(offset) + ":+" + std::to_string(count) +
                                   "] changed after " + after + ": first element " +
                                   std::to_string(i) + ", " + std::to_string(changed) +
                                   " elements differ");
        }
    }
  }

 private:
  struct Buffer {
    Buffer(const std::string& name, size_t count)
        : elements(count), storage(name, {1, count}, "bfloat16", false) {
      storage.allocate();
      storage.clear();
    }
    ~Buffer() { storage.free(); }
    size_t elements;
    llima::MLABuffer storage;
  };
  std::vector<std::unique_ptr<Buffer>> buffers_;
  std::vector<std::string> labels_;
  std::map<std::string, size_t> names_;
  std::map<std::tuple<size_t, size_t, size_t>, Cached> cache_;
};

class DeviceChain {
 public:
  struct Upload {
    DeviceSpan destination;
    std::string host;  // plan buffer, or empty when `sidecar` is set
    std::string sidecar;
    size_t source_offset = 0;
    size_t source_stride = 0;
    size_t selector_count = 0;
    bool cached = false;
  };
  struct Call {
    std::string graph;
    std::vector<DeviceSpan> inputs;
    std::vector<DeviceSpan> outputs;
  };
  struct Download {
    DeviceSpan source;
    std::string host;
    size_t host_offset = 0;
  };

  DeviceChain(std::string label, const nlohmann::json& args, DeviceArena& arena,
              const std::map<std::string, DeviceGraph>& graphs,
              std::map<std::string, std::vector<float>>& host)
      : label_(std::move(label)), arena_(arena) {
    const auto span = [&](const nlohmann::json& value, const std::string& where,
                          bool binding = false) {
      // [buffer, offset, count]; offset/count default to the whole buffer.
      if (!value.is_array() || value.empty() || value.size() > 3)
        throw std::runtime_error(where + ": device span must be [buffer, offset, count]");
      DeviceSpan result;
      result.buffer = arena_.index(value.at(0).get<std::string>());
      result.offset = value.size() > 1 ? value.at(1).get<size_t>() : 0;
      result.count = value.size() > 2 ? value.at(2).get<size_t>()
                                      : arena_.elements(result.buffer) - result.offset;
      check_span(result, where, binding);
      return result;
    };
    const auto host_buffer = [&](const std::string& name, const std::string& where) {
      auto found = host.find(name);
      if (found == host.end())
        throw std::runtime_error(where + ": undeclared plan buffer '" + name + "'");
      return &found->second;
    };

    for (const auto& item : args.value("uploads", nlohmann::json::array())) {
      const std::string where = label_ + " upload";
      Upload upload;
      upload.destination = span({item.at("dst"), item.value("dst_offset", size_t{0}),
                                 item.at("count").get<size_t>()}, where);
      upload.cached = item.value("cache", false);
      upload.source_offset = item.value("src_offset", size_t{0});
      if (item.contains("sidecar")) {
        upload.sidecar = item.at("sidecar").get<std::string>();
        upload.source_stride = item.value("source_stride", size_t{0});
        upload.selector_count = item.value("selector_count", size_t{0});
        if (upload.source_stride && !upload.selector_count)
          throw std::runtime_error(where + ": indexed sidecar has no selector_count");
      } else {
        upload.host = item.at("src").get<std::string>();
        const auto* source = host_buffer(upload.host, where);
        if (upload.source_offset + upload.destination.count > source->size())
          throw std::runtime_error(where + " reads past plan buffer '" + upload.host + "'");
        host_sources_.push_back(source);
      }
      if (!upload.sidecar.empty()) host_sources_.push_back(nullptr);
      uploads_.push_back(std::move(upload));
    }

    std::vector<std::filesystem::path> elfs;
    for (const auto& item : args.at("calls")) {
      Call call;
      call.graph = item.at("graph").get<std::string>();
      const std::string where = label_ + " graph " + call.graph;
      auto found = graphs.find(call.graph);
      if (found == graphs.end()) throw std::runtime_error(where + " has no ELF in bundle.json");
      const DeviceGraph& graph = found->second;
      if (graph.int8)
        throw std::runtime_error("INT8 graph cannot bypass boundary conversion in a shared chain: " +
                                 call.graph);
      if (!graph.plain_output)
        throw std::runtime_error(where + " needs a plain (untessellated) output layout");
      for (const auto& value : item.at("in")) call.inputs.push_back(span(value, where, true));
      for (const auto& value : item.at("out")) call.outputs.push_back(span(value, where, true));
      if (call.inputs.size() != graph.bindings.size())
        throw std::runtime_error(where + " binds " + std::to_string(call.inputs.size()) +
                                 " inputs but the ELF has " +
                                 std::to_string(graph.bindings.size()));
      for (size_t index = 0; index < call.inputs.size(); ++index)
        if (call.inputs[index].count != graph.bindings[index])
          throw std::runtime_error(where + " input " + std::to_string(index) +
                                   " size disagrees with the ELF binding");
      if (call.outputs.size() != 1 || call.outputs[0].count != graph.output_elements)
        throw std::runtime_error(where + " output size disagrees with the ELF");
      for (const auto& in : call.inputs)
        if (overlaps(in, call.outputs[0]))
          throw std::runtime_error(where + " reads and writes the same device span");

      std::vector<llima::MLABufferSlice> ifms, ofms;
      for (const auto& in : call.inputs) ifms.push_back(slice(in));
      ofms.push_back(slice(call.outputs[0]));
      models_.push_back(std::make_unique<llima::MLAModelWithBuffer>(graph.elf, ifms, ofms));
      models_.back()->load();
      stage_names_.push_back(call.graph);
      elfs.push_back(graph.elf);
      calls_.push_back(std::move(call));
    }
    if (calls_.empty()) throw std::runtime_error(label_ + " has no graph calls");
    segments_ = polima::queue_segments(elfs);

    for (const auto& item : args.value("downloads", nlohmann::json::array())) {
      const std::string where = label_ + " download";
      Download download;
      download.source = span({item.at("src"), item.value("src_offset", size_t{0}),
                              item.at("count").get<size_t>()}, where);
      download.host = item.at("dst").get<std::string>();
      download.host_offset = item.value("dst_offset", size_t{0});
      auto* target = host_buffer(download.host, where);
      if (download.host_offset + download.source.count > target->size())
        throw std::runtime_error(where + " writes past plan buffer '" + download.host + "'");
      host_targets_.push_back(target);
      downloads_.push_back(std::move(download));
    }
  }

  ~DeviceChain() {
    for (auto& model : models_) model->free();
    models_.clear();
  }
  DeviceChain(const DeviceChain&) = delete;
  DeviceChain& operator=(const DeviceChain&) = delete;

  void run(Sidecars& sidecars, uint32_t request_flags, RunTiming* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point since) {
      return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
    };
    if (timing) *timing = RunTiming{};

    auto started = Clock::now();
    for (size_t index = 0; index < uploads_.size(); ++index) upload(index, sidecars, request_flags);
    if (timing) timing->encode_ms = ms(started);

    started = Clock::now();
    run_models(models_, timing, &stage_names_, segments_);
    if (timing) timing->execute_ms = ms(started);

    started = Clock::now();
    for (size_t index = 0; index < downloads_.size(); ++index) download(index);
    if (timing) timing->decode_ms = ms(started);
    if (verify_cache()) arena_.verify_cached(label_);
  }

  const std::string& label() const { return label_; }
  const std::vector<Upload>& uploads() const { return uploads_; }
  const std::vector<Call>& calls() const { return calls_; }
  const std::vector<Download>& downloads() const { return downloads_; }
  // End index of each queued MLA submission (no ELF twice in one).
  const std::vector<size_t>& submissions() const { return segments_; }

  // Diagnostics only: always re-upload cached regions / check them after every program.
  static bool no_cache() { return flag("POLIMA_DEVICE_NO_CACHE"); }
  static bool verify_cache() { return flag("POLIMA_DEVICE_VERIFY_CACHE"); }
  static bool flag(const char* name) {
    const char* value = std::getenv(name);
    return value && *value && std::string(value) != "0";
  }

  static bool overlaps(const DeviceSpan& left, const DeviceSpan& right) {
    return left.buffer == right.buffer && left.offset < right.offset + right.count &&
           right.offset < left.offset + left.count;
  }

 private:
  void check_span(const DeviceSpan& span, const std::string& where, bool binding) const {
    const size_t elements = arena_.elements(span.buffer);
    if (!span.count || span.offset > elements || span.count > elements - span.offset)
      throw std::runtime_error(where + ": span outside device buffer '" +
                               arena_.label(span.buffer) + "'");
    // A sub-span is a raw DMA address/length handed to the MLA. Keep both
    // 64-byte aligned (the Runner path only ever binds whole buffers, which
    // the allocator page-aligns and pads to 16 bytes).
    const bool whole = span.offset == 0 && span.count == elements;
    if (binding && !whole && ((span.offset * 2) % 64 || (span.count * 2) % 64))
      throw std::runtime_error(where + ": device sub-span of '" + arena_.label(span.buffer) +
                               "' must be 64-byte aligned");
  }

  llima::MLABufferSlice slice(const DeviceSpan& span) {
    auto* storage = &arena_.storage(span.buffer);
    if (span.offset == 0 && span.count == arena_.elements(span.buffer))
      return llima::MLABufferSlice(storage);  // identical binding to Runner/SharedRunnerChain
    return llima::MLABufferSlice(storage, {0u, static_cast<uint32_t>(span.offset)},
                                 {1u, static_cast<uint32_t>(span.count)});
  }

  void upload(size_t index, Sidecars& sidecars, uint32_t request_flags) {
    const Upload& item = uploads_[index];
    const DeviceSpan& to = item.destination;
    const float* source = nullptr;
    size_t effective = item.source_offset;
    if (!item.sidecar.empty()) {
      const auto& values = sidecars.get(item.sidecar);
      if (item.source_stride) {
        if (request_flags >= item.selector_count)
          throw std::runtime_error("request flags " + std::to_string(request_flags) +
                                   " outside task catalog [0, " +
                                   std::to_string(item.selector_count) + ")");
        effective += static_cast<size_t>(request_flags) * item.source_stride;
      }
      if (effective + to.count > values.size())
        throw std::runtime_error(label_ + " upload reads past sidecar '" + item.sidecar + "'");
      source = values.data() + effective;
    } else {
      source = host_sources_[index]->data() + item.source_offset;
    }

    DeviceArena::Cached* cache = item.cached ? &arena_.cached(to) : nullptr;
    if (cache && cache->valid && !no_cache()) {
      if (!item.sidecar.empty() && cache->sidecar == item.sidecar &&
          cache->source_offset == effective)
        return;
      if (item.sidecar.empty() && cache->sidecar.empty() &&
          std::memcmp(cache->snapshot.data(), source, to.count * sizeof(float)) == 0)
        return;
    }
    kernels::f32_to_bf16(source, arena_.data(to.buffer) + to.offset, to.count);
    detail::flush_range(arena_.storage(to.buffer), to.offset * 2, to.count * 2);
    if (cache) {
      cache->valid = true;
      cache->sidecar = item.sidecar;
      cache->source_offset = effective;
      if (item.sidecar.empty()) cache->snapshot.assign(source, source + to.count);
      else cache->snapshot.clear();
      if (verify_cache()) {
        const uint16_t* written = arena_.data(to.buffer) + to.offset;
        cache->device.assign(written, written + to.count);
      }
    }
  }

  void download(size_t index) {
    const Download& item = downloads_[index];
    const DeviceSpan& from = item.source;
    detail::invalidate_range(arena_.storage(from.buffer), from.offset * 2, from.count * 2);
    const uint16_t* raw = arena_.data(from.buffer) + from.offset;
    float* out = host_targets_[index]->data() + item.host_offset;
    // Decode and check finiteness in one pass: a BF16 value is non-finite
    // exactly when its exponent bits are all set. Accelerator corruption must
    // become a failed request, never a motor command.
    const size_t bad = kernels::bf16_to_f32_checked(raw, out, from.count);
    if (bad != from.count)
      throw std::runtime_error("graph chain " + label_ + " produced a non-finite value in '" +
                               item.host + "' at index " + std::to_string(item.host_offset + bad));
  }

  std::string label_;
  DeviceArena& arena_;
  std::vector<Upload> uploads_;
  std::vector<const std::vector<float>*> host_sources_;  // parallel to uploads_
  std::vector<Call> calls_;
  std::vector<Download> downloads_;
  std::vector<std::vector<float>*> host_targets_;  // parallel to downloads_
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models_;
  std::vector<std::string> stage_names_;
  std::vector<size_t> segments_;
};

// Load-time proof that a plan's device programs never read stale device memory.
//
// Walks the steps in plan order with an empty device at request start: every
// span a graph or download reads must have been written earlier in the same
// request (by an upload or a graph). Cached upload regions must be exact
// repeats or disjoint, and nothing else may ever write them -- that is what
// makes skipping an unchanged cached upload equivalent to performing it.
class DeviceProgramChecker {
 public:
  explicit DeviceProgramChecker(const DeviceArena& arena) : written_(arena.count()) {}

  void collect_writes(const DeviceChain& chain) {
    for (const auto& upload : chain.uploads())
      (upload.cached ? cached_ : uncached_).push_back(upload.destination);
    for (const auto& call : chain.calls())
      for (const auto& out : call.outputs) uncached_.push_back(out);
  }

  void check_cached_regions() const {
    for (const auto& region : cached_) {
      for (const auto& other : uncached_)
        if (DeviceChain::overlaps(region, other))
          throw std::runtime_error("a cached device upload region is also written by a graph "
                                   "or an uncached upload");
      for (const auto& other : cached_)
        if (DeviceChain::overlaps(region, other) &&
            (region.offset != other.offset || region.count != other.count))
          throw std::runtime_error("cached device upload regions partially overlap");
    }
  }

  void simulate(const DeviceChain& chain) {
    for (const auto& upload : chain.uploads()) mark(upload.destination);
    for (const auto& call : chain.calls()) {
      for (const auto& in : call.inputs)
        if (!covered(in))
          throw std::runtime_error(chain.label() + " graph " + call.graph +
                                   " reads device memory not written earlier in the request");
      for (const auto& out : call.outputs) mark(out);
    }
    for (const auto& download : chain.downloads())
      if (!covered(download.source))
        throw std::runtime_error(chain.label() + " downloads device memory no graph wrote");
  }

 private:
  using Ranges = std::vector<std::pair<size_t, size_t>>;  // sorted, disjoint [begin, end)
  void mark(const DeviceSpan& span) {
    Ranges& ranges = written_[span.buffer];
    size_t begin = span.offset, end = span.offset + span.count;
    Ranges merged;
    for (const auto& [b, e] : ranges) {
      if (e < begin || b > end) merged.emplace_back(b, e);
      else { begin = std::min(begin, b); end = std::max(end, e); }
    }
    merged.emplace_back(begin, end);
    std::sort(merged.begin(), merged.end());
    ranges.swap(merged);
  }
  bool covered(const DeviceSpan& span) const {
    for (const auto& [b, e] : written_[span.buffer])
      if (b <= span.offset && span.offset + span.count <= e) return true;
    return false;
  }

  std::vector<Ranges> written_;
  std::vector<DeviceSpan> cached_;
  std::vector<DeviceSpan> uncached_;
};

}  // namespace polima
