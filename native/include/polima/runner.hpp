// One compiled graph on the MLA.
//
// Generalizes the `Runner` struct that act_llima.cpp, smolvla_som_server.cpp and
// smolvla_action_llima.cpp each define separately. The difference here is that
// shapes come from bundle.json rather than being baked into a constructor
// initializer list (act_llima.cpp:117-122), which is what lets one binary serve
// every bundle.
//
// Boundary I/O (2026-10-04 optimisation, byte-identical to the earlier path):
//
//   * Inputs are converted straight into the MLA buffer's CPU mapping and the
//     cache is flushed, instead of converting into a host staging vector and
//     having MLABuffer::upload copy it. Outputs are read straight from the
//     mapping after a cache invalidate. Measured on the devkit for a 360k
//     value boundary: 0.55 -> 0.46 ms in, 0.34 -> 0.24 ms out, identical bytes.
//   * The bf16 decode also reports the first non-finite value, so Plan no
//     longer needs a second pass over every graph output (3.6 ns/value).
//   * An input part whose plan buffer is unchanged since this graph last
//     uploaded it (same reuse token) is not converted or written again. GR00T's
//     non-persistent DiT graphs see the same LM output on all four denoising
//     steps; W8A8 builds additionally share one quantization of it through
//     QuantizedInputCache.
//
// Diagnostic switches (read once): POLIMA_STAGED_IO=1 restores the staging
// copies; POLIMA_DISABLE_INPUT_REUSE=1 converts every part on every call;
// POLIMA_DISABLE_MLA_QUEUE=1 submits chain and device-program stages one run()
// at a time.
#pragma once

#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <thread>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "polima/bf16.hpp"
#include "polima/kernels.hpp"
#include "polima/quantization.hpp"
#include "polima/tess.hpp"

namespace polima {

namespace llima = simaai::llima;

// The MLA runtime connection, shared by every Plan in the process.
//
// It used to be one guard per Plan, which was wrong the moment a second Plan
// could exist: two connects, and then the FIRST plan destroyed would
// disconnect the runtime out from under the one still using it. The interactive
// session makes that reachable -- `use` a second model and the first one's
// teardown pulls the runtime.
//
// So it is refcounted: connect on the first Plan, disconnect when the last one
// goes away. `acquire_mla_runtime()` is the only way to get one.
struct RuntimeGuard {
  RuntimeGuard() { llima::connect_mla_rt({}); }
  ~RuntimeGuard() { llima::disconnect_mla_rt(); }
  RuntimeGuard(const RuntimeGuard&) = delete;
  RuntimeGuard& operator=(const RuntimeGuard&) = delete;
};

std::shared_ptr<RuntimeGuard> acquire_mla_runtime();

struct RunTiming {
  double encode_ms = 0, upload_ms = 0, execute_ms = 0, download_ms = 0, decode_ms = 0;
  std::vector<double> stage_execute_ms;
  std::vector<std::string> stage_names;
};

struct RuntimeOptions {
  bool staged_io = false;
  bool input_reuse = true;
  bool mla_queue = true;

  static const RuntimeOptions& get() {
    static const RuntimeOptions options = [] {
      const auto flag = [](const char* name) {
        const char* value = std::getenv(name);
        return value && *value && std::string(value) != "0";
      };
      RuntimeOptions o;
      o.staged_io = flag("POLIMA_STAGED_IO");
      o.input_reuse = !flag("POLIMA_DISABLE_INPUT_REUSE");
      o.mla_queue = !flag("POLIMA_DISABLE_MLA_QUEUE");
      return o;
    }();
    return options;
  }
};

// One input tensor of a graph call. `token` identifies the plan buffer's
// current contents: equal non-zero tokens mean identical values. Zero means
// unknown, which always converts.
struct InputPart {
  const float* data = nullptr;
  size_t count = 0;
  uint64_t token = 0;
  // Read by several graph steps: worth keeping one quantized copy of.
  bool shared = false;
};

// Quantized copies of plan buffers, shared by every INT8 graph in a Plan. A
// W8A8 GR00T DiT has 16 graphs x 4 steps that all quantize the same LM output
// with the same scale; this does it once per request instead of 64 times.
// Entries are valid only for the token they were made from.
class QuantizedInputCache {
 public:
  const int8_t* get(const InputPart& part, const TensorQuantization& q) {
    if (!part.token) return nullptr;
    const Key key{part.token, part.data, q.multiplier, q.zero_point, q.elements};
    for (auto& slot : slots_)
      if (slot.valid && slot.key == key) return slot.codes.data();
    // Reuse slot storage: a fresh allocation per request costs page faults.
    Slot* slot = nullptr;
    for (auto& candidate : slots_)
      if (!candidate.valid) { slot = &candidate; break; }
    if (!slot) {
      if (slots_.size() < 8) slot = &slots_.emplace_back();
      else slot = &slots_[next_victim_++ % slots_.size()];
    }
    slot->valid = false;
    slot->codes.resize(q.elements);
    q.quantize_span(part.data, slot->codes.data(), q.elements);
    slot->key = key;
    slot->valid = true;
    return slot->codes.data();
  }
  // Called at the start of every request; tokens never repeat, so this only
  // lets slots be reused.
  void clear() {
    for (auto& slot : slots_) slot.valid = false;
  }

 private:
  using Key = std::tuple<uint64_t, const float*, float, int, size_t>;
  struct Slot {
    Key key{};
    bool valid = false;
    std::vector<int8_t> codes;
  };
  std::deque<Slot> slots_;  // stable addresses
  size_t next_victim_ = 0;
};

namespace detail {

inline double since_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

// sima_lmm 0.4.0 (SDK 2.1.3) added byte-range MLABuffer::flush_cache/invalidate_cache;
// 0.3.0 (SDK 2.1.2, the robot DevKit) has only the whole-buffer forms. Every CPU
// write into an MLA buffer is flushed right after it is written, so no dirty line is
// outstanding when a whole-buffer invalidate runs: the fallback moves the same bytes.
template <class Buffer>
inline void flush_range(const Buffer& buffer, size_t offset, size_t size) {
  if constexpr (requires { buffer.flush_cache(offset, size); }) buffer.flush_cache(offset, size);
  else buffer.flush_cache();
}

template <class Buffer>
inline void invalidate_range(const Buffer& buffer, size_t offset, size_t size) {
  if constexpr (requires { buffer.invalidate_cache(offset, size); })
    buffer.invalidate_cache(offset, size);
  else buffer.invalidate_cache();
}

// Converts `count` floats into bf16 directly inside an MLA buffer at element
// `offset` (or through MLABuffer::upload when staged I/O is requested).
inline void write_bf16(llima::MLABuffer& buffer, std::vector<uint16_t>& staging,
                       const float* source, size_t offset, size_t count, bool flush = true) {
  if (RuntimeOptions::get().staged_io) {
    to_bf16_span(source, staging.data() + offset, count);
    return;  // caller uploads the whole staging vector
  }
  auto* device = static_cast<uint16_t*>(buffer.get_virtual_addr()) + offset;
  kernels::f32_to_bf16(source, device, count);
  if (flush) flush_range(buffer, offset * sizeof(uint16_t), count * sizeof(uint16_t));
}

// Reads a plain bf16 MLA buffer into floats; returns the first non-finite
// index or `count`.
inline size_t read_bf16_checked(llima::MLABuffer& buffer, std::vector<uint16_t>& staging,
                                float* destination, size_t count) {
  if (RuntimeOptions::get().staged_io) {
    buffer.download(staging.data());
    return kernels::bf16_to_f32_checked(staging.data(), destination, count);
  }
  buffer.invalidate_cache();
  return kernels::bf16_to_f32_checked(static_cast<const uint16_t*>(buffer.get_virtual_addr()),
                                      destination, count);
}

}  // namespace detail

class Runner {
 public:
  Runner(std::string name, const std::filesystem::path& elf, size_t input_elements,
         size_t output_elements, DramLayout output_layout = DramLayout::Plain,
         size_t logical_width = 0, size_t logical_channels = 0,
         std::vector<TensorQuantization> input_quantization = {},
         TensorQuantization output_quantization = {})
      : name_(std::move(name)),
        input_elements_(input_elements),
        output_elements_(output_elements),
        output_layout_(output_layout),
        logical_width_(logical_width),
        logical_channels_(logical_channels),
        staged_input_(input_elements),
        staged_output_(output_elements),
        input_quantization_(std::move(input_quantization)),
        output_quantization_(output_quantization),
        quantized_(!input_quantization_.empty()),
        int8_input_(quantized_ ? input_elements : 0),
        int8_output_(quantized_ ? output_elements : 0),
        input_(name_ + "_ifm", {1, input_elements}, quantized_ ? "int8" : "bfloat16", false),
        output_(name_ + "_ofm", {1, output_elements}, quantized_ ? "int8" : "bfloat16", false),
        model_(elf, {llima::MLABufferSlice(&input_)}, {llima::MLABufferSlice(&output_)}) {
    if (quantized_) {
      size_t count = 0;
      for (const auto& q : input_quantization_) { q.validate(); count += q.elements; }
      output_quantization_.validate();
      if (count != input_elements_ || output_quantization_.elements != output_elements_ ||
          output_layout_ != DramLayout::Plain)
        throw std::runtime_error(name_ + " unsupported INT8 boundary contract");
      dequantize_table_ = output_quantization_.dequantize_table();
      dequantize_finite_ = table_is_finite(dequantize_table_);
    }
    input_.allocate();
    output_.allocate();
    input_.clear();
    output_.clear();
    model_.load();
  }

  ~Runner() {
    model_.free();
    input_.free();
    output_.free();
  }

  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // Single contiguous input (kept for callers that join their own buffers).
  void run(const float* input, float* output, RunTiming* timing = nullptr,
           const int8_t* raw_first_input = nullptr) {
    const std::vector<InputPart> parts{{input, input_elements_, 0}};
    run(parts, output, timing, raw_first_input);
  }

  // No allocation on this path for BF16 graphs: buffers are sized once in the
  // constructor. `parts` must cover the graph input exactly, in order.
  void run(const std::vector<InputPart>& parts, float* output, RunTiming* timing = nullptr,
           const int8_t* raw_first_input = nullptr, QuantizedInputCache* cache = nullptr) {
    using Clock = std::chrono::steady_clock;
    auto stamp = timing ? Clock::now() : Clock::time_point{};
    auto mark = [&](double RunTiming::*field) {
      if (timing) {
        const auto now = Clock::now();
        timing->*field = std::chrono::duration<double, std::milli>(now - stamp).count();
        stamp = now;
      }
    };
    encode(parts, raw_first_input, cache);
    mark(&RunTiming::encode_ms);
    upload();
    mark(&RunTiming::upload_ms);
    execute();
    mark(&RunTiming::execute_ms);
    download();
    mark(&RunTiming::download_ms);
    decode(output);
    mark(&RunTiming::decode_ms);
  }

  // The phases of run(), separately callable so that the MLA call can run on
  // another thread while this one prepares or reads a different graph
  // instance. encode + upload write the device input; execute is the only MLA
  // call; download + decode read the device output.
  void encode(const std::vector<InputPart>& parts, const int8_t* raw_first_input = nullptr,
              QuantizedInputCache* cache = nullptr) {
    size_t total = 0;
    for (const auto& part : parts) total += part.count;
    if (total != input_elements_)
      throw std::runtime_error(name_ + " packed input size disagrees with graph input");
    const auto& options = RuntimeOptions::get();
    if (resident_.size() != parts.size()) resident_.assign(parts.size(), Resident{});
    first_nonfinite_ = output_elements_;
    output_checked_ = false;
    if (quantized_) {
      encode_int8(parts, raw_first_input, cache);
      return;
    }
    size_t offset = 0;
    for (size_t index = 0; index < parts.size(); ++index) {
      const auto& part = parts[index];
      auto& resident = resident_[index];
      const bool reuse = options.input_reuse && !options.staged_io && part.token &&
                         resident.token == part.token && resident.data == part.data;
      if (!reuse) {
        resident = Resident{};  // stays invalid if the write is interrupted
        detail::write_bf16(input_, staged_input_, part.data, offset, part.count);
        resident = {part.token, part.data};
      }
      offset += part.count;
    }
  }

  void upload() {
    if (!RuntimeOptions::get().staged_io) return;
    if (quantized_) input_.upload(int8_input_.data());
    else input_.upload(staged_input_.data());
  }

  void execute() { model_.run(); }

  void download() {
    const bool staged = RuntimeOptions::get().staged_io;
    if (quantized_) {
      if (staged) output_.download(int8_output_.data());
      else output_.invalidate_cache();
    } else if (output_layout_ != DramLayout::Plain || staged) {
      output_.download(staged_output_.data());
    } else {
      output_.invalidate_cache();
    }
  }

  void decode(float* output) {
    const bool staged = RuntimeOptions::get().staged_io;
    if (quantized_) {
      const int8_t* codes = staged ? int8_output_.data()
                                   : static_cast<const int8_t*>(output_.get_virtual_addr());
      dequantize_span(output_quantization_, codes, output, output_elements_);
      output_checked_ = dequantize_finite_;
      return;
    }
    if (output_layout_ == DramLayout::Plain) {
      const auto* raw = staged ? staged_output_.data()
                               : static_cast<const uint16_t*>(output_.get_virtual_addr());
      first_nonfinite_ = kernels::bf16_to_f32_checked(raw, output, output_elements_);
      output_checked_ = true;
      return;
    }
    detessellate(output_layout_, staged_output_.data(), output_elements_, logical_width_,
                 logical_channels_, output);
    report_hwc16_layout(output);
  }

  // After run(): whether the decode already verified every output value, and
  // if so the first non-finite index (output_elements() when none).
  bool output_checked() const { return output_checked_; }
  size_t first_nonfinite() const { return first_nonfinite_; }

  bool accepts_float_parts() const { return true; }
  bool quantized() const { return quantized_; }
  const std::string& name() const { return name_; }
  size_t input_elements() const { return input_elements_; }
  size_t output_elements() const { return output_elements_; }
  const std::vector<TensorQuantization>& int8_input_contract() const { return input_quantization_; }
  // Raw codes of the last run, valid until this graph runs again.
  const int8_t* raw_int8_output() const {
    if (!quantized_) return nullptr;
    return RuntimeOptions::get().staged_io ? int8_output_.data()
                                           : static_cast<const int8_t*>(output_.get_virtual_addr());
  }
  const TensorQuantization& int8_output_contract() const { return output_quantization_; }
  bool can_reuse_first_int8_from(const Runner& producer) const {
    return quantized_ && producer.quantized_ && !input_quantization_.empty() &&
           input_quantization_.front().preserves_codes_from(producer.output_quantization_);
  }

 private:
  struct Resident {
    uint64_t token = 0;
    const float* data = nullptr;
  };

  // Quantizes every input segment straight into the MLA buffer. Parts must
  // line up with the quantization segments one to one; otherwise they are
  // joined first (the earlier behaviour).
  void encode_int8(const std::vector<InputPart>& parts, const int8_t* raw_first_input,
                   QuantizedInputCache* cache) {
    const auto& options = RuntimeOptions::get();
    int8_t* device = options.staged_io ? int8_input_.data()
                                       : static_cast<int8_t*>(input_.get_virtual_addr());
    bool aligned = parts.size() == input_quantization_.size();
    for (size_t i = 0; aligned && i < parts.size(); ++i)
      aligned = parts[i].count == input_quantization_[i].elements;
    if (!aligned) {
      joined_.resize(input_elements_);
      size_t offset = 0;
      for (const auto& part : parts) {
        std::copy_n(part.data, part.count, joined_.begin() + offset);
        offset += part.count;
      }
      offset = 0;
      for (size_t i = 0; i < input_quantization_.size(); ++i) {
        const auto& q = input_quantization_[i];
        if (i == 0 && raw_first_input) std::copy_n(raw_first_input, q.elements, device);
        else q.quantize_span(joined_.data() + offset, device + offset, q.elements);
        offset += q.elements;
      }
      std::fill(resident_.begin(), resident_.end(), Resident{});
      if (!options.staged_io) input_.flush_cache();
      return;
    }
    size_t offset = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
      const auto& q = input_quantization_[i];
      const auto& part = parts[i];
      auto& resident = resident_[i];
      bool written = true;
      if (i == 0 && raw_first_input) {
        std::copy_n(raw_first_input, q.elements, device + offset);
        resident = Resident{};
      } else if (options.input_reuse && !options.staged_io && part.token &&
                 resident.token == part.token && resident.data == part.data) {
        written = false;
      } else {
        resident = Resident{};  // stays invalid if quantizing throws
        const int8_t* cached =
            options.input_reuse && cache && part.shared ? cache->get(part, q) : nullptr;
        if (cached) kernels::copy_bytes(device + offset, cached, q.elements);
        else q.quantize_span(part.data, device + offset, q.elements);
        resident = {part.token, part.data};
      }
      if (written && !options.staged_io) detail::flush_range(input_, offset, q.elements);
      offset += q.elements;
    }
  }

  // A bare MLA ELF omits the MPK detessellation plugin. If an HWC16
  // result is non-finite, report both possible interpretations of the same
  // downloaded bytes. This leaves the declared output untouched; Plan still
  // rejects it through the existing finite check.
  void report_hwc16_layout(const float* output) const {
    const char* layout_diagnostics = std::getenv("POLIMA_LAYOUT_DIAGNOSTICS");
    if (!(layout_diagnostics && *layout_diagnostics && output_layout_ == DramLayout::Hwc16))
      return;
    size_t decoded_bad = output_elements_;
    size_t plain_bad = output_elements_;
    size_t decoded_bad_count = 0;
    size_t plain_bad_count = 0;
    float decoded_min = std::numeric_limits<float>::infinity();
    float decoded_max = -std::numeric_limits<float>::infinity();
    float plain_min = std::numeric_limits<float>::infinity();
    float plain_max = -std::numeric_limits<float>::infinity();
    double decoded_abs_sum = 0.0;
    double plain_abs_sum = 0.0;
    for (size_t index = 0; index < output_elements_; ++index) {
      const float decoded = output[index];
      const float plain = from_bf16(staged_output_[index]);
      if (!std::isfinite(decoded)) {
        if (decoded_bad == output_elements_) decoded_bad = index;
        ++decoded_bad_count;
      } else {
        decoded_min = std::min(decoded_min, decoded);
        decoded_max = std::max(decoded_max, decoded);
        decoded_abs_sum += std::abs(decoded);
      }
      if (!std::isfinite(plain)) {
        if (plain_bad == output_elements_) plain_bad = index;
        ++plain_bad_count;
      } else {
        plain_min = std::min(plain_min, plain);
        plain_max = std::max(plain_max, plain);
        plain_abs_sum += std::abs(plain);
      }
    }
    if (decoded_bad != output_elements_) {
      std::cerr << "layout diagnostic: graph " << name_
                << " HWC16 first_nonfinite=" << decoded_bad
                << " count=" << decoded_bad_count
                << " min=" << decoded_min << " max=" << decoded_max
                << " mean_abs="
                << decoded_abs_sum /
                       static_cast<double>(
                           std::max<size_t>(1, output_elements_ - decoded_bad_count))
                << "; plain_bf16 first_nonfinite=";
      if (plain_bad == output_elements_)
        std::cerr << "none";
      else
        std::cerr << plain_bad;
      std::cerr << " count=" << plain_bad_count
                << " min=" << plain_min << " max=" << plain_max
                << " mean_abs="
                << plain_abs_sum /
                       static_cast<double>(
                           std::max<size_t>(1, output_elements_ - plain_bad_count))
                << std::endl;
    }
  }

  std::string name_;
  size_t input_elements_;
  size_t output_elements_;
  DramLayout output_layout_;
  size_t logical_width_;
  size_t logical_channels_;
  std::vector<uint16_t> staged_input_;
  std::vector<uint16_t> staged_output_;
  std::vector<TensorQuantization> input_quantization_;
  TensorQuantization output_quantization_;
  bool quantized_ = false;
  std::vector<int8_t> int8_input_;
  std::vector<int8_t> int8_output_;
  std::vector<float> joined_;
  std::array<float, 256> dequantize_table_{};
  bool dequantize_finite_ = false;
  std::vector<Resident> resident_;
  bool output_checked_ = false;
  size_t first_nonfinite_ = 0;
  llima::MLABuffer input_;
  llima::MLABuffer output_;
  llima::MLAModelWithBuffer model_;
};

// Splits a sequence of graph calls into MLA queue submissions in which no ELF
// occurs twice; returns the end index of each submission.
//
// llima shares one model handle per ELF path. Queuing the same ELF twice in
// one add_to_queue/run_queue submission (with different buffers) gives the
// FIRST instance a wrong output from the second submission on on the
// installed runtime (sima-lmm 0.4.0 / MLASHM), while run() is correct
// (devkit, 2026-10-04, queue_dup.cpp: eagle_vision_connector and
// state_project). That is what broke the N1.6 SO-101 device program with two
// connector calls. Separate submissions of the same ELF are fine (the two
// camera vision chains, 30 requests byte-identical), so a repeated ELF starts
// a new submission instead of disabling the queue.
template <class Key>
std::vector<size_t> split_on_repeat(const std::vector<Key>& keys) {
  std::vector<size_t> ends;
  std::vector<const Key*> open;
  for (size_t index = 0; index < keys.size(); ++index) {
    for (const Key* seen : open)
      if (*seen == keys[index]) {
        ends.push_back(index);
        open.clear();
        break;
      }
    open.push_back(&keys[index]);
  }
  if (!keys.empty()) ends.push_back(keys.size());
  return ends;
}

inline std::vector<size_t> queue_segments(const std::vector<std::filesystem::path>& elfs) {
  std::vector<std::filesystem::path> keys;
  for (const auto& elf : elfs) keys.push_back(std::filesystem::weakly_canonical(elf));
  return split_on_repeat(keys);
}

// Submits `models` (in order) as queued MLA submissions split by `segments`.
inline void run_queued(llima::MLAModelWithBuffer* const* models, const std::vector<size_t>& segments) {
  size_t begin = 0;
  for (const size_t end : segments) {
    for (size_t index = begin; index < end; ++index) models[index]->add_to_queue();
    llima::MLAModelWithBuffer::run_queue();
    begin = end;
  }
}

// Runs every model in `models` in order: queued submissions to the MLA
// runtime (llima add_to_queue / run_queue, split by `segments` from
// queue_segments) instead of one round trip per graph, unless disabled
// (POLIMA_DISABLE_MLA_QUEUE=1) or per-stage timing is requested.
inline void run_models(std::vector<std::unique_ptr<llima::MLAModelWithBuffer>>& models,
                       RunTiming* timing, const std::vector<std::string>* names,
                       const std::vector<size_t>& segments) {
  using Clock = std::chrono::steady_clock;
  const bool per_stage = timing && std::getenv("POLIMA_PROFILE_STAGES");
  if (RuntimeOptions::get().mla_queue && !per_stage) {
    std::vector<llima::MLAModelWithBuffer*> raw;
    raw.reserve(models.size());
    for (auto& model : models) raw.push_back(model.get());
    run_queued(raw.data(), segments);
    return;
  }
  if (timing && names) timing->stage_names = *names;
  for (auto& model : models) {
    const auto started = timing ? Clock::now() : Clock::time_point{};
    model->run();
    if (timing) timing->stage_execute_ms.push_back(detail::since_ms(started));
  }
}

// One thread that makes MLA calls in submission order, so the plan's own
// thread can convert another graph's data meanwhile. The MLA still runs one
// graph at a time; only the CPU work around it overlaps. Measured on the
// devkit: a 2.5 ms graph run here while the caller converts 2.3 ms of data
// takes 2.8 ms in all.
class MlaWorker {
 public:
  MlaWorker() : thread_([this] { loop(); }) {}
  ~MlaWorker() {
    {
      const std::lock_guard<std::mutex> held(lock_);
      stop_ = true;
    }
    pending_.fetch_add(1);
    pending_.notify_all();
    thread_.join();
  }
  MlaWorker(const MlaWorker&) = delete;
  MlaWorker& operator=(const MlaWorker&) = delete;

  uint64_t submit(std::function<void()> job) {
    uint64_t ticket;
    {
      const std::lock_guard<std::mutex> held(lock_);
      ticket = ++submitted_;
      jobs_.push_back({ticket, std::move(job)});
    }
    pending_.fetch_add(1, std::memory_order_release);
    pending_.notify_one();
    return ticket;
  }

  // Blocks until `ticket` has run; rethrows its exception, if any.
  void wait(uint64_t ticket) {
    for (int spin = 0; completed_.load(std::memory_order_acquire) < ticket; ++spin) {
      if (spin < 4000) {
#if defined(__aarch64__)
        asm volatile("yield");
#endif
        continue;
      }
      const uint64_t seen = completed_.load(std::memory_order_acquire);
      if (seen < ticket) completed_.wait(seen, std::memory_order_acquire);
    }
    const std::lock_guard<std::mutex> held(lock_);
    auto failed = errors_.find(ticket);
    if (failed != errors_.end()) {
      auto error = failed->second;
      errors_.erase(failed);
      std::rethrow_exception(error);
    }
  }

 private:
  void loop() {
    uint64_t seen = 0;
    for (;;) {
      std::pair<uint64_t, std::function<void()>> job;
      bool have = false;
      {
        const std::lock_guard<std::mutex> held(lock_);
        if (stop_) return;
        if (!jobs_.empty()) {
          job = std::move(jobs_.front());
          jobs_.pop_front();
          have = true;
        }
      }
      if (!have) {
        // Jobs arrive back to back inside one request: spin briefly first.
        bool arrived = false;
        for (int spin = 0; spin < 4000 && !arrived; ++spin) {
          arrived = pending_.load(std::memory_order_acquire) != seen;
#if defined(__aarch64__)
          if (!arrived) asm volatile("yield");
#endif
        }
        if (!arrived) pending_.wait(seen, std::memory_order_acquire);
        seen = pending_.load(std::memory_order_acquire);
        continue;
      }
      try {
        job.second();
      } catch (...) {
        const std::lock_guard<std::mutex> held(lock_);
        errors_[job.first] = std::current_exception();
      }
      completed_.store(job.first, std::memory_order_release);
      completed_.notify_all();
    }
  }

  std::mutex lock_;
  std::deque<std::pair<uint64_t, std::function<void()>>> jobs_;
  std::map<uint64_t, std::exception_ptr> errors_;
  uint64_t submitted_ = 0;
  bool stop_ = false;
  std::atomic<uint64_t> pending_{0};
  std::atomic<uint64_t> completed_{0};
  std::thread thread_;
};

struct SharedStage {
  std::string name;
  std::filesystem::path elf;
  size_t input_elements = 0;
  size_t output_elements = 0;
};

inline std::vector<size_t> queue_segments(const std::vector<SharedStage>& stages) {
  std::vector<std::filesystem::path> elfs;
  for (const auto& stage : stages) elfs.push_back(stage.elf);
  return queue_segments(elfs);
}

// A linear sequence of HWC-compiled ELFs with device-resident intermediates.
// Only the first input is uploaded and only the final output is downloaded.
// Intermediate stages alternate between two MLABuffers, matching the
// ping-pong arrangement validated for GR00T EAGLE on ModaliX.
class SharedRunnerChain {
 public:
  SharedRunnerChain(std::string name, const std::vector<SharedStage>& stages,
                    DramLayout output_layout = DramLayout::Plain,
                    size_t logical_width = 0, size_t logical_channels = 0)
      : name_(std::move(name)), output_layout_(output_layout),
        logical_width_(logical_width), logical_channels_(logical_channels) {
    if (stages.size() < 2)
      throw std::runtime_error(name_ + " shared chain needs at least two stages");
    input_elements_ = stages.front().input_elements;
    output_elements_ = stages.back().output_elements;
    const size_t shared_elements = stages.front().output_elements;
    for (size_t index = 1; index < stages.size(); ++index) {
      if (stages[index].input_elements != shared_elements ||
          stages[index].output_elements != shared_elements)
        throw std::runtime_error(name_ + " stage " + stages[index].name +
                                 " does not match the shared buffer size");
    }
    if (output_elements_ != shared_elements)
      throw std::runtime_error(name_ + " final output does not match its shared buffer");

    staged_input_.resize(input_elements_);
    staged_output_.resize(output_elements_);
    input_ = std::make_unique<DeviceBuffer>(name_ + "_ifm", input_elements_);
    ping_ = std::make_unique<DeviceBuffer>(name_ + "_ping", shared_elements);
    pong_ = std::make_unique<DeviceBuffer>(name_ + "_pong", shared_elements);
    for (size_t index = 0; index < stages.size(); ++index) {
      stage_names_.push_back(stages[index].name);
      auto* source = index == 0 ? input_.get() : (index % 2 ? ping_.get() : pong_.get());
      auto* destination = index % 2 ? pong_.get() : ping_.get();
      models_.push_back(std::make_unique<llima::MLAModelWithBuffer>(
          stages[index].elf, std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(&source->storage)},
          std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(&destination->storage)}));
      models_.back()->load();
    }
    final_ = stages.size() % 2 ? ping_.get() : pong_.get();
    segments_ = queue_segments(stages);
  }

  ~SharedRunnerChain() {
    for (auto& model : models_) model->free();
    models_.clear();
  }

  SharedRunnerChain(const SharedRunnerChain&) = delete;
  SharedRunnerChain& operator=(const SharedRunnerChain&) = delete;

  void run(const float* input, float* output, RunTiming* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    auto stamp = timing ? Clock::now() : Clock::time_point{};
    auto mark = [&](double RunTiming::*field) {
      if (timing) {
        const auto now = Clock::now();
        timing->*field = std::chrono::duration<double, std::milli>(now - stamp).count();
        stamp = now;
      }
    };
    encode(input);
    mark(&RunTiming::encode_ms);
    upload();
    mark(&RunTiming::upload_ms);
    execute(timing);
    mark(&RunTiming::execute_ms);
    download();
    mark(&RunTiming::download_ms);
    decode(output);
    mark(&RunTiming::decode_ms);
  }

  // Phases of run(); see Runner. execute() is the only MLA work.
  void encode(const float* input) {
    output_checked_ = false;
    first_nonfinite_ = output_elements_;
    detail::write_bf16(input_->storage, staged_input_, input, 0, input_elements_);
  }
  void upload() {
    if (RuntimeOptions::get().staged_io) input_->storage.upload(staged_input_.data());
  }
  void execute(RunTiming* timing = nullptr) {
    run_models(models_, timing, &stage_names_, segments_);
  }
  void download() {
    if (output_layout_ != DramLayout::Plain || RuntimeOptions::get().staged_io)
      final_->storage.download(staged_output_.data());
    else
      final_->storage.invalidate_cache();
  }
  void decode(float* output) {
    if (output_layout_ == DramLayout::Plain) {
      const auto* raw = RuntimeOptions::get().staged_io
                            ? staged_output_.data()
                            : static_cast<const uint16_t*>(final_->storage.get_virtual_addr());
      first_nonfinite_ = kernels::bf16_to_f32_checked(raw, output, output_elements_);
      output_checked_ = true;
      return;
    }
    detessellate(output_layout_, staged_output_.data(), output_elements_, logical_width_,
                 logical_channels_, output);
  }

  bool output_checked() const { return output_checked_; }
  size_t first_nonfinite() const { return first_nonfinite_; }
  size_t input_elements() const { return input_elements_; }
  size_t output_elements() const { return output_elements_; }

 private:
  struct DeviceBuffer {
    DeviceBuffer(const std::string& name, size_t elements)
        : storage(name, {1, elements}, "bfloat16", false) {
      storage.allocate();
      storage.clear();
    }
    ~DeviceBuffer() { storage.free(); }
    llima::MLABuffer storage;
  };

  std::string name_;
  size_t input_elements_ = 0;
  size_t output_elements_ = 0;
  DramLayout output_layout_;
  size_t logical_width_;
  size_t logical_channels_;
  std::vector<uint16_t> staged_input_;
  std::vector<uint16_t> staged_output_;
  std::unique_ptr<DeviceBuffer> input_;
  std::unique_ptr<DeviceBuffer> ping_;
  std::unique_ptr<DeviceBuffer> pong_;
  DeviceBuffer* final_ = nullptr;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models_;
  std::vector<std::string> stage_names_;
  std::vector<size_t> segments_;
  bool output_checked_ = false;
  size_t first_nonfinite_ = 0;
};

}  // namespace polima
