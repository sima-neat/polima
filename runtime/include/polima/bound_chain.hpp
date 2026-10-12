// A device-resident chain whose graphs bind named
// persistent MLA buffers, enabled only by `run_bound_chain` plan steps.
//
// Generalises DiTRunnerChain (dit_chain.hpp), which hard-codes four inputs
// (hidden, temb, backbone, mask) per DiT pair. Here every graph declares, in MLA order, the
// logical names of its persistent input and output bindings (bundle.json
// "persistent_bindings"), so the same runtime serves:
//   * KV-cached DiT: "prelude" graphs project the LM output to per-block cross-attention K/V
//     once per request; the block graphs read those K/V buffers directly on the device;
//   * larger partitions (4 DiT blocks per graph, two masks per graph);
//   * the original two-block layout (equivalence check against run_dit_chain).
// Host-fed bindings ("bind") are uploaded only when their plan buffer changed (pointer, version,
// or a new request), exactly like DiTRunnerChain's shared inputs. The carried state ("carry")
// is uploaded on every call and ping-pongs between two device buffers. Prelude graphs run when
// any of their host-fed inputs was uploaded during this call.
//
// Boundary work uses the shared kernels (runner.hpp/kernels.hpp): host-fed
// bindings are encoded straight into the device mapping, the carried output is
// decoded and checked for non-finite values in one pass, and the graphs of a
// call go to the MLA as queued submissions in which no ELF occurs twice
// (POLIMA_DISABLE_MLA_QUEUE=1 runs them one by one; POLIMA_PROFILE_STAGES=1
// times each graph).
//
// Binding order: the device runtime presents persistent bindings in
// lexicographic order of their binding NAMES (output_0, output_1, output_10,
// ..., output_2), which differs from numeric order beyond ten bindings. The
// bundle builder writes persistent_bindings in that order; Plan refuses a
// graph whose listed "binding" names are not in that order.
#pragma once
#include "polima/runner.hpp"

#include <fstream>
#include <map>
#include <set>

namespace polima {

struct BoundGraph {
  std::string name;
  std::filesystem::path elf;
  std::vector<std::pair<std::string, size_t>> inputs;   // logical name, elements (MLA order)
  std::vector<std::pair<std::string, size_t>> outputs;
};

class BoundRunnerChain {
 public:
  BoundRunnerChain(const std::string& name, const std::vector<std::string>& host_names,
                   const std::vector<size_t>& host_counts, const std::string& carry,
                   const std::vector<BoundGraph>& prelude, const std::vector<BoundGraph>& graphs)
      : host_names_(host_names), carry_(carry) {
    if (host_names.size() != host_counts.size() || graphs.empty())
      throw std::runtime_error(name + ": invalid bound chain geometry");
    std::map<std::string, size_t> sizes;
    const auto declare = [&](const std::string& logical, size_t count) {
      auto [it, inserted] = sizes.emplace(logical, count);
      if (!inserted && it->second != count)
        throw std::runtime_error(name + ": binding " + logical + " size disagrees");
    };
    size_t carry_index = host_names.size();
    for (size_t i = 0; i < host_names.size(); ++i) {
      declare(host_names[i], host_counts[i]);
      if (host_names[i] == carry) carry_index = i;
    }
    if (carry_index == host_names.size()) throw std::runtime_error(name + ": carry is not host-fed");
    carry_index_ = carry_index;
    const size_t carry_count = host_counts[carry_index];
    std::set<std::string> produced;
    for (const auto& g : prelude) {
      for (const auto& [n, c] : g.inputs) {
        if (std::find(host_names.begin(), host_names.end(), n) == host_names.end())
          throw std::runtime_error(g.name + ": prelude input " + n + " is not host-fed");
        declare(n, c);
      }
      for (const auto& [n, c] : g.outputs) { declare(n, c); produced.insert(n); }
    }
    for (const auto& g : graphs) {
      if (g.outputs.size() != 1 || g.outputs[0].second != carry_count)
        throw std::runtime_error(g.name + ": block graph must have one carry-sized output");
      bool has_carry = false;
      for (const auto& [n, c] : g.inputs) {
        declare(n, c);
        has_carry |= n == carry;
        if (n != carry && !produced.count(n) &&
            std::find(host_names.begin(), host_names.end(), n) == host_names.end())
          throw std::runtime_error(g.name + ": input " + n + " has no producer");
      }
      if (!has_carry) throw std::runtime_error(g.name + ": block graph does not read " + carry);
    }
    for (const auto& [n, c] : sizes)
      if (n != carry) buffers_[n] = std::make_unique<DeviceBuffer>(name + "_" + n, c);
    carry_in_ = std::make_unique<DeviceBuffer>(name + "_carry_in", carry_count);
    ping_ = std::make_unique<DeviceBuffer>(name + "_ping", carry_count);
    pong_ = std::make_unique<DeviceBuffer>(name + "_pong", carry_count);
    for (size_t i = 0; i < host_names.size(); ++i) encoded_.emplace_back(host_counts[i]);
    raw_.resize(carry_count);
    valid_.assign(host_names.size(), false);
    sources_.assign(host_names.size(), nullptr);
    versions_.assign(host_names.size(), 0);

    const auto slice = [](DeviceBuffer* b) { return llima::MLABufferSlice(&b->storage); };
    for (const auto& g : prelude) {
      std::vector<llima::MLABufferSlice> in, out;
      std::vector<size_t> feeds;
      for (const auto& [n, c] : g.inputs) {
        in.push_back(slice(buffers_.at(n).get()));
        feeds.push_back(std::find(host_names.begin(), host_names.end(), n) - host_names.begin());
      }
      for (const auto& [n, c] : g.outputs) out.push_back(slice(buffers_.at(n).get()));
      prelude_.push_back({std::make_unique<llima::MLAModelWithBuffer>(g.elf, in, out), feeds, g.name});
      prelude_.back().model->load();
      names_.push_back(g.name);
    }
    std::vector<std::filesystem::path> keys;
    for (const auto& g : prelude) keys.push_back(std::filesystem::weakly_canonical(g.elf));
    for (size_t i = 0; i < graphs.size(); ++i) {
      keys.push_back(std::filesystem::weakly_canonical(graphs[i].elf));
      DeviceBuffer* source = i == 0 ? carry_in_.get() : (i % 2 ? ping_.get() : pong_.get());
      DeviceBuffer* destination = i % 2 ? pong_.get() : ping_.get();
      std::vector<llima::MLABufferSlice> in;
      for (const auto& [n, c] : graphs[i].inputs)
        in.push_back(slice(n == carry ? source : buffers_.at(n).get()));
      models_.push_back(std::make_unique<llima::MLAModelWithBuffer>(
          graphs[i].elf, in, std::vector<llima::MLABufferSlice>{slice(destination)}));
      models_.back()->load();
      names_.push_back(graphs[i].name);
    }
    final_ = graphs.size() % 2 ? ping_.get() : pong_.get();
    keys_ = std::move(keys);
    // Without prelude graphs every call runs the same sequence.
    if (prelude_.empty()) block_segments_ = split_on_repeat(keys_);
  }

  ~BoundRunnerChain() {
    for (auto& p : prelude_) p.model->free();
    for (auto& m : models_) m->free();
  }

  void begin_request() { std::fill(valid_.begin(), valid_.end(), false); }

  void run(const std::vector<const float*>& inputs, const std::vector<size_t>& versions,
           float* output, RunTiming* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point t) {
      return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    if (timing) *timing = RunTiming{};
    std::vector<bool> refreshed(host_names_.size(), false);
    const bool staged = RuntimeOptions::get().staged_io;
    first_nonfinite_ = raw_.size();
    for (size_t i = 0; i < host_names_.size(); ++i) {
      if (i != carry_index_ && valid_[i] && sources_[i] == inputs[i] && versions_[i] == versions[i])
        continue;
      valid_[i] = false;  // stays invalid if the write is interrupted
      auto& storage = (i == carry_index_ ? carry_in_ : buffers_.at(host_names_[i]))->storage;
      auto t = Clock::now();
      detail::write_bf16(storage, encoded_[i], inputs[i], 0, encoded_[i].size(), false);
      if (timing) timing->encode_ms += ms(t);
      t = Clock::now();
      if (staged) storage.upload(encoded_[i].data());
      else storage.flush_cache();
      if (timing) timing->upload_ms += ms(t);
      valid_[i] = true; sources_[i] = inputs[i]; versions_[i] = versions[i]; refreshed[i] = true;
    }
    auto t = Clock::now();
    active_.clear();
    active_keys_.clear();
    std::vector<std::string> stage_names;
    for (size_t p = 0; p < prelude_.size(); ++p) {
      const auto& feeds = prelude_[p].feeds;
      if (!std::any_of(feeds.begin(), feeds.end(), [&](size_t f) { return refreshed[f]; })) continue;
      active_.push_back(prelude_[p].model.get());
      active_keys_.push_back(keys_[p]);
      if (timing) stage_names.push_back(prelude_[p].name);
    }
    for (size_t i = 0; i < models_.size(); ++i) {
      active_.push_back(models_[i].get());
      active_keys_.push_back(keys_[prelude_.size() + i]);
      if (timing) stage_names.push_back(names_[prelude_.size() + i]);
    }
    const bool per_stage = timing && std::getenv("POLIMA_PROFILE_STAGES");
    if (RuntimeOptions::get().mla_queue && !per_stage) {
      run_queued(active_.data(), prelude_.empty() ? block_segments_ : split_on_repeat(active_keys_));
    } else {
      for (size_t i = 0; i < active_.size(); ++i) {
        const auto s = Clock::now();
        active_[i]->run();
        if (timing) timing->stage_execute_ms.push_back(ms(s));
      }
    }
    if (timing) {
      timing->stage_names = std::move(stage_names);
      timing->execute_ms = ms(t);
    }
    // Diagnostics only: POLIMA_BOUND_DUMP=<dir> writes every named device buffer (raw BF16) after
    // each call, numbered by call, so bindings can be compared with host references.
    if (const char* dump = std::getenv("POLIMA_BOUND_DUMP")) {
      static size_t call = 0;
      const std::filesystem::path dir = std::filesystem::path(dump) / std::to_string(call++);
      std::filesystem::create_directories(dir);
      for (const auto& [n, b] : buffers_) {
        std::vector<uint16_t> raw(b->storage.get_num_elems());
        b->storage.download(raw.data());
        std::ofstream(dir / (n + ".bf16"), std::ios::binary).write(
            reinterpret_cast<const char*>(raw.data()), raw.size() * 2);
      }
    }
    t = Clock::now();
    if (staged) final_->storage.download(raw_.data());
    else final_->storage.invalidate_cache();
    if (timing) timing->download_ms = ms(t);
    t = Clock::now();
    const auto* raw = staged ? raw_.data()
                             : static_cast<const uint16_t*>(final_->storage.get_virtual_addr());
    first_nonfinite_ = kernels::bf16_to_f32_checked(raw, output, raw_.size());
    if (timing) timing->decode_ms = ms(t);
  }

  // The decode verifies every output value; output_elements() means all finite.
  size_t first_nonfinite() const { return first_nonfinite_; }
  size_t output_elements() const { return raw_.size(); }

 private:
  struct DeviceBuffer {
    DeviceBuffer(const std::string& name, size_t count)
        : storage(name, {1, count}, "bfloat16", false) { storage.allocate(); storage.clear(); }
    ~DeviceBuffer() { storage.free(); }
    llima::MLABuffer storage;
  };
  struct Prelude {
    std::unique_ptr<llima::MLAModelWithBuffer> model;
    std::vector<size_t> feeds;
    std::string name;
  };
  std::vector<std::string> host_names_;
  std::string carry_;
  size_t carry_index_ = 0;
  std::map<std::string, std::unique_ptr<DeviceBuffer>> buffers_;
  std::unique_ptr<DeviceBuffer> carry_in_, ping_, pong_;
  DeviceBuffer* final_ = nullptr;
  std::vector<std::vector<uint16_t>> encoded_;
  std::vector<uint16_t> raw_;
  std::vector<bool> valid_;
  std::vector<const float*> sources_;
  std::vector<size_t> versions_;
  std::vector<Prelude> prelude_;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models_;
  std::vector<std::string> names_;
  std::vector<std::filesystem::path> keys_;  // canonical ELF path per prelude + block graph
  std::vector<size_t> block_segments_;
  std::vector<llima::MLAModelWithBuffer*> active_;
  std::vector<std::filesystem::path> active_keys_;
  size_t first_nonfinite_ = 0;
};

}  // namespace polima
