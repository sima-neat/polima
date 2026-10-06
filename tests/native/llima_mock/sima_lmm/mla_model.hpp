// CPU test double of MLAModelWithBuffer. Each "ELF" is a deterministic,
// file-name-keyed function of all bytes in its input bindings (read in binding
// order, exactly as a packed or multi-binding ELF sees them), written as BF16
// to its output binding. Two plans that bind the same bytes therefore produce
// byte-identical results, and any stale or misplaced byte changes them.
// The binding length is what the real runtime passes as ifm_len/ofm_len, so a
// packed input bound as a whole buffer includes its 16-byte row padding.
#pragma once
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>

#include "mla_buffer.hpp"
#include "polima/bf16.hpp"

namespace simaai::llima {

inline void connect_mla_rt(const std::vector<std::string>&) {}
inline void disconnect_mla_rt() {}

class MLAModelWithBuffer {
 public:
  MLAModelWithBuffer(std::filesystem::path path, std::vector<MLABufferSlice> ifms,
                     std::vector<MLABufferSlice> ofms)
      : seed_(std::hash<std::string>{}(path.filename().string())), ifms_(ifms), ofms_(ofms) {}
  void load() {}
  void free() {}
  void run() {
    ++mock_counters()["mla_runs"];
    std::vector<uint16_t> in;
    for (const auto& slice : ifms_) {
      const size_t n = slice.length() / 2;
      const auto* p = reinterpret_cast<const uint16_t*>(slice.address());
      in.insert(in.end(), p, p + n);
    }
    double total = 0.0;
    for (size_t i = 0; i < in.size(); ++i) total += polima::from_bf16(in[i]) * double((i % 97) + 1);
    const float global = static_cast<float>(std::fmod(total, 3.0));
    const float bias = static_cast<float>(seed_ % 1000) / 1000.0f;
    auto& out = ofms_.at(0);
    const size_t n = out.length() / 2;
    auto* o = reinterpret_cast<uint16_t*>(out.address());
    for (size_t j = 0; j < n; ++j) {
      const float a = polima::from_bf16(in[(j * 7 + seed_) % in.size()]);
      const float b = polima::from_bf16(in[(j * 131 + 3) % in.size()]);
      o[j] = polima::to_bf16(std::tanh(0.7f * a - 0.3f * b + 0.05f * global + bias) * 2.0f);
    }
  }
  // The installed runtime shares one model handle per ELF path and gives a
  // wrong result when one ELF is queued twice in a single submission
  // (queue_dup.cpp, devkit 2026-10-04). The double refuses it outright, so any
  // plan path that would do it fails here instead of passing silently.
  void add_to_queue() {
    for (const auto* queued : queue())
      if (queued->seed_ == seed_) {
        queue().clear();
        throw std::runtime_error("mock llima: one ELF queued twice in one submission");
      }
    queue().push_back(this);
  }
  static void run_queue() {
    ++mock_counters()["queue_submissions"];
    for (auto* model : queue()) {
      ++mock_counters()["queued_runs"];
      model->run();
    }
    queue().clear();
  }

 private:
  static std::vector<MLAModelWithBuffer*>& queue() {
    static std::vector<MLAModelWithBuffer*> q;
    return q;
  }
  size_t seed_;
  std::vector<MLABufferSlice> ifms_, ofms_;
};

}  // namespace simaai::llima
