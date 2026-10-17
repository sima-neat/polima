// Boundary-conversion kernels for the plan interpreter.
//
// Every kernel here has a scalar reference elsewhere (bf16.hpp, quantization.hpp,
// Plan's finite check) and must produce the same bytes. They differ only in
// speed:
//
//   * NEON bodies on aarch64 (the Modalix A65 cluster has no BF16 or I8MM
//     instructions, so bf16 rounding stays integer arithmetic, exactly the
//     scalar formula applied four lanes at a time);
//   * the finite check folded into the bf16 -> float decode, which reads every
//     value anyway, instead of a second pass over the decoded floats;
//   * a persistent worker pool for large tensors. Work is split into chunks
//     whose boundaries are multiples of 16 values, so a kernel whose NEON body
//     and scalar tail differ (INT8 quantize) still treats every index the way
//     the single-threaded call does.
//
// Measured on the devkit before this file existed (cpu_ops_bench.cpp): scalar
// finite check 3.6 ns/value against 0.6 ns with NEON; the BF16 encode was
// already auto-vectorized by -O3 and is memory-bound.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "polima/bf16.hpp"

namespace polima {
namespace kernels {

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

// A fixed set of threads that sleep between requests. The caller always takes
// a share of the work itself, so a pool of size 1 is plain single-threaded
// execution with no synchronisation at all.
//
// Size: POLIMA_CPU_THREADS (total participants including the caller). Unset
// means the measured default below; 1 disables threading.
class WorkerPool {
 public:
  static WorkerPool& instance() {
    static WorkerPool pool(configured_size());
    return pool;
  }

  static size_t configured_size() {
    const char* text = std::getenv("POLIMA_CPU_THREADS");
    if (text && *text) {
      const long value = std::strtol(text, nullptr, 10);
      if (value >= 1 && value <= 16) return static_cast<size_t>(value);
    }
    return kDefaultThreads;
  }

  // Four participants: measured best for 0.3-0.6 M value boundaries on the
  // A65 (memory-bound beyond that). One per physical core would be 8.
  static constexpr size_t kDefaultThreads = 4;

  size_t size() const { return workers_.size() + 1; }

  // Calls task(i) for every i in [0, tasks). Returns after all have finished.
  // The first exception thrown by any task is rethrown here.
  template <class F>
  void run(size_t tasks, F&& task) {
    if (tasks == 0) return;
    // A task that itself asks for parallel work runs it inline.
    if (tasks == 1 || workers_.empty() || inside_pool()) {
      for (size_t i = 0; i < tasks; ++i) task(i);
      return;
    }
    const std::lock_guard<std::mutex> serial(submit_);
    struct Inside {
      Inside() { inside_pool() = true; }
      ~Inside() { inside_pool() = false; }
    } inside;
    using Fn = std::remove_reference_t<F>;
    context_ = static_cast<void*>(&task);
    invoke_ = [](void* context, size_t index) { (*static_cast<Fn*>(context))(index); };
    tasks_ = tasks;
    next_.store(0, std::memory_order_relaxed);
    error_ = nullptr;
    pending_.store(workers_.size(), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    work();
    // Workers finish their last task quickly; spin rather than sleep.
    while (pending_.load(std::memory_order_acquire) != 0) {
#if defined(__aarch64__)
      asm volatile("yield");
#endif
    }
    if (error_) std::rethrow_exception(error_);
  }

  ~WorkerPool() {
    stop_.store(true, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    for (auto& thread : workers_) thread.join();
  }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

 private:
  static bool& inside_pool() {
    thread_local bool inside = false;
    return inside;
  }

  explicit WorkerPool(size_t participants) {
    for (size_t i = 1; i < participants; ++i)
      workers_.emplace_back([this] { loop(); });
  }

  void work() {
    for (;;) {
      const size_t index = next_.fetch_add(1, std::memory_order_relaxed);
      if (index >= tasks_) return;
      try {
        invoke_(context_, index);
      } catch (...) {
        const std::lock_guard<std::mutex> held(error_lock_);
        if (!error_) error_ = std::current_exception();
      }
    }
  }

  void loop() {
    inside_pool() = true;
    uint64_t seen = 0;  // the generation at construction, not at thread start
    for (;;) {
      // Spin briefly (requests arrive back to back within one graph
      // boundary), then sleep on the futex.
      uint64_t now = seen;
      for (int spin = 0; spin < 2000; ++spin) {
        now = generation_.load(std::memory_order_acquire);
        if (now != seen) break;
#if defined(__aarch64__)
        asm volatile("yield");
#endif
      }
      if (now == seen) {
        generation_.wait(seen, std::memory_order_acquire);
        now = generation_.load(std::memory_order_acquire);
        if (now == seen) continue;
      }
      seen = now;
      if (stop_.load(std::memory_order_relaxed)) return;
      work();
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  std::vector<std::thread> workers_;
  std::mutex submit_;
  std::mutex error_lock_;
  std::exception_ptr error_;
  void* context_ = nullptr;
  void (*invoke_)(void*, size_t) = nullptr;
  size_t tasks_ = 0;
  std::atomic<size_t> next_{0};
  std::atomic<size_t> pending_{0};
  std::atomic<uint64_t> generation_{0};
  std::atomic<bool> stop_{false};
};

// Values below which a single thread is used: waking workers costs more than
// it saves (measured; see RESULTS.json).
inline size_t parallel_threshold() {
  static const size_t value = [] {
    const char* text = std::getenv("POLIMA_PARALLEL_MIN_VALUES");
    if (text && *text) return static_cast<size_t>(std::strtoull(text, nullptr, 10));
    return size_t{131072};
  }();
  return value;
}

// Lower bound for compute-bound kernels (INT8 quantize/dequantize), which
// scale with threads at smaller sizes than the memory-bound bf16 copies.
inline size_t compute_parallel_threshold() {
  static const size_t value = [] {
    const char* text = std::getenv("POLIMA_PARALLEL_MIN_COMPUTE");
    if (text && *text) return static_cast<size_t>(std::strtoull(text, nullptr, 10));
    return size_t{32768};
  }();
  return value;
}

// Splits [0, count) into contiguous chunks whose starts are multiples of 16
// and calls body(begin, end) for each, possibly concurrently.
template <class F>
void parallel_chunks(size_t count, F&& body, size_t threshold = parallel_threshold()) {
  auto& pool = WorkerPool::instance();
  if (count < threshold || pool.size() == 1) {
    body(size_t{0}, count);
    return;
  }
  const size_t parts = pool.size();
  size_t chunk = (count + parts - 1) / parts;
  chunk = (chunk + 15) & ~size_t{15};
  const size_t tasks = (count + chunk - 1) / chunk;
  pool.run(tasks, [&](size_t task) {
    const size_t begin = task * chunk;
    body(begin, std::min(count, begin + chunk));
  });
}

// ---------------------------------------------------------------------------
// Finite checks
// ---------------------------------------------------------------------------

// Index of the first non-finite value, or `count` if all are finite.
// Same predicate as !std::isfinite: exponent bits all ones (inf or NaN).
inline size_t first_nonfinite_serial(const float* values, size_t count) {
  size_t i = 0;
#if defined(__aarch64__)
  const uint32_t* bits = reinterpret_cast<const uint32_t*>(values);
  const uint32x4_t exponent = vdupq_n_u32(0x7f800000u);
  for (; i + 16 <= count; i += 16) {
    uint32x4_t bad = vdupq_n_u32(0);
    for (size_t b = 0; b < 4; ++b) {
      const uint32x4_t x = vandq_u32(vld1q_u32(bits + i + 4 * b), exponent);
      bad = vorrq_u32(bad, vceqq_u32(x, exponent));
    }
    if (vmaxvq_u32(bad)) break;  // locate it below
  }
#endif
  for (; i < count; ++i)
    if (!std::isfinite(values[i])) return i;
  return count;
}

inline size_t first_nonfinite(const float* values, size_t count) {
  if (count < parallel_threshold()) return first_nonfinite_serial(values, count);
  std::atomic<size_t> first{count};
  parallel_chunks(count, [&](size_t begin, size_t end) {
    const size_t found = first_nonfinite_serial(values + begin, end - begin);
    if (found != end - begin) {
      size_t index = begin + found, current = first.load();
      while (index < current && !first.compare_exchange_weak(current, index)) {}
    }
  });
  return first.load();
}

// ---------------------------------------------------------------------------
// BF16
// ---------------------------------------------------------------------------

// Same integer formula as polima::to_bf16 for every input bit pattern.
inline void f32_to_bf16_serial(const float* source, uint16_t* destination, size_t count) {
  size_t i = 0;
#if defined(__aarch64__)
  const uint32_t* bits = reinterpret_cast<const uint32_t*>(source);
  const uint32x4_t bias = vdupq_n_u32(0x7fffu), one = vdupq_n_u32(1u);
  for (; i + 16 <= count; i += 16) {
    uint16x8_t out[2];
    for (size_t h = 0; h < 2; ++h) {
      uint32x4_t a = vld1q_u32(bits + i + 8 * h);
      uint32x4_t b = vld1q_u32(bits + i + 8 * h + 4);
      a = vaddq_u32(a, vaddq_u32(bias, vandq_u32(vshrq_n_u32(a, 16), one)));
      b = vaddq_u32(b, vaddq_u32(bias, vandq_u32(vshrq_n_u32(b, 16), one)));
      out[h] = vcombine_u16(vshrn_n_u32(a, 16), vshrn_n_u32(b, 16));
    }
    vst1q_u16(destination + i, out[0]);
    vst1q_u16(destination + i + 8, out[1]);
  }
#endif
  for (; i < count; ++i) destination[i] = to_bf16(source[i]);
}

inline void f32_to_bf16(const float* source, uint16_t* destination, size_t count) {
  parallel_chunks(count, [&](size_t begin, size_t end) {
    f32_to_bf16_serial(source + begin, destination + begin, end - begin);
  });
}

// ---------------------------------------------------------------------------
// uint8 camera frames
// ---------------------------------------------------------------------------

// destination[i] = table[c * 256 + source[i]], c = i % channels: a uint8 frame
// decoded through a per-channel table (see WireDecode in plan.hpp).
inline void u8_lookup(const uint8_t* source, float* destination, size_t count,
                      const float* table, size_t channels) {
  parallel_chunks(count, [&](size_t begin, size_t end) {
    size_t channel = begin % channels;
    for (size_t i = begin; i < end; ++i) {
      destination[i] = table[channel * 256 + source[i]];
      if (++channel == channels) channel = 0;
    }
  });
}

// bf16 -> float (exact widening) and, in the same pass, the index of the first
// value whose exponent is all ones -- precisely the values that decode to a
// non-finite float. Returns `count` when every value is finite.
inline size_t bf16_to_f32_checked_serial(const uint16_t* source, float* destination,
                                         size_t count) {
  size_t i = 0;
  bool bad = false;
#if defined(__aarch64__)
  uint32_t* out = reinterpret_cast<uint32_t*>(destination);
  const uint16x8_t exponent = vdupq_n_u16(0x7f80u);
  uint16x8_t flags = vdupq_n_u16(0);
  for (; i + 16 <= count; i += 16) {
    const uint16x8_t a = vld1q_u16(source + i);
    const uint16x8_t b = vld1q_u16(source + i + 8);
    flags = vorrq_u16(flags, vceqq_u16(vandq_u16(a, exponent), exponent));
    flags = vorrq_u16(flags, vceqq_u16(vandq_u16(b, exponent), exponent));
    vst1q_u32(out + i, vshll_n_u16(vget_low_u16(a), 16));
    vst1q_u32(out + i + 4, vshll_high_n_u16(a, 16));
    vst1q_u32(out + i + 8, vshll_n_u16(vget_low_u16(b), 16));
    vst1q_u32(out + i + 12, vshll_high_n_u16(b, 16));
  }
  bad = vmaxvq_u16(flags) != 0;
#endif
  for (; i < count; ++i) {
    destination[i] = from_bf16(source[i]);
    if ((source[i] & 0x7f80u) == 0x7f80u) bad = true;
  }
  if (!bad) return count;
  for (size_t j = 0; j < count; ++j)
    if ((source[j] & 0x7f80u) == 0x7f80u) return j;
  return count;
}

inline size_t bf16_to_f32_checked(const uint16_t* source, float* destination, size_t count) {
  if (count < parallel_threshold()) return bf16_to_f32_checked_serial(source, destination, count);
  std::atomic<size_t> first{count};
  parallel_chunks(count, [&](size_t begin, size_t end) {
    const size_t found = bf16_to_f32_checked_serial(source + begin, destination + begin, end - begin);
    if (found != end - begin) {
      size_t index = begin + found, current = first.load();
      while (index < current && !first.compare_exchange_weak(current, index)) {}
    }
  });
  return first.load();
}

// ---------------------------------------------------------------------------
// Copies
// ---------------------------------------------------------------------------

inline void copy_bytes(void* destination, const void* source, size_t bytes) {
  if (bytes < parallel_threshold() * 4) {
    std::memcpy(destination, source, bytes);
    return;
  }
  parallel_chunks(bytes, [&](size_t begin, size_t end) {
    std::memcpy(static_cast<char*>(destination) + begin,
                static_cast<const char*>(source) + begin, end - begin);
  });
}

}  // namespace kernels
}  // namespace polima
