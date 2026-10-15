// Bit-exactness tests for polima/kernels.hpp and polima/quantization.hpp.
//
// Every optimised kernel is compared against a straightforward reference
// implementation below (the "reference_" functions) on random data, edge values,
// injected non-finite values and, with --exhaustive, every float bit pattern.
//
// Needs no MLA runtime: builds on the DevKit (NEON paths) and on any host
// (scalar fallbacks). Run it under several POLIMA_CPU_THREADS /
// POLIMA_PARALLEL_MIN_VALUES settings to cover the threaded chunking;
// tests/unit/test_native_kernels.py does that on the host.
#include "polima/bf16.hpp"
#include "polima/kernels.hpp"
#include "polima/quantization.hpp"

#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace polima;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

// ---- reference implementations ------------------------------------------

size_t reference_first_nonfinite(const float* values, size_t count) {
  for (size_t index = 0; index < count; ++index)
    if (!std::isfinite(values[index])) return index;
  return count;
}

// Reference TensorQuantization::quantize_span.
bool reference_quantize_span(const TensorQuantization& q, const float* input, int8_t* output,
                          size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(input[i])) return false;  // the runtime throws "nonfinite INT8 input"
  size_t i = 0;
#if defined(__aarch64__)
  if (std::fegetround() == FE_TONEAREST) {
    const float32x4_t zp = vdupq_n_f32(static_cast<float>(q.zero_point));
    const float32x4_t low = vdupq_n_f32(-128.0f);
    const float32x4_t high = vdupq_n_f32(127.0f);
    for (; i + 16 <= count; i += 16) {
      int32x4_t lanes[4];
      for (size_t block = 0; block < 4; ++block) {
        const float32x4_t values = vld1q_f32(input + i + 4 * block);
        const float32x4_t scaled = vaddq_f32(vmulq_n_f32(values, q.multiplier), zp);
        lanes[block] = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(scaled, low), high));
      }
      const int16x8_t first = vcombine_s16(vmovn_s32(lanes[0]), vmovn_s32(lanes[1]));
      const int16x8_t second = vcombine_s16(vmovn_s32(lanes[2]), vmovn_s32(lanes[3]));
      vst1q_s8(output + i, vcombine_s8(vqmovn_s16(first), vqmovn_s16(second)));
    }
  }
#endif
  for (; i < count; ++i) output[i] = q.quantize(input[i]);
  return true;
}

// ---- data -----------------------------------------------------------------

std::vector<float> random_values(size_t count, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, scale);
  std::vector<float> values(count);
  for (auto& v : values) v = normal(rng);
  // Sprinkle edge values the conversions treat specially.
  const float edges[] = {0.0f, -0.0f, 1e-40f, -1e-40f, std::numeric_limits<float>::max(),
                         -std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
                         0.5f, -0.5f, 1.5f, 2.5f, -2.5f, 127.5f, -128.5f, 1e9f, -1e9f};
  for (size_t i = 0; i < count; i += 97) values[i] = edges[(i / 97) % 16];
  return values;
}

const size_t kSizes[] = {0, 1, 7, 15, 16, 17, 31, 33, 255, 1000, 4097, 78336, 131072 + 37,
                         262144 + 5, 522240, 602367};

// ---- tests ----------------------------------------------------------------

void test_finite() {
  for (size_t n : kSizes) {
    auto v = random_values(n, 3.0f, 1u + static_cast<uint32_t>(n));
    expect(kernels::first_nonfinite(v.data(), n) == n, "finite: all finite n=" + std::to_string(n));
    if (!n) continue;
    const float bad[] = {std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -std::numeric_limits<float>::quiet_NaN()};
    const size_t positions[] = {0, n / 3, n / 2 + 1, n - 1};
    for (size_t p : positions)
      for (float b : bad) {
        auto w = v;
        w[p] = b;
        if (p + 5 < n) w[p + 5] = b;  // a later one must not win
        expect(kernels::first_nonfinite(w.data(), n) == reference_first_nonfinite(w.data(), n),
               "finite: index n=" + std::to_string(n) + " p=" + std::to_string(p));
      }
  }
}

void test_bf16_random() {
  for (size_t n : kSizes) {
    auto v = random_values(n, 5.0f, 11u + static_cast<uint32_t>(n));
    std::vector<uint16_t> fast(n), ref(n);
    kernels::f32_to_bf16(v.data(), fast.data(), n);
    to_bf16_span(v.data(), ref.data(), n);
    expect(fast == ref, "f32_to_bf16 n=" + std::to_string(n));
    std::vector<float> back_fast(n), back_ref(n);
    const size_t bad = kernels::bf16_to_f32_checked(ref.data(), back_fast.data(), n);
    from_bf16_span(ref.data(), back_ref.data(), n);
    expect(std::memcmp(back_fast.data(), back_ref.data(), n * sizeof(float)) == 0,
           "bf16_to_f32 n=" + std::to_string(n));
    expect(bad == reference_first_nonfinite(back_ref.data(), n), "bf16 check clean n=" + std::to_string(n));
    if (n) {
      for (size_t p : {size_t{0}, n / 2, n - 1}) {
        auto w = ref;
        w[p] = 0x7fc0;  // NaN
        if (p + 3 < n) w[p + 3] = 0xff80;  // -inf
        std::vector<float> wf(n), wr(n);
        const size_t found = kernels::bf16_to_f32_checked(w.data(), wf.data(), n);
        from_bf16_span(w.data(), wr.data(), n);
        expect(std::memcmp(wf.data(), wr.data(), n * sizeof(float)) == 0, "bf16 decode with NaN");
        expect(found == reference_first_nonfinite(wr.data(), n), "bf16 check index n=" + std::to_string(n));
      }
    }
  }
}

void test_bf16_exhaustive() {
  // Every float bit pattern through the encoder, 2^20 at a time.
  const size_t block = size_t{1} << 20;
  std::vector<float> values(block);
  std::vector<uint16_t> fast(block), ref(block);
  for (uint64_t base = 0; base < (uint64_t{1} << 32); base += block) {
    for (size_t i = 0; i < block; ++i) {
      const uint32_t bits = static_cast<uint32_t>(base + i);
      std::memcpy(&values[i], &bits, 4);
    }
    kernels::f32_to_bf16(values.data(), fast.data(), block);
    to_bf16_span(values.data(), ref.data(), block);
    if (fast != ref) { expect(false, "f32_to_bf16 exhaustive block " + std::to_string(base)); return; }
  }
  // Every bf16 pattern through the checked decoder.
  std::vector<uint16_t> all(65536);
  for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<uint16_t>(i);
  std::vector<float> f(65536), r(65536);
  kernels::bf16_to_f32_checked(all.data(), f.data(), all.size());
  from_bf16_span(all.data(), r.data(), all.size());
  expect(std::memcmp(f.data(), r.data(), f.size() * 4) == 0, "bf16 decode exhaustive");
  for (size_t i = 0; i < all.size(); ++i) {
    const size_t found = kernels::bf16_to_f32_checked(&all[i], &f[i], 1);
    expect((found == 0) == !std::isfinite(r[i]), "bf16 single check " + std::to_string(i));
  }
}

void test_quantize(bool exhaustive) {
  const TensorQuantization contracts[] = {
      {0, 7.332070402615161f, -3}, {0, 0.025499999659538274f, 127}, {0, 1.0f, 0},
      {0, 108.6023885323905f, 33}, {0, 0.10595864810000155f, -12}, {0, 128.0f, 0}};
  for (auto contract : contracts) {
    for (size_t n : kSizes) {
      contract.elements = n ? n : 1;
      for (float scale : {0.1f, 3.0f, 100.0f}) {
        auto v = random_values(n, scale, 21u + static_cast<uint32_t>(n));
        std::vector<int8_t> fast(n, 0x55), ref(n, 0x33);
        bool fast_ok = true;
        try { contract.quantize_span(v.data(), fast.data(), n); } catch (const std::runtime_error& e) {
          fast_ok = false;
          expect(std::string(e.what()) == "nonfinite INT8 input", "quantize message");
        }
        const bool ref_ok = reference_quantize_span(contract, v.data(), ref.data(), n);
        expect(fast_ok == ref_ok && (!ref_ok || fast == ref),
               "quantize n=" + std::to_string(n) + " m=" + std::to_string(contract.multiplier));
        if (n) {
          for (size_t p : {size_t{0}, n / 2, n - 1}) {
            auto w = v;
            w[p] = std::numeric_limits<float>::quiet_NaN();
            bool threw = false;
            try { contract.quantize_span(w.data(), fast.data(), n); } catch (const std::runtime_error&) {
              threw = true;
            }
            expect(threw, "quantize rejects NaN n=" + std::to_string(n) + " p=" + std::to_string(p));
          }
        }
      }
      // Dequantize: table vs scalar for all 256 codes, and a span.
      const auto table = contract.dequantize_table();
      for (int code = -128; code <= 127; ++code) {
        const float a = table[static_cast<size_t>(code + 128)];
        const float b = contract.dequantize(static_cast<int8_t>(code));
        expect(std::memcmp(&a, &b, 4) == 0, "dequantize table code " + std::to_string(code));
      }
      std::vector<int8_t> codes(n);
      std::mt19937 rng(5);
      for (auto& c : codes) c = static_cast<int8_t>(rng());
      std::vector<float> fast(n), ref(n);
      dequantize_span(contract, codes.data(), fast.data(), n);
      for (size_t i = 0; i < n; ++i) ref[i] = contract.dequantize(codes[i]);
      expect(std::memcmp(fast.data(), ref.data(), n * 4) == 0, "dequantize span n=" + std::to_string(n));
    }
  }
  if (!exhaustive) return;
  // Every finite float through both quantize paths for one GR00T contract.
  TensorQuantization q{size_t{1} << 20, 7.332070402615161f, -3};
  const size_t block = q.elements;
  std::vector<float> values(block);
  std::vector<int8_t> fast(block), ref(block);
  for (uint64_t base = 0; base < (uint64_t{1} << 32); base += block) {
    for (size_t i = 0; i < block; ++i) {
      uint32_t bits = static_cast<uint32_t>(base + i);
      if ((bits & 0x7f800000u) == 0x7f800000u) bits &= ~0x00800000u;  // keep finite
      std::memcpy(&values[i], &bits, 4);
    }
    q.quantize_span(values.data(), fast.data(), block);
    reference_quantize_span(q, values.data(), ref.data(), block);
    if (fast != ref) { expect(false, "quantize exhaustive block " + std::to_string(base)); return; }
  }
}

void test_pool_exceptions() {
  // An exception on a worker reaches the caller and the pool stays usable.
  for (int round = 0; round < 3; ++round) {
    bool caught = false;
    try {
      kernels::WorkerPool::instance().run(16, [](size_t i) {
        if (i == 11) throw std::runtime_error("boom");
      });
    } catch (const std::runtime_error&) { caught = true; }
    expect(caught, "pool rethrows");
  }
  std::vector<int> hits(1000, 0);
  kernels::WorkerPool::instance().run(hits.size(), [&](size_t i) { hits[i] += 1; });
  bool once = true;
  for (int h : hits) once = once && h == 1;
  expect(once, "pool runs every task exactly once");
}

}  // namespace

int main(int argc, char** argv) {
  const bool exhaustive = argc > 1 && std::string(argv[1]) == "--exhaustive";
  std::printf("kernel tests: threads=%zu threshold=%zu neon=%s exhaustive=%s\n",
              kernels::WorkerPool::instance().size(), kernels::parallel_threshold(),
#if defined(__aarch64__)
              "yes",
#else
              "no",
#endif
              exhaustive ? "yes" : "no");
  test_finite();
  test_bf16_random();
  test_quantize(exhaustive);
  test_pool_exceptions();
  if (exhaustive) test_bf16_exhaustive();
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all kernel tests passed\n");
  return 0;
}
