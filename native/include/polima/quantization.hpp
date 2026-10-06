#pragma once

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "polima/kernels.hpp"

namespace polima {

struct TensorQuantization {
  size_t elements = 0;
  float multiplier = 1.0f;
  int zero_point = 0;

  void validate() const {
    if (!elements || !std::isfinite(multiplier) || multiplier <= 0 ||
        zero_point < -128 || zero_point > 127)
      throw std::runtime_error("invalid INT8 boundary quantization");
  }

  int8_t quantize(float value) const {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite INT8 input");
    const float scaled = value * multiplier + static_cast<float>(zero_point);
    return static_cast<int8_t>(std::nearbyint(std::clamp(scaled, -128.0f, 127.0f)));
  }

  // A producer's raw INT8 codes may feed this input directly only if the
  // normal dequantize/requantize boundary preserves every possible code.
  bool preserves_codes_from(const TensorQuantization& producer) const {
    if (elements != producer.elements) return false;
    for (int code = -128; code <= 127; ++code)
      if (static_cast<int>(quantize(producer.dequantize(static_cast<int8_t>(code)))) != code)
        return false;
    return true;
  }

  // The contract of a span of `count` values: every value must be finite
  // (otherwise "nonfinite INT8 input" is thrown and nothing is promised about
  // `output`); values [0, neon_end(count)) use the NEON arithmetic below and
  // the rest the scalar quantize(). With the default rounding mode the two
  // agree, but the split is kept exactly so that no index can change bits.
  static size_t neon_end(size_t count) {
#if defined(__aarch64__)
    if (std::fegetround() == FE_TONEAREST) return count & ~size_t{15};
#endif
    (void)count;
    return 0;
  }

  // One chunk [begin, end) of a span of `count` values whose start `input`
  // and `output` point at. `begin` must be a multiple of 16. The finite check
  // runs in the same pass as the conversion instead of a separate scalar scan.
  void quantize_chunk(const float* input, int8_t* output, size_t begin, size_t end,
                      size_t neon_limit) const {
    size_t i = begin;
#if defined(__aarch64__)
    const size_t stop = std::min(end, neon_limit);
    if (i < stop) {
      const float32x4_t zp = vdupq_n_f32(static_cast<float>(zero_point));
      const float32x4_t low = vdupq_n_f32(-128.0f);
      const float32x4_t high = vdupq_n_f32(127.0f);
      const uint32x4_t exponent = vdupq_n_u32(0x7f800000u);
      uint32x4_t bad = vdupq_n_u32(0);
      for (; i + 16 <= stop; i += 16) {
        int32x4_t lanes[4];
        for (size_t block = 0; block < 4; ++block) {
          const float32x4_t values = vld1q_f32(input + i + 4 * block);
          bad = vorrq_u32(bad, vceqq_u32(vandq_u32(vreinterpretq_u32_f32(values), exponent),
                                         exponent));
          // Keep the multiply and add separate, as in quantize(float).
          const float32x4_t scaled = vaddq_f32(vmulq_n_f32(values, multiplier), zp);
          lanes[block] = vcvtnq_s32_f32(vminq_f32(vmaxq_f32(scaled, low), high));
        }
        const int16x8_t first = vcombine_s16(vmovn_s32(lanes[0]), vmovn_s32(lanes[1]));
        const int16x8_t second = vcombine_s16(vmovn_s32(lanes[2]), vmovn_s32(lanes[3]));
        vst1q_s8(output + i, vcombine_s8(vqmovn_s16(first), vqmovn_s16(second)));
      }
      if (vmaxvq_u32(bad)) throw std::runtime_error("nonfinite INT8 input");
    }
#else
    (void)neon_limit;
#endif
    for (; i < end; ++i) output[i] = quantize(input[i]);
  }

  // Preserve the scalar contract, including rejection of nonfinite inputs.
  // On the devkit, convert sixteen values per NEON iteration. The remaining
  // values and non-default rounding modes use the original scalar operation.
  void quantize_span(const float* input, int8_t* output, size_t count) const {
    const size_t limit = neon_end(count);
    // Rounding mode is per thread: only the default mode may use workers.
    if (std::fegetround() != FE_TONEAREST) {
      quantize_chunk(input, output, 0, count, limit);
      return;
    }
    kernels::parallel_chunks(
        count, [&](size_t begin, size_t end) { quantize_chunk(input, output, begin, end, limit); },
        kernels::compute_parallel_threshold());
  }

  float dequantize(int8_t value) const {
    return (static_cast<int>(value) - zero_point) / multiplier;
  }

  // All 256 results of dequantize(), indexed by code + 128. Built with the
  // scalar function itself, so a table lookup returns identical bits.
  std::array<float, 256> dequantize_table() const {
    std::array<float, 256> table{};
    for (int code = -128; code <= 127; ++code)
      table[static_cast<size_t>(code + 128)] = dequantize(static_cast<int8_t>(code));
    return table;
  }
};

// True when every possible dequantized value (hence every output) is finite.
inline bool table_is_finite(const std::array<float, 256>& table) {
  for (float value : table)
    if (!std::isfinite(value)) return false;
  return true;
}

// TensorQuantization::dequantize over a span: the same integer subtract,
// exact int -> float conversion and IEEE division, sixteen values per NEON
// iteration (vector FDIV rounds exactly like scalar FDIV), split over the
// worker pool. Measured on the A65: a 256-entry table lookup was slower
// (3.7 ns/value) than division (2.5 ns/value) single-threaded.
inline void dequantize_span(const TensorQuantization& q, const int8_t* input, float* output,
                            size_t count) {
  kernels::parallel_chunks(
      count,
      [&](size_t begin, size_t end) {
        size_t i = begin;
#if defined(__aarch64__)
        const int32x4_t zp = vdupq_n_s32(q.zero_point);
        const float32x4_t multiplier = vdupq_n_f32(q.multiplier);
        for (; i + 16 <= end; i += 16) {
          const int8x16_t codes = vld1q_s8(input + i);
          const int16x8_t low = vmovl_s8(vget_low_s8(codes));
          const int16x8_t high = vmovl_high_s8(codes);
          const int32x4_t lanes[4] = {vmovl_s16(vget_low_s16(low)), vmovl_high_s16(low),
                                      vmovl_s16(vget_low_s16(high)), vmovl_high_s16(high)};
          for (size_t b = 0; b < 4; ++b)
            vst1q_f32(output + i + 4 * b,
                      vdivq_f32(vcvtq_f32_s32(vsubq_s32(lanes[b], zp)), multiplier));
        }
#endif
        for (; i < end; ++i) output[i] = q.dequantize(input[i]);
      },
      kernels::compute_parallel_threshold());
}

}  // namespace polima
