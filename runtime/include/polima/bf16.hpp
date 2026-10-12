// bfloat16 conversion.
//
// The rounding is round-to-nearest-even, expressed as a magic-constant add:
// adding 0x7fff plus the low bit of the surviving mantissa rounds the discarded
// 16 bits correctly, including the halfway case.
#pragma once

#include <cstdint>
#include <cstring>

namespace polima {

inline uint16_t to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float from_bf16(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline void to_bf16_span(const float* source, uint16_t* destination, size_t count) {
  for (size_t index = 0; index < count; ++index) destination[index] = to_bf16(source[index]);
}

inline void from_bf16_span(const uint16_t* source, float* destination, size_t count) {
  for (size_t index = 0; index < count; ++index) destination[index] = from_bf16(source[index]);
}

}  // namespace polima
