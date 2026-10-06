// MLA DRAM tessellation.
//
// When a graph is compiled with TensorDRAMLayout.HWC16, the MLA writes its
// output in a 16-lane byte-planed form that must be reassembled on the host.
// The algorithm is transcribed from SmolVLA's smolvla_som_server.cpp, which is
// the only place in the legacy tree that implements it.
//
// Two things happen at once:
//
//   1. Byte planes. Each bfloat16 value is split across two byte planes 16
//      lanes apart, so a word is recovered as
//          low  = bytes[block * 32 + lane]
//          high = bytes[block * 32 + 16 + lane]
//          word = low | (high << 8)
//
//   2. Channel-block reorder. Channels are grouped in blocks of 16 and strided
//      by width, so the logical index is
//          logical[x * C + cb * 16 + lane] = physical[(cb * W + x) * 16 + lane]
//
// Keep this implementation for bundles whose measured bare-ELF contract is
// genuinely HWC16. ACT, GR00T, and the current SmolVLA graphs have all been
// measured as plain at their host-visible output boundaries.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "polima/bf16.hpp"

namespace polima {

enum class DramLayout { Plain, Hwc16 };

inline DramLayout parse_dram_layout(const std::string& name) {
  if (name == "plain" || name.empty()) return DramLayout::Plain;
  if (name == "hwc16") return DramLayout::Hwc16;
  throw std::runtime_error("unknown dram_layout: " + name);
}

// Straight bf16 -> float, for the plain layout.
inline void detessellate_plain(const uint16_t* raw, size_t elements, float* out) {
  from_bf16_span(raw, out, elements);
}

// Reassemble the byte-planed, channel-blocked HWC16 form.
//
// `elements` counts logical values. `width` and `channels` describe the logical
// tensor; channels must be a multiple of 16, which is what the layout means.
inline void detessellate_hwc16(const uint16_t* raw, size_t elements, size_t width,
                               size_t channels, float* out) {
  if (channels == 0 || channels % 16 != 0)
    throw std::runtime_error("hwc16 requires channels to be a non-zero multiple of 16");
  if (width == 0) throw std::runtime_error("hwc16 requires a non-zero width");

  const auto* bytes = reinterpret_cast<const uint8_t*>(raw);
  const size_t channel_blocks = channels / 16;
  const size_t rows = elements / (width * channels);

  for (size_t row = 0; row < rows; ++row) {
    const size_t row_base = row * width * channels;
    for (size_t block = 0; block < channel_blocks; ++block) {
      for (size_t x = 0; x < width; ++x) {
        const size_t physical_block = (block * width + x);
        for (size_t lane = 0; lane < 16; ++lane) {
          const size_t byte_base = (row_base + physical_block * 16) * 2;
          const uint16_t low = bytes[byte_base + lane];
          const uint16_t high = bytes[byte_base + 16 + lane];
          const uint16_t word = static_cast<uint16_t>(low | (high << 8));
          out[row_base + x * channels + block * 16 + lane] = from_bf16(word);
        }
      }
    }
  }
}

inline void detessellate(DramLayout layout, const uint16_t* raw, size_t elements, size_t width,
                         size_t channels, float* out) {
  if (layout == DramLayout::Hwc16)
    detessellate_hwc16(raw, elements, width, channels, out);
  else
    detessellate_plain(raw, elements, out);
}

}  // namespace polima
