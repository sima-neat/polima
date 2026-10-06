// CPU test double of the sima-lmm 0.4.0 MLABuffer API used by PoLiMa.
// It checks data flow and binding arithmetic only; it says nothing about the
// accelerator. Signatures follow /usr/include/sima_lmm/mla_buffer.hpp.
#pragma once
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::llima {

inline std::map<std::string, size_t>& mock_counters() {
  static std::map<std::string, size_t> counters;
  return counters;
}

class MLABuffer {
 public:
  MLABuffer(std::string name, std::vector<size_t> shape, std::string dtype, bool)
      : name_(std::move(name)), shape_(std::move(shape)) {
    elem_ = dtype == "int8" ? 1 : 2;
    size_t count = 1;
    for (auto d : shape_) count *= d;
    size_ = count * elem_;
    padded_ = (size_ + 15) / 16 * 16;
  }
  void allocate() {
    if (!bytes_.empty()) throw std::runtime_error("double allocate " + name_);
    bytes_.assign(padded_, 0xA5);  // garbage until cleared, like real DMS memory
  }
  void free() { bytes_.clear(); }
  void clear(bool = true) { std::fill(bytes_.begin(), bytes_.end(), 0); }
  void upload(const void* data, size_t = 0, size_t = 0, bool = true) {
    std::memcpy(bytes_.data(), data, size_);
    ++mock_counters()["uploads_whole"];
  }
  void download(void* data) const {
    std::memcpy(data, bytes_.data(), size_);
    ++mock_counters()["downloads_whole"];
  }
  void flush_cache() const {}
  void flush_cache(size_t offset, size_t size) const { check(offset, size); }
  void invalidate_cache() const {}
  void invalidate_cache(size_t offset, size_t size) const { check(offset, size); }
  void* get_virtual_addr() const { return const_cast<uint8_t*>(bytes_.data()); }
  uint64_t get_buf_len(const std::optional<std::vector<uint32_t>>& shape = std::nullopt) const {
    if (!shape) return padded_;
    uint64_t n = shape->back() * elem_;
    for (size_t i = 0; i + 1 < shape->size(); ++i) n *= (*shape)[i];
    return n;
  }
  uint32_t offset_bytes(const std::optional<std::vector<uint32_t>>& begins) const {
    if (!begins) return 0;
    // shape {1, N}: strides {N, 1}
    return static_cast<uint32_t>(((*begins)[0] * shape_.back() + (*begins)[1]) * elem_);
  }
  const std::string& get_name() const { return name_; }
  size_t get_num_elems() const { return size_ / elem_; }
  size_t padded() const { return padded_; }
  uint8_t* raw() { return bytes_.data(); }

 private:
  void check(size_t offset, size_t size) const {
    if (offset > padded_ || size > padded_ - offset)
      throw std::out_of_range("cache op outside " + name_);
  }
  std::string name_;
  std::vector<size_t> shape_;
  size_t elem_ = 2, size_ = 0, padded_ = 0;
  std::vector<uint8_t> bytes_;
};

class MLABufferSlice {
 public:
  MLABufferSlice(MLABuffer* buf = nullptr) : buf_(buf) {}
  MLABufferSlice(MLABuffer* buf, std::vector<uint32_t> begins) : buf_(buf), begins_(begins) {}
  MLABufferSlice(MLABuffer* buf, std::vector<uint32_t> begins, std::vector<uint32_t> shapes)
      : buf_(buf), begins_(begins), shapes_(shapes) {}
  MLABuffer* get_buf_ptr() const { return buf_; }
  uint8_t* address() const { return buf_->raw() + buf_->offset_bytes(begins_); }
  uint64_t length() const { return buf_->get_buf_len(shapes_); }

 private:
  MLABuffer* buf_;
  std::optional<std::vector<uint32_t>> begins_, shapes_;
};

}  // namespace simaai::llima
