// Runtime constants loaded from the bundle.
//
// Checkpoint-specific constants (normalization statistics, fixed embeddings)
// live in bundles/<id>/constants/ rather than in the binary, so the same binary
// serves any number of bundles. ACT needs none of this (it normalizes on the
// client from normalization_stats.npz), which is why its sidecar list is empty.
#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace polima {

inline std::vector<float> read_f32(const std::filesystem::path& path, size_t count = 0) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  stream.seekg(0, std::ios::end);
  const auto bytes = static_cast<size_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);

  if (count == 0) count = bytes / sizeof(float);
  if (bytes < count * sizeof(float))
    throw std::runtime_error(path.string() + ": expected " + std::to_string(count) +
                             " floats, file holds " + std::to_string(bytes / sizeof(float)));

  std::vector<float> values(count);
  stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
  if (static_cast<size_t>(stream.gcount()) != count * sizeof(float))
    throw std::runtime_error("short read from " + path.string());
  return values;
}

inline void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot write " + path.string());
  stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
}

// Lazily-loaded constants/*.f32, keyed by the name used in plan.json.
class Sidecars {
 public:
  Sidecars() = default;
  explicit Sidecars(std::filesystem::path directory) : directory_(std::move(directory)) {}

  const std::vector<float>& get(const std::string& name) {
    auto found = cache_.find(name);
    if (found != cache_.end()) return found->second;
    const auto path = directory_ / name;
    if (!std::filesystem::exists(path))
      throw std::runtime_error("plan requires sidecar '" + name + "' but " + path.string() +
                               " does not exist");
    return cache_.emplace(name, read_f32(path)).first->second;
  }

  bool has(const std::string& name) const {
    return std::filesystem::exists(directory_ / name);
  }

 private:
  std::filesystem::path directory_;
  std::map<std::string, std::vector<float>> cache_;
};

}  // namespace polima
