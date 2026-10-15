#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "polima/sidecar.hpp"

namespace polima {
// Cosine and mean absolute error, the same pair the host-side smoke test uses.
struct Agreement {
  double cosine = 0.0;
  double mean_abs = 0.0;
  double max_abs = 0.0;
};

inline Agreement compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  if (actual.empty() || actual.size() != expected.size())
    throw std::runtime_error("fixture comparison: action sizes do not match");
  Agreement result;
  double dot = 0.0, left = 0.0, right = 0.0, total = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
      throw std::runtime_error("fixture comparison: non-finite action");
    dot += static_cast<double>(actual[index]) * expected[index];
    left += static_cast<double>(actual[index]) * actual[index];
    right += static_cast<double>(expected[index]) * expected[index];
    const double gap = std::fabs(actual[index] - expected[index]);
    total += gap;
    result.max_abs = std::max(result.max_abs, gap);
  }
  result.cosine = (left > 0 && right > 0) ? dot / (std::sqrt(left) * std::sqrt(right)) : 0.0;
  result.mean_abs = total / actual.size();
  return result;
}

inline Agreement compare_fixture(const std::filesystem::path& root,
                                 const std::vector<float>& actual,
                                 const std::vector<float>& expected,
                                 const std::vector<float>& state) {
  const auto manifest = nlohmann::json::parse(std::ifstream(root / "bundle.json"));
  if (manifest.value("policy", "") != "groot") return compare(actual, expected);
  const auto plan = nlohmann::json::parse(std::ifstream(root / "plan.json"));
  if (plan.at("result") != "action_chunk") return compare(actual, expected);

  const auto mean = read_f32(root / "constants/action_mean");
  const auto scale = read_f32(root / "constants/action_std");
  const auto dimensions = plan.at("profile").at("action_dim").get<size_t>();
  const auto relative = plan.value("runtime_context", nlohmann::json::object())
                            .value("relative_action_dim", size_t{0});
  if (!dimensions || actual.empty() || actual.size() % dimensions ||
      actual.size() != expected.size() || mean.size() != actual.size() ||
      scale.size() != actual.size() || relative > dimensions ||
      (relative && state.size() < dimensions))
    throw std::runtime_error("fixture comparison: invalid action normalization geometry");

  auto normalized_actual = actual;
  auto normalized_expected = expected;
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto joint = i % dimensions;
    const float offset = joint < relative ? state[joint] : 0.0f;
    if (!std::isfinite(scale[i]) || scale[i] <= 0 ||
        !std::isfinite(mean[i]) || !std::isfinite(offset))
      throw std::runtime_error("fixture comparison: invalid action normalization constants");
    // Undo the plan's relative-action decoding before comparing model coordinates.
    normalized_actual[i] = (actual[i] - offset - mean[i]) / scale[i];
    normalized_expected[i] = (expected[i] - offset - mean[i]) / scale[i];
  }
  return compare(normalized_actual, normalized_expected);
}
}  // namespace polima
