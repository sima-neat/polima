#include "polima/compatibility.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#ifndef POLIMA_EXPECTED_PLATFORM
#define POLIMA_EXPECTED_PLATFORM "unknown"
#endif

#ifndef POLIMA_EXPECTED_LLIMA
#define POLIMA_EXPECTED_LLIMA "unknown"
#endif

#ifndef POLIMA_SOURCE_COMMIT
#define POLIMA_SOURCE_COMMIT "unknown"
#endif

namespace polima {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n\"");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n\"");
  return value.substr(first, last - first + 1);
}

std::string build_value(const std::filesystem::path& path, const std::string& wanted) {
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    const auto separator = line.find('=');
    if (separator != std::string::npos && trim(line.substr(0, separator)) == wanted)
      return trim(line.substr(separator + 1));
  }
  return {};
}

std::string package_version(const std::filesystem::path& path, const std::string& wanted) {
  std::ifstream input(path);
  std::string line;
  std::string package;
  std::string version;
  std::string status;
  auto finish = [&]() -> std::string {
    if (package == wanted && status == "install ok installed") return version;
    return {};
  };

  while (std::getline(input, line)) {
    if (line.empty()) {
      const auto found = finish();
      if (!found.empty()) return found;
      package.clear();
      version.clear();
      status.clear();
      continue;
    }
    const auto separator = line.find(':');
    if (separator == std::string::npos) continue;
    const auto field = line.substr(0, separator);
    const auto value = trim(line.substr(separator + 1));
    if (field == "Package") package = value;
    if (field == "Version") version = value;
    if (field == "Status") status = value;
  }
  return finish();
}

}  // namespace

RuntimeCompatibility runtime_compatibility(const std::filesystem::path& buildinfo,
                                           const std::filesystem::path& package_status) {
  RuntimeCompatibility result;
  result.platform = build_value(buildinfo, "DISTRO_VERSION");
  result.llima = package_version(package_status, "sima-lmm-core");

  if (result.platform.empty()) {
    result.detail = "Modalix platform version is unavailable in " + buildinfo.string();
  } else if (result.llima.empty()) {
    result.detail = "sima-lmm-core is not installed";
  } else if (result.platform != POLIMA_EXPECTED_PLATFORM ||
             result.llima != POLIMA_EXPECTED_LLIMA) {
    result.detail = "runtime was built for platform " + std::string(POLIMA_EXPECTED_PLATFORM) +
                    " and sima-lmm-core " + POLIMA_EXPECTED_LLIMA +
                    "; installed platform " + result.platform + ", sima-lmm-core " +
                    result.llima;
  } else {
    result.compatible = true;
    result.detail = "platform and LLiMa match the native build";
  }
  return result;
}

void require_runtime_compatibility() {
  const auto result = runtime_compatibility();
  if (!result.compatible)
    throw std::runtime_error("incompatible native runtime: " + result.detail +
                             "; reinstall PoLiMa for this DevKit's SDK and LLiMa");
}

int print_runtime_info(std::ostream& out) {
  const auto result = runtime_compatibility();
  out << "source=" << POLIMA_SOURCE_COMMIT << " platform=" << POLIMA_EXPECTED_PLATFORM
      << " llima=" << POLIMA_EXPECTED_LLIMA << "\n"
      << "installed_platform=" << result.platform << " installed_llima=" << result.llima << "\n"
      << result.detail << std::endl;
  return result.compatible ? 0 : 1;
}

}  // namespace polima
