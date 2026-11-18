#pragma once

#include <filesystem>
#include <ostream>
#include <string>

namespace polima {

struct RuntimeCompatibility {
  bool compatible = false;
  std::string platform;
  std::string llima;
  std::string detail;
};

RuntimeCompatibility runtime_compatibility(
    const std::filesystem::path& buildinfo = "/etc/buildinfo",
    const std::filesystem::path& package_status = "/var/lib/dpkg/status");

void require_runtime_compatibility();
// Prints the build and installed identities; returns the process exit code.
int print_runtime_info(std::ostream& out);

}  // namespace polima
