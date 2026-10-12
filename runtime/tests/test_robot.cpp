// Camera roles for SO-101 bundles: each physical camera lands in its own role.
// Built and run on the host by tests/unit/test_native_robot.py.
#include "polima/robot.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static int failures = 0;

static void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

static fs::path bundle(const std::string& name, const std::string& robot) {
  const fs::path root = fs::temp_directory_path() / ("polima-test-robot-" + name);
  fs::create_directories(root);
  std::ofstream(root / "plan.json") << R"({"robot": )" << robot << "}";
  return root;
}

int main() {
  const std::vector<polima::CameraDevice> cameras = {
      {"usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0", "/dev/wrist", ""},
      {"usb-046d_HD_Pro_Webcam_C920_A1B2C3-video-index0", "/dev/overhead", ""},
  };
  const std::string hints = R"("camera_hints": {"overhead": "C920", "wrist": "Sonix"}, "calibration_id": "so-arm101")";
  // N1.6 and ACT name the roles overhead/wrist; N1.7 names the overhead camera
  // after its video key, "front". Both must give the same assignment.
  for (const auto& [name, roles] : {
           std::pair{"overhead", R"([["overhead", "Overhead"], ["wrist", "Wrist"]])"},
           std::pair{"front", R"([["front", "front"], ["wrist", "wrist"]])"}}) {
    const auto description = polima::read_robot_description(
        bundle(name, std::string(R"({"camera_roles": )") + roles + ", " + hints + "}"));
    const auto assignment = polima::assign_cameras(description, cameras);
    expect(description.roles.front().first == "overhead", std::string(name) + ": first role is overhead");
    expect(assignment.problems.empty(), std::string(name) + ": no assignment problems");
    expect(assignment.assigned.count("overhead") && assignment.assigned.at("overhead") == "/dev/overhead",
           std::string(name) + ": the C920 is the overhead camera");
    expect(assignment.assigned.count("wrist") && assignment.assigned.at("wrist") == "/dev/wrist",
           std::string(name) + ": the Sonix camera is the wrist camera");
  }
  if (!failures) std::cout << "robot tests passed\n";
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
