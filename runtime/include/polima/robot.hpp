// The robot side of polima-cli.
//
// The DevKit runs the policy server and drives the arm; the host only compiles
// and deploys. So the robot commands belong in the binary already on the DevKit,
// not in a second Python install shipped alongside it.
//
// What the DevKit needs to know about the robot -- camera roles, which physical
// device fills each, joint order, fps -- travels in plan.json's `robot` section,
// written from RobotSpec at pack time. That is what lets this do discovery
// without a PolicySpec, and the DevKit deliberately has no polima Python.
//
// The control loop itself is not here and cannot be: driving an SO-101 means
// lerobot, which is Python. polima-cli launches it in the DevKit's
// /media/nvme/lerobot venv, the same way it launches polima_server.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace polima {

struct CameraDevice {
  std::string name;   // usb-046d_HD_Pro_Webcam_C920-video-index0
  std::string path;   // /dev/v4l/by-id/<name>
  std::string node;   // the /dev/videoN it resolves to
};

// The robot half of plan.json, written from RobotSpec at pack time.
struct RobotDescription {
  bool present = false;
  std::vector<std::pair<std::string, std::string>> roles;   // (role, label)
  std::map<std::string, std::string> hints;                 // role -> by-id substring
  std::map<std::string, int> camera_capture_fps;             // role -> hardware FPS
  std::string fourcc = "MJPG";
  std::string calibration_id;
  int fps = 30;
  int default_port = 0;
  int actions_per_chunk = 0;
  int max_relative_target = 12;
};

struct CameraAssignment {
  std::map<std::string, std::string> assigned;   // role -> device path
  std::vector<std::string> problems;
};

std::vector<CameraDevice> list_cameras(
    const std::filesystem::path& by_id = "/dev/v4l/by-id");
std::vector<std::string> list_serial_ports(const std::filesystem::path& dev = "/dev");

RobotDescription read_robot_description(const std::filesystem::path& bundle_root);
CameraAssignment assign_cameras(const RobotDescription& description,
                                const std::vector<CameraDevice>& cameras);

// Serve both camera feeds in a browser without opening the follower arm or
// starting the policy server. Runs in the foreground until Ctrl-C.
int run_camera_preview(const std::filesystem::path& bundle_root,
                       const CameraAssignment& cameras,
                       const std::string& fourcc = "MJPG",
                       int preview_port = 5001,
                       const std::filesystem::path& venv = "/media/nvme/lerobot");

// Run the bundle's proven LeRobot control client in the foreground. The
// policy server remains a separate background process; Ctrl-C therefore stops
// robot motion without tearing down the loaded model.
int run_robot_client(const std::filesystem::path& bundle_root,
                     const RobotDescription& description,
                     const std::string& robot_port,
                     const CameraAssignment& cameras,
                     const std::string& server_address,
                     const std::filesystem::path& venv = "/media/nvme/lerobot");

}  // namespace polima
