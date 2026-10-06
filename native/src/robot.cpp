#include "polima/robot.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <set>
#include <csignal>
#include <stdexcept>

#include <dirent.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace polima {
namespace fs = std::filesystem;
namespace {

using json = nlohmann::json;

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lowered(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return text;
}

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.rfind(prefix, 0) == 0;
}

void supervisor_noop(int) {}

std::vector<std::string> default_camera_hints(const RobotDescription& description,
                                              const std::string& role) {
  // Bundle hints remain authoritative, but these vendor/model families make
  // an ACT bundle built for a C920 work with the board's C922 replacement.
  // This is deliberately a list of stable by-id names, never /dev/videoN.
  // The reference MyCobot rig uses the WSD camera as the fixed/front view and
  // the Generic camera as the arm-mounted/wrist view. Do not reuse the SO-101
  // role assumptions: swapped feeds are valid tensors but differ from training.
  if (description.calibration_id == "mycobot280") {
    if (role == "front") return {"202406051510"};
    if (role == "wrist") return {"20220523"};
    return {};
  }
  if (role == "overhead") return {"c920", "c922", "logitech", "046d"};
  if (role == "front") return {"generic", "usb_2.0_camera", "20220523"};
  if (role == "wrist") return {"wsd", "usb_camera", "202406051510"};
  return {};
}

std::vector<const CameraDevice*> matching_cameras(
    const std::vector<CameraDevice>& cameras, const std::vector<std::string>& hints,
    const std::vector<std::string>& taken) {
  std::vector<const CameraDevice*> matches;
  for (const auto& camera : cameras) {
    if (std::find(taken.begin(), taken.end(), camera.path) != taken.end()) continue;
    const std::string name = lowered(camera.name);
    if (std::any_of(hints.begin(), hints.end(), [&](const std::string& hint) {
          return name.find(lowered(hint)) != std::string::npos;
        }))
      matches.push_back(&camera);
  }
  return matches;
}

}  // namespace

std::vector<CameraDevice> list_cameras(const fs::path& by_id) {
  std::vector<CameraDevice> cameras;
  std::error_code code;
  if (!fs::is_directory(by_id, code)) return cameras;

  for (const auto& entry : fs::directory_iterator(by_id, code)) {
    const std::string name = entry.path().filename().string();
    // Only the capture node. Higher indices are metadata streams: they open
    // fine and deliver nothing, which is a confusing way to get a black feed.
    if (!ends_with(name, "-video-index0")) continue;
    CameraDevice camera;
    camera.name = name;
    camera.path = entry.path().string();
    const fs::path resolved = fs::read_symlink(entry.path(), code);
    if (!code) camera.node = fs::weakly_canonical(by_id / resolved, code).string();
    cameras.push_back(std::move(camera));
  }
  std::sort(cameras.begin(), cameras.end(),
            [](const CameraDevice& a, const CameraDevice& b) { return a.name < b.name; });
  return cameras;
}

std::vector<std::string> list_serial_ports(const fs::path& dev) {
  // Stable links first. ttyACM indices follow enumeration order and can swap
  // across a reboot; calibration is keyed to the physical arm, so silently
  // choosing the wrong raw index is unsafe.
  const std::vector<fs::path> stable_directories = {
      dev / "serial/by-id", dev / "serial/by-path"};
  for (const fs::path& stable : stable_directories) {
    std::vector<std::string> ports;
    std::set<std::string> devices;
    std::error_code stable_code;
    if (!fs::is_directory(stable, stable_code)) continue;
    for (const auto& entry : fs::directory_iterator(stable, stable_code)) {
      const fs::path resolved = fs::weakly_canonical(entry.path(), stable_code);
      if (stable_code) continue;
      const std::string name = resolved.filename().string();
      if (!starts_with(name, "ttyACM") && !starts_with(name, "ttyUSB")) continue;
      if (devices.insert(resolved.string()).second) ports.push_back(entry.path().string());
    }
    if (!ports.empty()) {
      std::sort(ports.begin(), ports.end());
      return ports;
    }
  }

  std::vector<std::string> ports;
  std::error_code code;
  if (!fs::is_directory(dev, code)) return ports;
  for (const auto& entry : fs::directory_iterator(dev, code)) {
    const std::string name = entry.path().filename().string();
    if (starts_with(name, "ttyACM") || starts_with(name, "ttyUSB"))
      ports.push_back(entry.path().string());
  }
  std::sort(ports.begin(), ports.end());
  return ports;
}

RobotDescription read_robot_description(const fs::path& bundle_root) {
  RobotDescription description;
  std::ifstream stream(bundle_root / "plan.json");
  if (!stream) return description;
  json plan;
  try {
    stream >> plan;
  } catch (const std::exception&) {
    return description;
  }
  if (!plan.contains("robot")) return description;

  const json& robot = plan.at("robot");
  description.present = true;
  description.fps = robot.value("fps", 30);
  description.actions_per_chunk = robot.value("actions_per_chunk", 0);
  description.max_relative_target = robot.value("max_relative_target", 12);
  description.calibration_id = robot.value("calibration_id", "");
  description.fourcc = robot.value("camera_fourcc", "MJPG");
  const json capture_fps = robot.value("camera_capture_fps", json::object());
  for (auto item = capture_fps.begin(); item != capture_fps.end(); ++item)
    description.camera_capture_fps[item.key()] = item.value().get<int>();
  // Compatibility for MyCobot bundles packed before capture settings became
  // explicit. Their plan said MJPG even though the proven dual-camera launcher
  // used YUYV with device-specific rates.
  if (description.calibration_id == "mycobot280" &&
      description.camera_capture_fps.empty()) {
    description.fourcc = "YUYV";
    description.camera_capture_fps = {{"front", 30}, {"wrist", 25}};
  }
  for (const auto& pair : robot.value("camera_roles", json::array())) {
    if (pair.is_array() && pair.size() >= 2)
      description.roles.push_back({pair[0].get<std::string>(), pair[1].get<std::string>()});
  }
  // Bind the object once. `robot.value(...)` returns a fresh temporary on every
  // call, so iterating begin() to end() across two calls compares iterators
  // into two different containers.
  const json hints = robot.value("camera_hints", json::object());
  for (auto item = hints.begin(); item != hints.end(); ++item)
    description.hints[item.key()] = item.value().get<std::string>();
  if (plan.contains("wire"))
    description.default_port = plan.at("wire").value("default_port", 0);
  return description;
}

CameraAssignment assign_cameras(const RobotDescription& description,
                                const std::vector<CameraDevice>& cameras) {
  CameraAssignment result;
  std::vector<std::string> taken;

  for (const auto& [role, label] : description.roles) {
    (void)label;
    const auto hint = description.hints.find(role);
    std::vector<std::string> hints;
    if (hint != description.hints.end() && !hint->second.empty()) hints.push_back(hint->second);
    std::vector<const CameraDevice*> matches = matching_cameras(cameras, hints, taken);
    if (matches.empty())
      matches = matching_cameras(cameras, default_camera_hints(description, role), taken);
    if (matches.empty()) {
      result.problems.push_back(role + ": no matching " + role + " camera");
      continue;
    }
    if (matches.size() > 1) {
      result.problems.push_back(role + ": " + std::to_string(matches.size()) +
                                " cameras match; pass --" + role + "-camera");
      continue;
    }
    // Deliberately never falls back on enumeration order: /dev/videoN is
    // assigned in plug order, so a reboot can swap two cameras, and a swapped
    // pair raises nothing at all -- the policy runs and the arm reaches for the
    // wrong place.
    result.assigned[role] = matches.front()->path;
    taken.push_back(matches.front()->path);
  }
  return result;
}

int run_camera_preview(const fs::path& bundle_root,
                       const CameraAssignment& cameras,
                       const std::string& fourcc,
                       int preview_port,
                       const fs::path& venv) {
  const fs::path script = bundle_root / "robot_client/preview_robot_cameras.py";
  const fs::path python = venv / "bin/python";
  if (!fs::is_regular_file(script))
    throw std::runtime_error(
        "bundle has no robot_client/preview_robot_cameras.py; redeploy it");
  if (!fs::exists(python))
    throw std::runtime_error("LeRobot environment not found at " + venv.string());
  auto primary = cameras.assigned.find("overhead");
  if (primary == cameras.assigned.end()) primary = cameras.assigned.find("front");
  const auto wrist = cameras.assigned.find("wrist");
  if (primary == cameras.assigned.end() || wrist == cameras.assigned.end())
    throw std::runtime_error("both primary and wrist cameras must be assigned");
  if (preview_port <= 0 || preview_port > 65535)
    throw std::runtime_error("preview port must be between 1 and 65535");

  // Keep the native parent (and therefore the cross-controller lock) alive
  // while a terminal or Studio signal is handled by the preview child.
  ::signal(SIGINT, supervisor_noop);
  ::signal(SIGTERM, supervisor_noop);
  ::signal(SIGUSR1, supervisor_noop);
  const pid_t pid = ::fork();
  if (pid < 0) throw std::runtime_error("fork failed while starting camera preview");
  if (pid == 0) {
    const std::string port = std::to_string(preview_port);
    const fs::path focus = bundle_root / "robot_client/camera_focus_config.json";
    ::execl(python.c_str(), python.c_str(), script.c_str(),
            "--perspective", primary->second.c_str(),
            "--wrist", wrist->second.c_str(),
            "--host", "0.0.0.0",
            "--port", port.c_str(),
            "--fourcc", fourcc.c_str(),
            "--focus-config", focus.c_str(),
            "--view-only",
            static_cast<char*>(nullptr));
    ::_exit(127);
  }

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 1;
}

int run_robot_client(const fs::path& bundle_root,
                     const RobotDescription& description,
                     const std::string& robot_port,
                     const CameraAssignment& cameras,
                     const std::string& server_address,
                     const fs::path& venv) {
  const fs::path launcher = bundle_root / "robot_client/start.sh";
  const fs::path python = venv / "bin/python";
  if (!fs::is_regular_file(launcher))
    throw std::runtime_error("bundle has no robot_client/start.sh; repack and redeploy it");
  if (!fs::exists(python))
    throw std::runtime_error("LeRobot environment not found at " + venv.string());
  if (robot_port.empty()) throw std::runtime_error("no follower-arm serial port selected");

  const auto wrist = cameras.assigned.find("wrist");
  auto primary = cameras.assigned.find("overhead");
  std::string primary_option = "--perspective-camera";
  if (primary == cameras.assigned.end()) {
    primary = cameras.assigned.find("front");
    primary_option = "--front-camera";
  }
  if (primary == cameras.assigned.end() || wrist == cameras.assigned.end())
    throw std::runtime_error("both primary and wrist cameras must be assigned");

  // Studio broadcasts SIGUSR1 to the process group for an emergency halt.
  // The native supervisor stays alive to reap the Python client; the client
  // installs its own SIGUSR1 handler and immediately disconnects the motors.
  ::signal(SIGINT, supervisor_noop);
  ::signal(SIGTERM, supervisor_noop);
  ::signal(SIGUSR1, supervisor_noop);
  const pid_t pid = ::fork();
  if (pid < 0) throw std::runtime_error("fork failed while starting robot client");
  if (pid == 0) {
    ::setenv("LEROBOT_VENV", venv.c_str(), 1);
    // Legacy launchers defaulted to hand-built trees such as
    // /media/nvme/ACT_rcwb_f_t_100000. A PoLiMa bundle is relocatable, so
    // always point both supported launchers at the bundle selected by the
    // caller. Setting both is harmless and keeps this generic native layer
    // independent of the policy name.
    ::setenv("ACT_SOM_ROOT", bundle_root.c_str(), 1);
    ::setenv("SMOLVLA_SOM_ROOT", bundle_root.c_str(), 1);
    ::setenv("POLIMA_CAMERA_FOURCC", description.fourcc.c_str(), 1);
    for (const auto& [role, rate] : description.camera_capture_fps) {
      const std::string value = std::to_string(rate);
      if (role == "front" && !std::getenv("POLIMA_FRONT_CAMERA_FPS"))
        ::setenv("POLIMA_FRONT_CAMERA_FPS", value.c_str(), 1);
      if (role == "wrist" && !std::getenv("POLIMA_WRIST_CAMERA_FPS"))
        ::setenv("POLIMA_WRIST_CAMERA_FPS", value.c_str(), 1);
    }
    const std::string fps = std::to_string(description.fps);
    const std::string actions = std::to_string(description.actions_per_chunk);
    const std::string target = std::to_string(description.max_relative_target);
    ::execl("/bin/bash", "bash", launcher.c_str(),
            "--robot-port", robot_port.c_str(),
            primary_option.c_str(), primary->second.c_str(),
            "--wrist-camera", wrist->second.c_str(),
            "--server-address", server_address.c_str(),
            "--fps", fps.c_str(),
            "--max-relative-target", target.c_str(),
            "--actions-per-chunk", actions.c_str(),
            static_cast<char*>(nullptr));
    ::_exit(127);
  }

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 1;
}

}  // namespace polima
