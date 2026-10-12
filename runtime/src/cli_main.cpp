// polima_cli -- run a bundle once, files in, files out.
//
// The same plan interpreter as polima_server, without the socket. The deploy
// smoke test and `polima run --local` use it, and it is the fastest way to
// bisect a bad bundle because it can dump every intermediate buffer.
//
//   polima_cli --bundle <bundle>
//              --input-dir <bundle>/fixtures/inputs
//              --output /tmp/actions.f32 [--dump-stages /tmp/stages]

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "polima/plan.hpp"
#include "polima/repl.hpp"
#include "polima/robot.hpp"
#include "polima/service.hpp"
#include "polima/sidecar.hpp"
#include "polima/socket.hpp"

namespace fs = std::filesystem;

namespace {

bool confirm(const std::string& question) {
  std::cout << question << " [y/N] " << std::flush;
  std::string answer;
  if (!std::getline(std::cin, answer)) return false;
  return !answer.empty() && (answer[0] == 'y' || answer[0] == 'Y');
}

fs::path root_path() {
  const char* value = std::getenv("POLIMA_ROOT");
  return value ? fs::path(value) : fs::path("/media/nvme/polima");
}

int run_python_command(const std::string& command, int argc, char** argv) {
  const char* configured = std::getenv("LEROBOT_VENV");
  const fs::path python =
      (configured ? fs::path(configured) : fs::path("/media/nvme/lerobot")) / "bin/python";
  if (!fs::exists(python))
    throw std::runtime_error("PoLiMa's Python commands need the LeRobot environment at " +
                             python.string());

  // Deploy synchronizes the current PoLiMa package here. Put its parent at
  // the front of sys.path so robot/studio commands cannot silently import a
  // stale copy installed in the LeRobot virtual environment.
  const char* root = std::getenv("POLIMA_ROOT");
  const fs::path python_root =
      (root ? fs::path(root) : fs::path("/media/nvme/polima")) / "python";
  const char* inherited = std::getenv("PYTHONPATH");
  std::string python_path = python_root.string();
  if (inherited && *inherited) python_path += ":" + std::string(inherited);
  ::setenv("PYTHONPATH", python_path.c_str(), 1);

  std::vector<std::string> values = {
      python.string(), "-m", "polima.cli.main", command};
  for (int index = 2; index < argc; ++index) values.emplace_back(argv[index]);
  std::vector<char*> child_argv;
  child_argv.reserve(values.size() + 1);
  for (auto& value : values) child_argv.push_back(value.data());
  child_argv.push_back(nullptr);
  ::execv(python.c_str(), child_argv.data());
  throw std::runtime_error("could not launch PoLiMa Python CLI: " +
                           std::string(std::strerror(errno)));
}

int acquire_control_lock(const fs::path& root) {
  fs::create_directories(root / "var/run");
  const fs::path path = root / "var/run/control.lock";
  const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0660);
  if (fd < 0 || ::flock(fd, LOCK_EX | LOCK_NB) < 0) {
    if (fd >= 0) ::close(fd);
    throw std::runtime_error(
        "another PoLiMa controller is active; stop it before changing device state");
  }
  return fd;
}

polima::CameraAssignment camera_assignment(const polima::RobotDescription& description,
                                           const polima::Args& args) {
  auto result = polima::assign_cameras(description, polima::list_cameras());
  for (const auto& [role, label] : description.roles) {
    (void)label;
    for (const std::string& name : {role, std::string(role == "overhead" ? "front" : "")}) {
      const std::string option = "--" + name + "-camera";  // --front-camera: the overhead one
      if (!name.empty() && args.has(option)) result.assigned[role] = args.get(option);
    }
  }
  return result;
}

void print_robot(const polima::RobotDescription& description,
                 const polima::CameraAssignment& assignment,
                 const std::vector<std::string>& ports) {
  std::cout << "arm\n";
  if (ports.empty()) std::cout << "  none (expected a serial/by-id or ttyACM device)\n";
  for (const auto& port : ports) std::cout << "  " << port << "\n";
  std::cout << "cameras\n";
  for (const auto& [role, label] : description.roles) {
    const auto found = assignment.assigned.find(role);
    std::cout << "  " << role << " (" << label << ")  "
              << (found == assignment.assigned.end() ? "unassigned" : found->second) << "\n";
  }
  for (const auto& problem : assignment.problems) std::cout << "  ! " << problem << "\n";
}

std::string prompt_device(const std::string& label,
                          const std::vector<std::string>& candidates,
                          const std::string& detected) {
  if (candidates.empty())
    throw std::runtime_error("no " + label + " devices detected");

  // A detected camera may have been removed after selection for another role.
  // Enter must not silently return that already-taken device.
  const bool detected_available =
      std::find(candidates.begin(), candidates.end(), detected) != candidates.end();
  const std::string usable_default = detected_available ? detected : "";

  std::cout << "\n" << label << ":\n";
  for (size_t index = 0; index < candidates.size(); ++index) {
    std::cout << "  " << index + 1 << ") " << candidates[index];
    if (candidates[index] == usable_default) std::cout << " <- detected default";
    std::cout << "\n";
  }
  std::cout << "Select " << label;
  if (!usable_default.empty()) std::cout << " (Enter uses detected default)";
  std::cout << ": " << std::flush;

  std::string selection;
  if (!std::getline(std::cin, selection)) return "";
  if (selection.empty()) {
    if (!usable_default.empty()) return usable_default;
    throw std::runtime_error("select " + label + " by number; no detected default is available");
  }
  char* end = nullptr;
  const long number = std::strtol(selection.c_str(), &end, 10);
  if (end != selection.c_str() && *end == '\0' && number >= 1 &&
      static_cast<size_t>(number) <= candidates.size())
    return candidates[static_cast<size_t>(number - 1)];
  throw std::runtime_error("select " + label + " by number, or press Enter for the default");
}

std::vector<std::string> camera_paths(const std::vector<polima::CameraDevice>& cameras,
                                      const std::vector<std::string>& excluded) {
  std::vector<std::string> paths;
  for (const auto& camera : cameras) {
    if (std::find(excluded.begin(), excluded.end(), camera.path) == excluded.end())
      paths.push_back(camera.path);
  }
  return paths;
}

std::string prompt_bundle(const fs::path& store) {
  const auto entries = polima::scan_store(store);
  if (entries.empty()) {
    std::cout << "no bundles in " << store << "\n";
    return "";
  }
  std::cout << "Installed bundles:\n";
  for (size_t index = 0; index < entries.size(); ++index) {
    const auto& entry = entries[index];
    std::cout << "  " << index + 1 << ") " << entry.name;
    if (!entry.policy.empty()) std::cout << "  policy=" << entry.policy;
    if (!entry.robot.empty()) std::cout << "  robot=" << entry.robot;
    if (!entry.action_shape.empty())
      std::cout << "  actions=" << entry.action_shape;
    if (!entry.managed) std::cout << " [not a PoLiMa bundle]";
    if (entry.current) std::cout << " <- current";
    std::cout << "\n";
  }
  std::cout << "Select a bundle by number or name (blank cancels): " << std::flush;
  std::string selection;
  if (!std::getline(std::cin, selection)) return "";
  return selection;
}

int device_command(int argc, char** argv, const polima::Args& args) {
  if (argc < 2) return -1;
  const std::string command = argv[1];
  if (command != "server" && command != "robot" && command != "activate") return -1;

  std::string action = argc >= 3 && argv[2][0] != '-' ? argv[2] : "";
  const bool interactive = ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO);
  const fs::path root = root_path();
  fs::path bundle = args.get("--bundle", (root / "current").string());

  if (command == "activate") {
    [[maybe_unused]] const int control_lock = acquire_control_lock(root);
    const fs::path store = args.get("--models-dir", (root / "models").string());
    std::string selection = action;
    const bool guided = selection.empty();
    if (selection.empty()) {
      if (!interactive)
        throw std::runtime_error("usage: polima activate <n|bundle-id>");
      selection = prompt_bundle(store);
      if (selection.empty()) {
        std::cout << "nothing changed\n";
        return 0;
      }
    }

    const std::string selected = polima::resolve_bundle(store, selection);
    const auto state = polima::server_state(root);
    if (state.running) {
      if (interactive && !args.has("--yes") &&
          !confirm("Stop the running policy server and activate the new bundle?")) {
        std::cout << "nothing changed\n";
        return 0;
      }
      polima::stop_server(root);
      std::cout << "stopped policy server\n";
    }
    const std::string active = polima::activate_bundle(store, selected);
    std::cout << "current -> " << active << "\n";
    const bool start = args.has("--start") ||
                       (guided && confirm("Start the selected bundle's policy server now?"));
    if (start) {
      const fs::path active_bundle = store / active;
      const auto description = polima::read_robot_description(active_bundle);
      const int port = args.get_int("--port", description.default_port);
      if (port <= 0)
        throw std::runtime_error("bundle declares no server port; pass --port N");
      const int pid = polima::start_server(root, root / "current", port);
      if (!pid)
        throw std::runtime_error("server failed to start; see var/log/server.log");
      std::cout << "started pid=" << pid << " port=" << port << " bundle=" << active << "\n";
    }
    return 0;
  }

  if (command == "server") {
    const auto state = polima::server_state(root);
    if (action.empty()) {
      std::cout << (state.running ? "running pid=" + std::to_string(state.pid) : "not running")
                << " current=" << (state.bundle.empty() ? "-" : state.bundle) << "\n";
      if (!interactive) return state.running ? 0 : 1;
      if (state.running) {
        if (!confirm("Stop the policy server?")) {
          std::cout << "nothing changed\n";
          return 0;
        }
        action = "stop";
      } else {
        if (!confirm("Start the active bundle's policy server?")) {
          std::cout << "nothing changed\n";
          return 0;
        }
        action = "start";
      }
    }
    if (action == "status") {
      std::cout << (state.running ? "running pid=" + std::to_string(state.pid) : "not running")
                << " current=" << (state.bundle.empty() ? "-" : state.bundle) << "\n";
      return state.running ? 0 : 1;
    }
    if (action == "stop") {
      [[maybe_unused]] const int control_lock = acquire_control_lock(root);
      std::cout << (polima::stop_server(root) ? "stopped\n" : "not running\n");
      return 0;
    }
    if (action == "start") {
      [[maybe_unused]] const int control_lock = acquire_control_lock(root);
      if (!args.has("--bundle")) {
        const fs::path store = args.get("--models-dir", (root / "models").string());
        const auto entries = polima::scan_store(store);
        const bool active = std::any_of(entries.begin(), entries.end(),
                                        [](const polima::StoreEntry& entry) {
                                          return entry.managed && entry.current;
                                        });
        if (!active) {
          if (!interactive)
            throw std::runtime_error(
                "no active bundle; run `polima activate <n|bundle-id>` first");
          std::cout << "No PoLiMa bundle is active. Choose one before starting the server.\n";
          const std::string selection = prompt_bundle(store);
          if (selection.empty()) {
            std::cout << "nothing started\n";
            return 0;
          }
          const std::string active_bundle = polima::activate_bundle(store, selection);
          std::cout << "current -> " << active_bundle << "\n";
        }
        bundle = root / "current";
      }
      const auto description = polima::read_robot_description(bundle);
      const int port = args.get_int("--port", description.default_port);
      if (port <= 0) throw std::runtime_error("bundle declares no server port");
      if (state.running) polima::stop_server(root);
      const int pid = polima::start_server(root, bundle, port);
      if (!pid) throw std::runtime_error("server failed to start; see var/log/server.log");
      std::cout << "started pid=" << pid << " port=" << port << " bundle="
                << fs::weakly_canonical(bundle).filename().string() << "\n";
      return 0;
    }
    throw std::runtime_error("usage: polima server [status|start|stop]");
  }

  // The bundle's robot client and camera preview import PoLiMa from this root's
  // python/ folder, so each install (POLIMA_ROOT) runs its own Python code.
  const fs::path python_root = root / "python";
  ::setenv("POLIMA_PYTHON_ROOT", python_root.c_str(), 0);
  const char* inherited = std::getenv("PYTHONPATH");
  ::setenv("PYTHONPATH", (python_root.string() + (inherited && *inherited ? ":" + std::string(inherited) : ""))
                             .c_str(), 1);
  auto description = polima::read_robot_description(bundle);
  if (!description.present)
    throw std::runtime_error("bundle has no robot description; repack it with polima compile");
  auto assignment = camera_assignment(description, args);
  const auto ports = polima::list_serial_ports();
  if (action.empty() || action == "status" || action == "doctor") {
    std::cout << "bundle\n  " << fs::weakly_canonical(bundle).filename().string()
              << " (port " << description.default_port << ")\n";
    print_robot(description, assignment, ports);
    const fs::path python = args.get("--lerobot-venv", "/media/nvme/lerobot") + "/bin/python";
    std::cout << "lerobot\n  " << (fs::exists(python) ? python.string() : "missing") << "\n";
    if (!action.empty() || !interactive) return 0;
    std::cout << "Choose the follower and camera devices before starting.\n";
    action = "run";
  }
  if (action == "preview") {
    [[maybe_unused]] const int control_lock = acquire_control_lock(root);
    for (const auto& [role, label] : description.roles) {
      (void)label;
      if (!assignment.assigned.count(role))
        throw std::runtime_error("camera " + role + " is unassigned; pass --" + role + "-camera");
      if (!fs::exists(assignment.assigned.at(role)))
        throw std::runtime_error("camera " + role + " not found at " + assignment.assigned.at(role));
    }
    print_robot(description, assignment, {});
    std::cout << "Starting camera-only preview. No policy server or follower arm will be started.\n";
    const fs::path venv = args.get("--lerobot-venv", "/media/nvme/lerobot");
    return polima::run_camera_preview(
        bundle, assignment, description.fourcc, args.get_int("--preview-port", 5001), venv);
  }
  if (action != "run" && action != "start")
    throw std::runtime_error("usage: polima robot [status|preview|run]");

  if (args.has("--fast") && args.has("--slow"))
    throw std::runtime_error("choose either --fast or --slow");
  if (args.has("--slow")) ::setenv("POLIMA_FAST", "0", 1);
  else if (args.has("--fast")) ::setenv("POLIMA_FAST", "1", 1);
  else ::setenv("POLIMA_FAST", "1", 0);
  const std::string speed = std::getenv("POLIMA_FAST");
  description.fps = (speed == "0" || speed == "false" || speed == "no") ? 15 : 30;
  if (args.has("--fps")) description.fps = args.get_int("--fps", description.fps);
  if (args.has("--actions-per-chunk"))
    description.actions_per_chunk =
        args.get_int("--actions-per-chunk", description.actions_per_chunk);
  if (args.has("--max-relative-target"))
    description.max_relative_target =
        args.get_int("--max-relative-target", description.max_relative_target);
  [[maybe_unused]] const int control_lock = acquire_control_lock(root);

  std::string robot_port = args.get("--robot-port", "");
  if (robot_port.empty()) {
    const std::string detected = ports.size() == 1 ? ports.front() : "";
    if (interactive && !args.has("--yes")) {
      robot_port = prompt_device("follower serial interface", ports, detected);
      if (robot_port.empty()) {
        std::cout << "nothing started\n";
        return 0;
      }
    } else {
      if (ports.size() != 1)
        throw std::runtime_error("select one follower arm with --robot-port");
      robot_port = ports.front();
    }
  }
  if (!fs::exists(robot_port))
    throw std::runtime_error("follower arm not found at " + robot_port);

  const auto cameras = polima::list_cameras();
  std::vector<std::string> selected_cameras;
  for (const auto& [role, label] : description.roles) {
    (void)label;
    const std::string option = "--" + role + "-camera";
    if (!args.has(option) && interactive && !args.has("--yes")) {
      const auto found = assignment.assigned.find(role);
      const std::string detected = found == assignment.assigned.end() ? "" : found->second;
      const std::string selected = prompt_device(role + " camera",
                                                 camera_paths(cameras, selected_cameras), detected);
      if (selected.empty()) {
        std::cout << "nothing started\n";
        return 0;
      }
      assignment.assigned[role] = selected;
    }
    if (!assignment.assigned.count(role))
      throw std::runtime_error("camera " + role + " is unassigned; pass --" + role + "-camera");
    if (!fs::exists(assignment.assigned.at(role)))
      throw std::runtime_error("camera " + role + " not found at " + assignment.assigned.at(role));
    if (std::find(selected_cameras.begin(), selected_cameras.end(), assignment.assigned.at(role)) !=
        selected_cameras.end())
      throw std::runtime_error("the same camera was selected for more than one role");
    selected_cameras.push_back(assignment.assigned.at(role));
  }

  const int port = args.get_int("--port", description.default_port);
  if (port <= 0) throw std::runtime_error("bundle declares no server port");
  if (!args.has("--yes")) {
    if (!interactive)
      throw std::runtime_error("robot control needs confirmation; pass --yes for non-interactive use");
    if (!confirm("Start the policy server and robot client? The follower arm may move")) {
      std::cout << "nothing started\n";
      return 0;
    }
  }
  const auto existing_server = polima::server_state(root);
  std::error_code bundle_code;
  const std::string requested_bundle =
      fs::weakly_canonical(bundle, bundle_code).filename().string();
  const bool reuse_server = existing_server.running && existing_server.accepting &&
      existing_server.port == port && !bundle_code &&
      existing_server.served_bundle == requested_bundle;
  if (existing_server.running && !reuse_server) polima::stop_server(root);
  const int server_pid = reuse_server
      ? existing_server.pid
      : polima::start_server(root, bundle, port);
  if (!server_pid) throw std::runtime_error("server failed to start; see var/log/server.log");

  std::cout << (reuse_server ? "reusing" : "started") << " server pid=" << server_pid
            << " port=" << port << "\n";
  print_robot(description, assignment, {robot_port});
  std::cout << "Starting the robot client in the foreground. Ctrl-C stops robot control; "
               "the policy server remains running.\n";
  const fs::path venv = args.get("--lerobot-venv", "/media/nvme/lerobot");
  const std::string task_id = args.get("--task", "0");
  ::setenv("POLIMA_TASK_ID", task_id.c_str(), 1);
  if (args.has("--front-camera-fps"))
    ::setenv("POLIMA_FRONT_CAMERA_FPS", args.get("--front-camera-fps", "").c_str(), 1);
  if (args.has("--wrist-camera-fps"))
    ::setenv("POLIMA_WRIST_CAMERA_FPS", args.get("--wrist-camera-fps", "").c_str(), 1);
  if (args.has("--no-auto-complete")) ::setenv("POLIMA_AUTO_COMPLETE", "0", 1);
  return polima::run_robot_client(bundle, description, robot_port, assignment,
                                  "127.0.0.1:" + std::to_string(port), venv);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2) {
      const std::string command = argv[1];
      // These robot actions are Python; the rest drive the arm natively.
      const bool robot_python = command == "robot" &&
          std::any_of(argv + 2, argv + argc, [](const char* value) {
            const std::string action = value;
            return action == "calibrate" || action == "export-improvement-manifest";
          });
      if (command == "studio" || command == "help" || robot_python)
        return run_python_command(command, argc, argv);
    }

    polima::Args args(argc, argv);
    if (args.has("--help") || args.has("-h")) {
      std::cout <<
        "polima-cli -- run a bundle once, files in, files out.\n\n"
        "  --bundle DIR        bundle directory (e.g. /media/nvme/polima/current)\n"
        "  --input-dir DIR     one <tensor>.f32 per wire input\n"
        "                      (default: <bundle>/fixtures/inputs)\n"
        "  --output FILE       write the result as float32\n"
        "  --dump-stages DIR   write every intermediate buffer, for bisecting\n"
        "  --verbose           per-step timings\n"
        "  --models-dir DIR    the DevKit's model store\n"
        "                      (default: $POLIMA_ROOT/models or /media/nvme/polima/models)\n"
        "  -h, --help          this message\n\n"
        "Device lifecycle:\n"
        "  polima activate [n|bundle-id] [--start] [--port N] [--models-dir DIR]\n"
        "  polima server status|start|stop [--bundle DIR] [--port N]\n"
        "  polima studio [status|start|stop|restart|enable|disable|logs|open|serve]\n"
        "  polima robot [status|preview|run] [--bundle DIR] [--robot-port PATH] [--yes]\n"
        "                         [--overhead-camera PATH --wrist-camera PATH]\n"
        "                         [--preview-port N]\n"
        "                         [--fps N --actions-per-chunk N --max-relative-target DEG]\n"
        "                         [--fast | --slow] [--task N] [--no-auto-complete]\n"
        "                         (run interactively picks detected serial and cameras)\n"
        "\nWith no --bundle it opens an interactive session over the model\n"
        "store: the model loads once and every command after that is fast.\n";
      return 0;
    }

    const int controlled = device_command(argc, argv, args);
    if (controlled >= 0) return controlled;

    // Studio uses this for fixture benchmarks. Holding the same advisory lock
    // as robot/preview commands makes a benchmark mutually exclusive with all
    // device control, including a manually-started terminal client.
    [[maybe_unused]] const int exclusive_control =
        args.has("--exclusive-control") ? acquire_control_lock(root_path()) : -1;
    if (args.has("--exclusive-control") && polima::server_state(root_path()).running)
      throw std::runtime_error("stop the policy server before running an exclusive benchmark");

    // No bundle named -> interactive session. `--bundle` keeps the one-shot
    // path, which scripts and the deploy smoke test depend on.
    const fs::path store = args.get(
        "--models-dir", (root_path() / "models").string());
    if (!args.has("--bundle") || args.has("--interactive"))
      return polima::repl(store, args.get("--bundle", ""), args.has("--verbose"));
    const fs::path bundle = args.get("--bundle");
    const bool verbose = args.has("--verbose");

    polima::Plan plan(bundle, verbose);
    if (args.has("--dump-stages"))
      plan.set_dump_stages(args.get("--dump-stages"));
    const auto& wire = plan.wire();

    const fs::path input_dir = args.get("--input-dir", (bundle / "fixtures/inputs").string());
    const fs::path output = args.get("--output", "");

    // Read one .f32 per declared request tensor. Names match the wire
    // description, so fixtures/inputs/image0.f32 feeds tensor "image0".
    std::map<std::string, std::vector<float>> staging;
    std::map<std::string, const float*> inputs;
    for (const auto& [name, elements] : wire.request_tensors) {
      const fs::path path = input_dir / (name + ".f32");
      staging[name] = polima::read_f32(path, elements);
      inputs[name] = staging[name].data();
      if (verbose) std::cout << "read " << path << " (" << elements << " floats)" << std::endl;
    }

    const auto started = std::chrono::steady_clock::now();
    const auto& result = plan.execute(inputs);
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();

    if (!output.empty()) {
      polima::write_f32(output, result);
      std::cout << "wrote " << output << " (" << result.size() << " floats)" << std::endl;
    }

    std::cout << "total_ms=" << total_ms << " count=" << result.size();
    for (const auto& timing : plan.stage_timings()) std::cout << " " << timing;
    std::cout << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
