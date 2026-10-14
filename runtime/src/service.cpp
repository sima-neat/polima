#include "polima/service.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace polima {
namespace fs = std::filesystem;
namespace {

fs::path pid_path(const fs::path& root) { return root / "var" / "run" / "server.pid"; }
fs::path log_path(const fs::path& root) { return root / "var" / "log" / "server.log"; }

bool alive(int pid) {
  if (pid <= 0 || ::kill(pid, 0) != 0) return false;
  std::ifstream status("/proc/" + std::to_string(pid) + "/stat");
  std::string stat;
  std::getline(status, stat);
  const auto end = stat.rfind(')');
  // An exited child may remain a zombie until its parent reaps it.
  return end == std::string::npos || end + 2 >= stat.size() || stat[end + 2] != 'Z';
}

// The pid file survives reboots, so its pid may now belong to another program.
bool is_server(int pid) {
  std::ifstream command("/proc/" + std::to_string(pid) + "/cmdline", std::ios::binary);
  std::string program;
  std::getline(command, program, char{0});
  return fs::path(program).filename() == "polima_server";
}

bool listening(int port) {
  const int socket = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket < 0) return false;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const bool connected =
      ::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
  ::close(socket);
  return connected;
}

}  // namespace

ServerState server_state(const fs::path& root) {
  ServerState state;
  std::ifstream stream(pid_path(root));
  if (stream) stream >> state.pid;
  state.running = alive(state.pid) && is_server(state.pid);
  if (!state.running) state.pid = 0;

  std::error_code code;
  if (state.running) {
    std::ifstream command("/proc/" + std::to_string(state.pid) + "/cmdline",
                          std::ios::binary);
    std::vector<std::string> arguments;
    std::string argument;
    while (std::getline(command, argument, char{0})) arguments.push_back(argument);
    for (size_t index = 0; index + 1 < arguments.size(); ++index) {
      if (arguments[index] == "--bundle") {
        const fs::path served = fs::weakly_canonical(arguments[++index], code);
        if (!code) state.served_bundle = served.filename().string();
      } else if (arguments[index] == "--port") {
        try {
          state.port = std::stoi(arguments[++index]);
        } catch (...) {
          state.port = 0;
        }
      }
    }
    state.accepting = state.port > 0 && listening(state.port);
  }

  const fs::path current = root / "current";
  if (fs::exists(current, code)) {
    const fs::path resolved = fs::canonical(current, code);
    if (!code) state.bundle = resolved.filename().string();
  }
  return state;
}

bool stop_server(const fs::path& root, double timeout_s) {
  const ServerState state = server_state(root);
  if (!state.running) {
    std::error_code code;
    fs::remove(pid_path(root), code);
    return false;
  }

  ::kill(state.pid, SIGTERM);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_s * 1000));
  while (std::chrono::steady_clock::now() < deadline) {
    if (!alive(state.pid)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // A server mid-inference will not see SIGTERM until the MLA call returns, so
  // the wait above is not optional -- but it cannot be unbounded either.
  if (alive(state.pid)) {
    throw std::runtime_error("Policy server did not stop cleanly; model switch aborted. "
                             "Stop robot motion and recover the accelerator from an authorized terminal.");
  }

  std::error_code code;
  fs::remove(pid_path(root), code);
  return true;
}

int start_server(const fs::path& root, const fs::path& bundle, int port) {
  const fs::path binary = root / "bin" / "polima_server";
  if (!fs::exists(binary)) return 0;
  // Otherwise the new server fails to bind while the readiness check below
  // connects to the other process and reports success.
  if (listening(port))
    throw std::runtime_error("port " + std::to_string(port) +
                             " is already in use by another process; stop it or pass --port N");

  std::error_code code;
  fs::create_directories(log_path(root).parent_path(), code);
  fs::create_directories(pid_path(root).parent_path(), code);

  const pid_t pid = ::fork();
  if (pid < 0) return 0;
  if (pid == 0) {
    // Detach fully: new session, no controlling terminal, stdin closed. The
    // shell version of this needed `< /dev/null` or the server inherited the
    // ssh channel and held the connection open; the same applies here.
    ::setsid();
    const int null = ::open("/dev/null", O_RDONLY);
    if (null >= 0) { ::dup2(null, STDIN_FILENO); ::close(null); }
    const int log = ::open(log_path(root).c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (log >= 0) {
      ::dup2(log, STDOUT_FILENO);
      ::dup2(log, STDERR_FILENO);
      ::close(log);
    }
    const std::string port_text = std::to_string(port);
    ::execl(binary.c_str(), "polima_server", "--bundle", bundle.c_str(),
            "--port", port_text.c_str(), static_cast<char*>(nullptr));
    ::_exit(127);
  }

  std::ofstream stream(pid_path(root));
  stream << pid << "\n";
  stream.close();

  // Model loading happens before bind and can take several seconds. Returning
  // immediately made `robot run` race the server and report a false connection
  // failure, so starting is not considered successful until the socket opens.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!alive(pid)) return 0;
    if (listening(port)) return pid;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  ::kill(pid, SIGTERM);
  return 0;
}

}  // namespace polima
