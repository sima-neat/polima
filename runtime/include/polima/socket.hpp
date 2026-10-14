// Blocking socket helpers and argument parsing.
#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <optional>
#include <string>
#include <vector>

namespace polima {

inline void read_all(int fd, void* destination, size_t size) {
  auto* bytes = static_cast<uint8_t*>(destination);
  while (size) {
    const ssize_t count = recv(fd, bytes, size, 0);
    if (count <= 0) throw std::runtime_error("socket read failed");
    bytes += count;
    size -= static_cast<size_t>(count);
  }
}

inline void write_all(int fd, const void* source, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(source);
  while (size) {
    const ssize_t count = send(fd, bytes, size, MSG_NOSIGNAL);
    if (count <= 0) throw std::runtime_error("socket write failed");
    bytes += count;
    size -= static_cast<size_t>(count);
  }
}

// Bounds how long one connection may stall a blocking read or write, so an idle
// or vanished client cannot hold a single-threaded server indefinitely.
inline void set_io_timeout(int fd, int seconds) {
  timeval timeout{};
  timeout.tv_sec = seconds;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

inline int listen_on(int port, int backlog = 4) {
  int server = socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) throw std::runtime_error("socket() failed");
  int yes = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ||
      listen(server, backlog)) {
    close(server);
    throw std::runtime_error("bind/listen failed on port " + std::to_string(port));
  }
  return server;
}

class Args {
 public:
  Args(int argc, char** argv) : argc_(argc), argv_(argv) {}

  bool has(const std::string& key) const {
    for (int index = 1; index < argc_; ++index)
      if (key == argv_[index]) return true;
    return false;
  }

  // The fallback is optional rather than an empty-string sentinel, so
  // `get(key, "")` -- "absent is fine, and the default is empty", as `--output`
  // and an unselected model need -- differs from `get(key)` -- "required".
  std::string get(const std::string& key,
                  const std::optional<std::string>& fallback = std::nullopt) const {
    for (int index = 1; index + 1 < argc_; ++index)
      if (key == argv_[index]) return argv_[index + 1];
    if (fallback.has_value()) return *fallback;
    throw std::runtime_error("missing required argument " + key);
  }

  int get_int(const std::string& key, int fallback) const {
    for (int index = 1; index + 1 < argc_; ++index) {
      if (key != argv_[index]) continue;
      const std::string text = argv_[index + 1];
      size_t used = 0;
      int value = 0;
      try {
        value = std::stoi(text, &used);
      } catch (const std::exception&) {
        used = 0;
      }
      if (used == 0 || used != text.size())
        throw std::runtime_error(key + " expects a whole number, got '" + text + "'");
      return value;
    }
    return fallback;
  }

 private:
  int argc_;
  char** argv_;
};

}  // namespace polima
