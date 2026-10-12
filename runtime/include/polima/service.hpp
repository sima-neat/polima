// Managing polima_server from inside the interactive session.
//
// The session and the server are separate processes competing for one
// accelerator, so `unload` that frees only the session's models leaves the MLA
// just as occupied as before. To actually release it, the server has to stop --
// polima_server holds its Plan for its whole life by design, since a serving
// process that can drop its model mid-request is worse than one you restart.
//
// Paths match polima/deploy/service.py exactly (var/run/server.pid,
// var/log/server.log, bin/polima_server), because both manage the same process
// and disagreeing about where the pid file lives is how you end up with two
// servers on one port.
#pragma once

#include <filesystem>
#include <string>

namespace polima {

struct ServerState {
  int pid = 0;
  bool running = false;
  std::string bundle;      // what `current` points at, i.e. what it would serve
  std::string served_bundle;  // canonical bundle loaded by the live process
  int port = 0;
  bool accepting = false;
};

// Reads var/run/server.pid and checks the process is alive.
ServerState server_state(const std::filesystem::path& root);

// SIGTERM, wait, then SIGKILL. Returns true if something was stopped.
bool stop_server(const std::filesystem::path& root, double timeout_s = 5.0);

// fork + setsid + exec polima_server, appending to var/log/server.log.
// Returns the pid, or 0 on failure.
int start_server(const std::filesystem::path& root, const std::filesystem::path& bundle,
                 int port);

}  // namespace polima
