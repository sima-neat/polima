// polima_server -- one TCP inference server for every policy.
//
// Everything policy-specific comes from the bundle: framing from plan.json's
// "wire" block, ELF paths and tensor sizes from bundle.json.
//
//   polima_server --bundle /media/nvme/polima/current --port 8092
//
// Wire format: a 16-byte request header (magic, version, request_id, flags)
// followed by each request tensor as little-endian float32, answered with a
// 24-byte response header (magic, version, request_id, status, latency_ms,
// count) followed by the result.

#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "polima/plan.hpp"
#include "polima/socket.hpp"

namespace {

using Clock = std::chrono::steady_clock;

#pragma pack(push, 1)
struct RequestHeader {
  uint32_t magic, version, request_id, flags;
};
struct ResponseHeader {
  uint32_t magic, version, request_id, status;
  float latency_ms;
  uint32_t count;
};
#pragma pack(pop)

std::atomic<bool> g_running{true};

void handle_signal(int) { g_running = false; }

void install_stop_handlers() {
  // sigaction with sa_flags = 0, NOT std::signal(). On glibc std::signal()
  // installs a BSD-style handler with SA_RESTART, which transparently restarts
  // the blocking accept() below instead of letting it fail with EINTR. The
  // handler does set g_running, but the loop never gets back to its condition
  // to notice, so the server would ignore SIGTERM until a client connects.
  //
  // A SIGKILLed server never releases its DMA-coherent buffers, so each one
  // fragments the MLA's CMA pool until loads start failing with
  // MLA_LOAD_FAILED on a DevKit with a gigabyte free. Clearing SA_RESTART is
  // what makes a graceful stop -- and an unfragmented pool -- possible.
  struct sigaction action{};
  action.sa_handler = handle_signal;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  ::sigaction(SIGINT, &action, nullptr);
  ::sigaction(SIGTERM, &action, nullptr);
  std::signal(SIGPIPE, SIG_IGN);
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    polima::Args args(argc, argv);
    if (args.has("--help") || args.has("-h")) {
      std::cout <<
        "polima-server -- serve a bundle over the policy's TCP wire protocol.\n\n"
        "  --bundle DIR   bundle directory (e.g. /media/nvme/polima/current)\n"
        "  --port N       listen port (default: the policy's wire port)\n"
        "  --verbose      per-step timings on every request\n"
        "  -h, --help     this message\n";
      return 0;
    }
    const std::string bundle = args.get("--bundle");
    const bool verbose = args.has("--verbose");

    polima::Plan plan(bundle, verbose);
    const auto& wire = plan.wire();
    const int port = args.get_int("--port", wire.default_port);
    if (port <= 0) throw std::runtime_error("no --port given and the bundle declares none");

    // Read the socket straight into the plan's own input buffers. Staging them
    // separately would copy the payload twice -- 7.4 MB per request for ACT's
    // two 480x640x3 images.
    std::vector<std::pair<float*, size_t>> input_slots;
    input_slots.reserve(wire.request_tensors.size());
    for (const auto& [name, elements] : wire.request_tensors)
      input_slots.emplace_back(plan.input_buffer(name), elements);

    install_stop_handlers();

    const int server = polima::listen_on(port);
    std::cout << "READY port=" << port << " bundle=" << plan.bundle_id()
              << " policy=" << plan.policy() << " magic=0x" << std::hex << wire.magic << std::dec
              << std::endl;

    while (g_running) {
      const int client = accept(server, nullptr, nullptr);
      // EINTR here is the stop signal arriving; `continue` re-tests g_running.
      if (client < 0) continue;

      RequestHeader header{};
      try {
        polima::read_all(client, &header, sizeof(header));
        if (header.magic != wire.magic || header.version != wire.version)
          throw std::runtime_error("protocol mismatch");

        for (const auto& [destination, elements] : input_slots)
          polima::read_all(client, destination, elements * sizeof(float));

        const auto started = Clock::now();
        const auto& result = plan.execute(header.flags);
        const auto total_ms = static_cast<float>(elapsed_ms(started));

        std::cout << "request=" << header.request_id << " total_ms=" << total_ms
                  << " count=" << result.size();
        for (const auto& timing : plan.stage_timings()) std::cout << " " << timing;
        std::cout << std::endl;

        ResponseHeader response{wire.magic, wire.version, header.request_id, 0, total_ms,
                                static_cast<uint32_t>(result.size())};
        polima::write_all(client, &response, sizeof(response));
        polima::write_all(client, result.data(), result.size() * sizeof(float));
      } catch (const std::exception& error) {
        std::cerr << "request error: " << error.what() << std::endl;
        ResponseHeader response{wire.magic, wire.version, header.request_id, 1, 0.0f, 0};
        try {
          polima::write_all(client, &response, sizeof(response));
        } catch (...) {
        }
      }
      close(client);
    }

    close(server);
    std::cout << "shutting down" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
