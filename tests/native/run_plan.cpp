// Runs a bundle's plan.json against the CPU llima double (tests/native/llima_mock),
// reading <name>.u8 for uint8 wire tensors and <name>.f32 otherwise:
//   run_plan <bundle> <inputs_dir> <out.f32> [repeats=2] [dump_dir]
// Writes the first result; exits non-zero on any Plan error or if a repeated
// request differs (cached uploads and request isolation).
#include <cstring>
#include <iostream>

#include "polima/plan.hpp"
#include "polima/sidecar.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: run_plan bundle inputs out [repeats] [dump]\n";
    return 2;
  }
  try {
    polima::Plan plan(argv[1]);
    if (argc > 5) plan.set_dump_stages(argv[5]);
    std::map<std::string, std::vector<float>> staged;
    std::map<std::string, const float*> inputs;
    for (const auto& [name, elements] : plan.wire().request_tensors) {
      staged[name] = polima::read_wire_input(argv[2], name, elements, plan.wire());
      inputs[name] = staged[name].data();
    }
    const int repeats = argc > 4 ? std::stoi(argv[4]) : 2;
    std::vector<float> first;
    for (int r = 0; r < repeats; ++r) {
      const auto& result = plan.execute(inputs);
      if (r == 0) {
        first = result;
      } else if (std::memcmp(first.data(), result.data(), first.size() * 4) != 0) {
        std::cerr << "repeat " << r << " differs from the first run\n";
        return 3;
      }
    }
    polima::write_f32(argv[3], first);
    for (const auto& [k, v] : simaai::llima::mock_counters()) std::cout << k << "=" << v << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
