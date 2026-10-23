#include "polima/plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace polima {

std::shared_ptr<RuntimeGuard> acquire_mla_runtime() {
  // Connect once for the process and disconnect when the last Plan is gone.
  // The weak_ptr is what makes the second half work: a released Plan drops the
  // refcount, and only a drop to zero tears the runtime down.
  static std::mutex lock;
  static std::weak_ptr<RuntimeGuard> shared;
  const std::lock_guard<std::mutex> held(lock);
  if (auto existing = shared.lock()) return existing;
  auto created = std::make_shared<RuntimeGuard>();
  shared = created;
  return created;
}

namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  json value;
  stream >> value;
  return value;
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void report_phase(const std::string& graph, const std::string& out,
                  const RunTiming& timing, double prepare_ms) {
  std::cerr << "POLIMA_PHASE " << json{
      {"graph", graph}, {"out", out}, {"prepare_ms", prepare_ms},
      {"encode_ms", timing.encode_ms}, {"upload_ms", timing.upload_ms},
      {"execute_ms", timing.execute_ms}, {"download_ms", timing.download_ms},
      {"decode_ms", timing.decode_ms},
      {"stage_names", timing.stage_names},
      {"stage_execute_ms", timing.stage_execute_ms}}.dump() << '\n';
}

}  // namespace

std::vector<float> read_wire_input(const std::filesystem::path& directory,
                                   const std::string& name, size_t elements,
                                   const WireDescription& wire) {
  const auto decode = wire.decode.find(name);
  const auto frame = directory / (name + ".u8");
  if (decode == wire.decode.end() || !std::filesystem::is_regular_file(frame))
    return read_f32(directory / (name + ".f32"), elements);
  std::ifstream stream(frame, std::ios::binary);
  std::vector<uint8_t> bytes(elements);
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(elements));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error(frame.string() + " must hold exactly " + std::to_string(elements) +
                             " bytes");
  std::vector<float> values(elements);
  decode->second.apply(bytes.data(), values.data(), elements);
  return values;
}

namespace {

[[noreturn]] void throw_nonfinite(const std::string& stage, size_t index) {
  throw std::runtime_error(stage + " produced a non-finite value at index " +
                           std::to_string(index));
}

// How graph outputs are checked for non-finite values. Every mode checks every
// value of every graph output and of the final result; they differ in cost.
//   fused  (default) the bf16/INT8 decode reports the first non-finite value
//          in the same pass; outputs it cannot vouch for get a NEON scan.
//   neon   a separate NEON scan of every graph output after the decode.
//   scalar a separate std::isfinite loop (3.6 ns/value).
// POLIMA_FINITE_SCAN selects; POLIMA_FULL_FINITE_SCAN=1 is an alias for neon.
enum class FiniteScan { Fused, Neon, Scalar };

FiniteScan finite_scan_mode() {
  static const FiniteScan mode = [] {
    const char* full = std::getenv("POLIMA_FULL_FINITE_SCAN");
    if (full && *full && std::string(full) != "0") return FiniteScan::Neon;
    const char* text = std::getenv("POLIMA_FINITE_SCAN");
    const std::string value = text ? text : "";
    if (value == "neon") return FiniteScan::Neon;
    if (value == "scalar") return FiniteScan::Scalar;
    return FiniteScan::Fused;
  }();
  return mode;
}

void require_finite(const std::string& stage, const std::vector<float>& values) {
  if (finite_scan_mode() == FiniteScan::Scalar) {
    for (size_t index = 0; index < values.size(); ++index)
      if (!std::isfinite(values[index])) throw_nonfinite(stage, index);
    return;
  }
  const size_t index = kernels::first_nonfinite(values.data(), values.size());
  if (index != values.size()) throw_nonfinite(stage, index);
}

// A graph output whose decode already located the first non-finite value
// (or `size` for none) needs no second pass in the default mode.
void require_finite_checked(const std::string& stage, const std::vector<float>& values,
                            bool checked, size_t first_nonfinite) {
  if (checked && finite_scan_mode() == FiniteScan::Fused) {
    if (first_nonfinite < values.size()) throw_nonfinite(stage, first_nonfinite);
    return;
  }
  require_finite(stage, values);
}

// The device runtime binds persistent buffers in lexicographic order of
// their binding names (output_0, output_1, output_10, ..., output_2). A graph
// whose bundle.json lists "binding" names in any other order would be bound
// to the wrong buffers once it has more than ten bindings; refuse it.
void require_binding_order(const std::string& graph, const nlohmann::json& bindings) {
  for (const char* side : {"inputs", "outputs"}) {
    if (!bindings.contains(side)) continue;
    std::vector<std::string> names;
    for (const auto& b : bindings.at(side))
      if (b.contains("binding")) names.push_back(b.at("binding").get<std::string>());
    if (!std::is_sorted(names.begin(), names.end()))
      throw std::runtime_error("graph '" + graph + "' lists persistent " + side +
                               " out of the runtime's binding-name order");
  }
}

struct GraphInfo {
  std::string name;
  std::filesystem::path elf;
  size_t inputs = 0;
  size_t outputs = 0;
  DramLayout output_layout = DramLayout::Plain;
  size_t logical_width = 0;
  size_t logical_channels = 0;
  std::string external_layout = "compiler";
};

}  // namespace

Opcode parse_opcode(const std::string& name) {
  if (name == "run_elf") return Opcode::RunElf;
  if (name == "run_elf_chain") return Opcode::RunElfChain;
  if (name == "run_dit_chain") return Opcode::RunDiTChain;
  if (name == "run_device_chain") return Opcode::RunDeviceChain;
  if (name == "run_bound_chain") return Opcode::RunBoundChain;
  if (name == "pack") return Opcode::Pack;
  if (name == "slice") return Opcode::Slice;
  if (name == "gather_strided") return Opcode::GatherStrided;
  if (name == "pixel_unshuffle") return Opcode::PixelUnshuffle;
  if (name == "scale") return Opcode::Scale;
  if (name == "clip_unit") return Opcode::ClipUnit;
  if (name == "matvec") return Opcode::MatVec;
  if (name == "sincos_time") return Opcode::SincosTime;
  if (name == "sincos_state") return Opcode::SincosState;
  if (name == "euler") return Opcode::Euler;
  if (name == "normalize") return Opcode::Normalize;
  if (name == "denormalize") return Opcode::Denormalize;
  throw std::runtime_error("unknown opcode '" + name + "' in plan.json");
}

const char* opcode_name(Opcode opcode) {
  switch (opcode) {
    case Opcode::RunElf: return "run_elf";
    case Opcode::RunElfChain: return "run_elf_chain";
    case Opcode::RunDiTChain: return "run_dit_chain";
    case Opcode::RunDeviceChain: return "run_device_chain";
    case Opcode::RunBoundChain: return "run_bound_chain";
    case Opcode::Pack: return "pack";
    case Opcode::Slice: return "slice";
    case Opcode::GatherStrided: return "gather_strided";
    case Opcode::PixelUnshuffle: return "pixel_unshuffle";
    case Opcode::Scale: return "scale";
    case Opcode::ClipUnit: return "clip_unit";
    case Opcode::MatVec: return "matvec";
    case Opcode::SincosTime: return "sincos_time";
    case Opcode::SincosState: return "sincos_state";
    case Opcode::Euler: return "euler";
    case Opcode::Normalize: return "normalize";
    case Opcode::Denormalize: return "denormalize";
  }
  return "?";
}

Plan::Plan(const std::filesystem::path& bundle_root, bool verbose)
    : root_(std::filesystem::canonical(bundle_root)), verbose_(verbose),
      collect_timings_(verbose) {
  const json manifest = read_json(root_ / "bundle.json");
  const json plan = read_json(root_ / "plan.json");

  bundle_id_ = manifest.value("bundle_id", "");
  policy_ = manifest.value("policy", "");
  if (manifest.contains("smoke")) {
    const json& smoke = manifest.at("smoke");
    smoke_cosine_min_ = smoke.value("cosine_min", smoke_cosine_min_);
    smoke_mean_abs_max_ = smoke.value("mean_abs_max", smoke_mean_abs_max_);
  }
  result_ = plan.at("result").get<std::string>();

  // Allocate every buffer once, at load time. Nothing on the request path
  // resizes or reallocates.
  for (auto item = plan.at("buffers").begin(); item != plan.at("buffers").end(); ++item) {
    const auto size = item.value().get<size_t>();
    buffer_sizes_[item.key()] = size;
    buffers_[item.key()] = std::vector<float>(size, 0.0f);
  }
  if (!buffer_sizes_.count(result_))
    throw std::runtime_error("plan result '" + result_ + "' is not a declared buffer");

  if (plan.contains("wire")) {
    const json& wire = plan.at("wire");
    wire_.magic = wire.value("magic", 0u);
    wire_.version = wire.value("version", 1u);
    wire_.default_port = wire.value("default_port", 0);
    wire_.response_elements = wire.value("response_elements", size_t{0});
    for (const auto& tensor : wire.value("request_tensors", json::array())) {
      const auto name = tensor.at("name").get<std::string>();
      const auto elements = tensor.at("elements").get<size_t>();
      // The server reads requests straight into these buffers.
      const auto declared = buffer_sizes_.find(name);
      if (declared == buffer_sizes_.end())
        throw std::runtime_error("wire request tensor '" + name + "' is not a declared buffer");
      if (elements > declared->second)
        throw std::runtime_error("wire request tensor '" + name + "' has " +
                                 std::to_string(elements) + " elements but its buffer holds " +
                                 std::to_string(declared->second));
      wire_.request_tensors.emplace_back(name, elements);
      const auto dtype = tensor.value("dtype", std::string("float32"));
      if (dtype == "uint8") {
        const json& decode = tensor.at("decode");
        const float scale = decode.at("scale").get<float>();
        const auto mean = decode.at("mean").get<std::vector<float>>();
        const auto stdev = decode.at("std").get<std::vector<float>>();
        if (mean.empty() || mean.size() != stdev.size() || elements % mean.size() ||
            !(scale > 0.0f))
          throw std::runtime_error("wire request tensor '" + name + "' has an invalid uint8 decode");
        WireDecode table{mean.size(), std::vector<float>(mean.size() * 256)};
        for (size_t c = 0; c < mean.size(); ++c) {
          if (!(stdev[c] != 0.0f))
            throw std::runtime_error("wire request tensor '" + name + "' has a zero std");
          for (int x = 0; x < 256; ++x)
            table.table[c * 256 + x] = (static_cast<float>(x) / scale - mean[c]) / stdev[c];
        }
        wire_.decode.emplace(name, std::move(table));
      } else if (dtype != "float32") {
        throw std::runtime_error("wire request tensor '" + name + "' has unsupported dtype " + dtype);
      }
    }
  }

  sidecars_ = Sidecars(root_ / "constants");

  std::set<std::string> chained_graphs;
  std::set<std::string> regular_graphs;
  for (const auto& entry : plan.at("steps")) {
    const auto op = entry.at("op").get<std::string>();
    const auto args = entry.value("args", json::object());
    if (op == "run_elf") regular_graphs.insert(args.at("graph").get<std::string>());
    if (op == "run_elf_chain" || op == "run_dit_chain" || op == "run_bound_chain")
      for (const auto& graph : args.at("graphs"))
        chained_graphs.insert(graph.get<std::string>());
    if (op == "run_device_chain")
      for (const auto& call : args.at("calls"))
        chained_graphs.insert(call.at("graph").get<std::string>());
    if (op == "run_bound_chain")
      for (const auto& graph : args.value("prelude", json::array()))
        chained_graphs.insert(graph.get<std::string>());
  }
  for (const auto& name : chained_graphs)
    if (regular_graphs.count(name))
      throw std::runtime_error("graph '" + name +
                               "' cannot be used both directly and in a shared chain");

  // Graph metadata drives both ordinary Runners and shared chains. Chained
  // graphs are not also loaded as ordinary Runners: GR00T has 25 of them and
  // loading every ELF twice would waste substantial device memory.
  guard_ = acquire_mla_runtime();
  std::map<std::string, GraphInfo> graph_infos;
  std::map<std::string, std::function<std::unique_ptr<Runner>()>> runner_factories;
  std::map<std::string, DeviceGraph> device_graphs;
  for (const auto& graph : manifest.at("graphs")) {
    const auto name = graph.at("name").get<std::string>();
    const auto elf = root_ / graph.at("elf").get<std::string>();
    if (!std::filesystem::exists(elf))
      throw std::runtime_error("bundle.json names a missing ELF: " + elf.string());

    GraphInfo info{name, elf, graph.at("input_elements").get<size_t>(),
                   graph.at("output_elements").get<size_t>(),
                   parse_dram_layout(graph.value("dram_layout", "plain")),
                   graph.value("logical_width", size_t{0}),
                   graph.value("logical_channels", size_t{0}),
                   graph.value("external_dram_layout", "compiler")};
    graph_infos[name] = info;
    {
      DeviceGraph device{name, elf, info.inputs, info.outputs,
                         graph.value("persistent_input_elements", std::vector<size_t>{}),
                         info.external_layout, info.output_layout == DramLayout::Plain,
                         graph.contains("int8_inputs")};
      if (device.bindings.empty()) device.bindings.push_back(info.inputs);
      device_graphs[name] = std::move(device);
    }
    std::vector<TensorQuantization> int8_inputs;
    TensorQuantization int8_output;
    if (graph.contains("int8_inputs")) {
      if (chained_graphs.count(name))
        throw std::runtime_error("INT8 graph cannot bypass boundary conversion in a shared chain");
      const auto decode_quant = [](const auto& q) {
        return TensorQuantization{q.at("elements").template get<size_t>(),
                                  q.at("multiplier").template get<float>(),
                                  q.at("zero_point").template get<int>()};
      };
      for (const auto& q : graph.at("int8_inputs")) int8_inputs.push_back(decode_quant(q));
      int8_output = decode_quant(graph.at("int8_output"));
    }
    runner_factories[name] = [=] {
      return std::make_unique<Runner>(name, elf, info.inputs, info.outputs, info.output_layout,
                                      info.logical_width, info.logical_channels, int8_inputs,
                                      int8_output);
    };
    if (!chained_graphs.count(name)) runners_[name] = runner_factories.at(name)();
    if (verbose_)
      std::cout << "loaded " << name << " (" << graph.value("elf_bytes", 0) << " bytes)"
                << std::endl;
  }

  if (plan.contains("device_buffers")) {
    device_arena_ = std::make_unique<DeviceArena>();
    for (auto item = plan.at("device_buffers").begin(); item != plan.at("device_buffers").end();
         ++item)
      device_arena_->declare(item.key(), item.value().get<size_t>());
  }

  // Decode the steps once, so the request path does no JSON work.
  size_t step_index = 0;
  for (const auto& entry : plan.at("steps")) {
    Step step;
    step.opcode = parse_opcode(entry.at("op").get<std::string>());
    step.out = entry.at("out").get<std::string>();
    step.raw = entry.dump();
    const json args = entry.value("args", json::object());

    if (args.contains("graph")) step.graph = args.at("graph").get<std::string>();
    if (args.contains("graphs"))
      for (const auto& name : args.at("graphs")) step.graphs.push_back(name.get<std::string>());
    if (args.contains("src")) step.source = args.at("src").get<std::string>();
    if (args.contains("prelude"))
      for (const auto& name : args.at("prelude")) step.prelude.push_back(name.get<std::string>());
    if (args.contains("bind"))
      for (const auto& name : args.at("bind")) step.binds.push_back(name.get<std::string>());
    step.carry = args.value("carry", std::string{});
    if (args.contains("in"))
      for (const auto& name : args.at("in")) step.inputs.push_back(name.get<std::string>());
    if (args.contains("parts"))
      for (const auto& part : args.at("parts")) {
        PackPart decoded;
        decoded.source = part.at("src").get<std::string>();
        decoded.destination_offset = part.at("dst_offset").get<size_t>();
        decoded.count = part.at("count").get<size_t>();
        decoded.source_offset = part.value("src_offset", size_t{0});
        decoded.source_stride = part.value("source_stride", size_t{0});
        decoded.selector_count = part.value("selector_count", size_t{0});
        decoded.sidecar = part.value("sidecar", false);
        if (decoded.source_stride && !decoded.selector_count)
          throw std::runtime_error("indexed pack part has no selector_count");
        step.parts.push_back(std::move(decoded));
      }
    if (args.contains("group_widths"))
      step.group_widths = args.at("group_widths").get<std::vector<size_t>>();
    step.offset = args.value("offset", size_t{0});
    step.stride = args.value("stride", size_t{0});
    step.take = args.value("take", size_t{0});
    step.count = args.value("count", size_t{0});
    step.size = args.value("size", size_t{0});
    step.destination_stride = args.value("dst_stride", size_t{0});
    step.destination_offset = args.value("dst_offset", size_t{0});
    step.clear = args.value("clear", false);
    // grid remains a reader fallback for version-1 square plans.
    step.height = args.value("height", args.value("grid", size_t{0}));
    step.width = args.value("width", step.height);
    step.channels = args.value("channels", size_t{0});
    step.factor = args.value("factor", size_t{0});
    step.scalar = args.value("scalar", 0.0f);
    step.min_period = args.value("min_period", 0.0f);
    step.max_period = args.value("max_period", 0.0f);
    step.weights = args.value("weights", std::string{});
    step.bias = args.value("bias", std::string{});
    step.mean = args.value("mean", std::string{});
    step.std_dev = args.value("std", std::string{});
    if (args.contains("clip_min") || args.contains("clip_max")) {
      step.clip = true;
      step.clip_min = args.value("clip_min", -3.4e38f);
      step.clip_max = args.value("clip_max", 3.4e38f);
    }
    step.rows = args.value("rows", size_t{0});
    step.cols = args.value("cols", size_t{0});

    if (!buffer_sizes_.count(step.out))
      throw std::runtime_error("step writes undeclared buffer '" + step.out + "'");
    if (step.opcode == Opcode::RunElf && !runners_.count(step.graph))
      throw std::runtime_error("step runs graph '" + step.graph + "' with no ELF in bundle.json");
    if (step.opcode == Opcode::RunDiTChain) {
      if (step.inputs.size()!=5 || step.graphs.size()!=16)
        throw std::runtime_error("run_dit_chain requires five buffers and sixteen ordered partitions");
      std::array<size_t,5> counts{};
      for(size_t i=0;i<5;++i) counts[i]=buffer_sizes_.at(step.inputs[i]);
      if(buffer_sizes_.at(step.out)!=counts[0]) throw std::runtime_error("DiT output size mismatch");
      std::vector<SharedStage> stages;
      for(size_t i=0;i<step.graphs.size();++i) {
        const auto number=[](size_t n){return (n<10 ? "0" : "")+std::to_string(n);};
        const auto expected="dit_blocks_"+number(2*i)+"_"+number(2*i+1);
        const auto& name=step.graphs[i];
        if(name!=expected)throw std::runtime_error("DiT graph order mismatch");
        const auto& g=graph_infos.at(name);
        if(g.external_layout!="HWC" || g.output_layout!=DramLayout::Plain)
          throw std::runtime_error("DiT chain requires plain HWC boundaries");
        const auto& gm=*std::find_if(manifest.at("graphs").begin(),manifest.at("graphs").end(),
            [&](const json& v){return v.at("name")==name;});
        if(gm.value("persistent_input_elements",std::vector<size_t>{})!=
            std::vector<size_t>{counts[0],counts[1],counts[2],counts[3]})
          throw std::runtime_error("Missing or incorrect persistent DiT binding contract");
        stages.push_back({name,g.elf,g.inputs,g.outputs});
      }
      // Identical graph sequence and geometry share one loaded chain across timesteps.
      step.chain="dit";
      for(const auto& name:step.graphs)step.chain+="/"+name;
      for(auto n:counts)step.chain+="/"+std::to_string(n);
      if(!dit_chains_.count(step.chain))
        dit_chains_[step.chain]=std::make_unique<DiTRunnerChain>("dit",stages,counts);
    }
    if (step.opcode == Opcode::RunDeviceChain) {
      if (!device_arena_)
        throw std::runtime_error("run_device_chain needs plan-level device_buffers: " + step.raw);
      step.chain = "device_" + std::to_string(step_index) + "_" + args.value("label", step.out);
      device_chains_[step.chain] = std::make_unique<DeviceChain>(
          step.chain, args, *device_arena_, device_graphs, buffers_);
    }
    if (step.opcode == Opcode::RunBoundChain) {
      if (step.binds.size() != step.inputs.size() || step.carry.empty())
        throw std::runtime_error("run_bound_chain needs bind[] matching in[] and a carry: " + step.raw);
      const auto bound = [&](const std::string& name) {
        const auto& gm = *std::find_if(manifest.at("graphs").begin(), manifest.at("graphs").end(),
            [&](const json& v) { return v.at("name") == name; });
        if (!gm.contains("persistent_bindings"))
          throw std::runtime_error("graph '" + name + "' has no persistent_bindings");
        BoundGraph g{name, graph_infos.at(name).elf, {}, {}};
        for (const auto& b : gm.at("persistent_bindings").at("inputs"))
          g.inputs.emplace_back(b.at("name").get<std::string>(), b.at("elements").get<size_t>());
        for (const auto& b : gm.at("persistent_bindings").at("outputs"))
          g.outputs.emplace_back(b.at("name").get<std::string>(), b.at("elements").get<size_t>());
        require_binding_order(name, gm.at("persistent_bindings"));
        return g;
      };
      std::vector<BoundGraph> prelude, graphs;
      for (const auto& name : step.prelude) prelude.push_back(bound(name));
      for (const auto& name : step.graphs) graphs.push_back(bound(name));
      std::vector<size_t> counts;
      for (const auto& name : step.inputs) counts.push_back(buffer_sizes_.at(name));
      const auto carry = std::find(step.binds.begin(), step.binds.end(), step.carry);
      if (carry == step.binds.end() ||
          buffer_sizes_.at(step.out) != counts[carry - step.binds.begin()])
        throw std::runtime_error("run_bound_chain output must match the carried binding");
      // Identical graph sequence and geometry share one loaded chain across timesteps.
      step.chain = "bound";
      for (const auto& name : step.prelude) step.chain += "/" + name;
      for (const auto& name : step.graphs) step.chain += "/" + name;
      for (size_t i = 0; i < counts.size(); ++i) step.chain += "/" + step.binds[i] + ":" + std::to_string(counts[i]);
      if (!bound_chains_.count(step.chain))
        bound_chains_[step.chain] = std::make_unique<BoundRunnerChain>(
            "bound", step.binds, counts, step.carry, prelude, graphs);
    }
    if (step.opcode == Opcode::RunElfChain) {
      if (step.graphs.size() < 2)
        throw std::runtime_error("run_elf_chain needs at least two graphs: " + step.raw);
      if (step.inputs.size() != 1)
        throw std::runtime_error("run_elf_chain needs exactly one input: " + step.raw);
      std::vector<SharedStage> stages;
      for (const auto& name : step.graphs) {
        auto found = graph_infos.find(name);
        if (found == graph_infos.end())
          throw std::runtime_error("shared chain graph '" + name + "' has no ELF");
        if (found->second.external_layout != "HWC")
          throw std::runtime_error("shared chain graph '" + name +
                                   "' was not compiled with external HWC layout");
        stages.push_back({name, found->second.elf, found->second.inputs, found->second.outputs});
      }
      const auto& tail = graph_infos.at(step.graphs.back());
      step.chain = "chain_" + std::to_string(step_index);
      chains_[step.chain] = std::make_unique<SharedRunnerChain>(
          step.chain, stages, tail.output_layout, tail.logical_width, tail.logical_channels);
      if (buffer_sizes_.at(step.inputs.front()) != chains_.at(step.chain)->input_elements() ||
          buffer_sizes_.at(step.out) != chains_.at(step.chain)->output_elements())
        throw std::runtime_error("run_elf_chain plan buffer sizes disagree with bundle metadata");
    }
    for (const auto& name : step.inputs)
      if (!buffer_sizes_.count(name))
        throw std::runtime_error("step reads undeclared buffer '" + name + "'");
    if (!step.source.empty() && !buffer_sizes_.count(step.source))
      throw std::runtime_error("step reads undeclared buffer '" + step.source + "'");
    for (const auto& part : step.parts)
      if (!part.sidecar && !buffer_sizes_.count(part.source))
        throw std::runtime_error("pack part reads undeclared buffer '" + part.source + "'");

    if (step.opcode == Opcode::Pack && !step.parts.empty()) {
      step.constant_pack = true;
      for (const auto& part : step.parts) step.constant_pack = step.constant_pack && part.sidecar;
    }
    steps_.push_back(std::move(step));
    ++step_index;
  }

  if (device_arena_) {
    DeviceProgramChecker checker(*device_arena_);
    for (const auto& step : steps_)
      if (step.opcode == Opcode::RunDeviceChain) checker.collect_writes(*device_chains_.at(step.chain));
    checker.check_cached_regions();
    for (const auto& step : steps_)
      if (step.opcode == Opcode::RunDeviceChain) checker.simulate(*device_chains_.at(step.chain));
  }

  // A sidecar-only pack is constant only if its buffer has no other writer:
  // no other step, and not a request tensor (written from outside the plan).
  std::map<std::string, size_t> writers;
  for (const auto& step : steps_)
    for (const auto& name : written_buffers(step)) ++writers[name];
  for (const auto& [name, elements] : wire_.request_tensors) writers[name] += 2;
  for (auto& step : steps_)
    if (step.constant_pack && writers[step.out] != 1) step.constant_pack = false;

  // Group run_elf reads of one buffer between writes; a group of two or more
  // reads the same values, which is when sharing a quantized copy pays.
  {
    std::map<std::string, std::vector<std::pair<size_t, size_t>>> open;
    const auto close = [&](const std::string& name) {
      auto found = open.find(name);
      if (found == open.end()) return;
      if (found->second.size() > 1)
        for (const auto& [step, input] : found->second) steps_[step].shared_inputs[input] = true;
      open.erase(found);
    };
    for (size_t index = 0; index < steps_.size(); ++index) {
      auto& step = steps_[index];
      if (step.opcode == Opcode::RunElf) {
        step.shared_inputs.assign(step.inputs.size(), false);
        for (size_t input = 0; input < step.inputs.size(); ++input)
          open[step.inputs[input]].emplace_back(index, input);
      }
      for (const auto& name : written_buffers(step)) close(name);
    }
    while (!open.empty()) close(open.begin()->first);
  }

  // Twin lanes (see plan.hpp). Longest match first; lane 1 gets its own
  // instance of each graph it runs. A second instance of an ELF shares the
  // loaded model (llima keys models by path), so it costs only its buffers.
  for (size_t begin = 0; begin < steps_.size();) {
    size_t found = 0;
    for (size_t length = std::min<size_t>(64, (steps_.size() - begin) / 2); length >= 2; --length)
      if (lanes_match(begin, length)) {
        found = length;
        break;
      }
    if (!found) {
      ++begin;
      continue;
    }
    twin_blocks_.push_back({begin, found});
    for (size_t k = begin + found; k < begin + 2 * found; ++k)
      if (steps_[k].opcode == Opcode::RunElf && !twin_runners_.count(steps_[k].graph))
        twin_runners_[steps_[k].graph] = runner_factories.at(steps_[k].graph)();
    if (verbose_)
      std::cout << "overlapping steps " << begin << "-" << begin + found - 1 << " with "
                << begin + found << "-" << begin + 2 * found - 1 << std::endl;
    begin += 2 * found;
  }
}

bool Plan::lanes_match(size_t begin, size_t length) const {
  if (begin + 2 * length > steps_.size()) return false;
  std::set<std::string> reads[2], writes[2];
  size_t mla_steps = 0;
  for (size_t k = 0; k < length; ++k) {
    const Step* lane[2] = {&steps_[begin + k], &steps_[begin + length + k]};
    if (lane[0]->opcode != lane[1]->opcode) return false;
    switch (lane[0]->opcode) {
      case Opcode::RunElf:
        if (lane[0]->graph != lane[1]->graph) return false;
        ++mla_steps;
        break;
      case Opcode::RunElfChain:
        if (lane[0]->graphs != lane[1]->graphs) return false;
        ++mla_steps;
        break;
      case Opcode::PixelUnshuffle:
        break;
      default:
        return false;  // only steps whose effects are fully known here
    }
    for (size_t l = 0; l < 2; ++l) {
      for (const auto& name : lane[l]->inputs) reads[l].insert(name);
      if (!lane[l]->source.empty()) reads[l].insert(lane[l]->source);
      writes[l].insert(lane[l]->out);
    }
  }
  // One graph call per lane is enough when the lane also has host work
  // (length >= 2): chained N1.6 bundles keep only the vision chain and the
  // pixel unshuffle per camera on the host side of run_device_chain, and
  // camera 0's decode + unshuffle then hides under camera 1's vision chain.
  if (mla_steps < 1) return false;
  for (const auto& name : writes[0])
    if (reads[1].count(name) || writes[1].count(name)) return false;
  for (const auto& name : writes[1])
    if (reads[0].count(name)) return false;
  return true;
}

std::vector<float>& Plan::buffer(const std::string& name) {
  auto found = buffers_.find(name);
  if (found == buffers_.end()) throw std::runtime_error("no buffer '" + name + "'");
  return found->second;
}

const std::vector<float>& Plan::buffer(const std::string& name) const {
  auto found = buffers_.find(name);
  if (found == buffers_.end()) throw std::runtime_error("no buffer '" + name + "'");
  return found->second;
}

const std::vector<float>& Plan::read_buffer(const Step& step) const {
  if (!step.source.empty()) return buffer(step.source);
  if (!step.inputs.empty()) return buffer(step.inputs.front());
  throw std::runtime_error("step needs a source buffer but names none: " + step.raw);
}

void Plan::run_step(const Step& step) {
  auto& out = buffer(step.out);
  if (step.out == last_int8_buffer_) {
    last_int8_runner_ = nullptr;
    last_int8_buffer_.clear();
  }

  switch (step.opcode) {
    case Opcode::RunElf: {
      auto& runner = *runners_.at(step.graph);
      // A single input buffer is passed straight through; several are
      // concatenated (the plan validator guarantees the sizes line up).
      const int8_t* raw_first = nullptr;
      if (last_int8_runner_ && !std::getenv("POLIMA_DISABLE_INT8_HANDOFF") &&
          !step.inputs.empty() && step.inputs.front() == last_int8_buffer_ &&
          runner.can_reuse_first_int8_from(*last_int8_runner_))
        raw_first = last_int8_runner_->raw_int8_output();
      const bool profile = std::getenv("POLIMA_PROFILE_PHASES") != nullptr;
      RunTiming timing;
      double prepare_ms = 0.0;
      // Every input buffer is passed as its own part: BF16 graphs convert
      // each straight into the MLA buffer, INT8 graphs quantize each against
      // its own segment, and neither joins them into a temporary first.
      const auto prepare_start = Clock::now();
      prepare_parts(step, runner.input_elements());
      if (profile) prepare_ms = elapsed_ms(prepare_start);
      runner.run(parts_, out.data(), profile ? &timing : nullptr, raw_first, &quantized_inputs_);
      graph_checked_ = runner.output_checked();
      graph_first_nonfinite_ = runner.first_nonfinite();
      if (profile) report_phase(step.graph, step.out, timing, prepare_ms);
      last_int8_runner_ = runner.raw_int8_output() ? &runner : nullptr;
      last_int8_buffer_ = last_int8_runner_ ? step.out : std::string{};
      break;
    }

    case Opcode::RunDiTChain: {
      last_int8_runner_=nullptr;last_int8_buffer_.clear();
      std::array<const float*,5> inputs{};std::array<size_t,5> versions{};
      for(size_t i=0;i<5;++i){inputs[i]=buffer(step.inputs[i]).data();versions[i]=buffer_versions_[step.inputs[i]];}
      const bool profile=std::getenv("POLIMA_PROFILE_PHASES")!=nullptr;
      RunTiming timing;
      auto& chain=*dit_chains_.at(step.chain);
      chain.run(inputs,versions,out.data(),profile ? &timing : nullptr);
      graph_checked_=chain.output_elements()==out.size();
      graph_first_nonfinite_=chain.first_nonfinite();
      if(profile)report_phase("dit_chain",step.out,timing,0.0);
      break;
    }

    case Opcode::RunDeviceChain: {
      last_int8_runner_ = nullptr;
      last_int8_buffer_.clear();
      const bool profile = std::getenv("POLIMA_PROFILE_PHASES") != nullptr;
      RunTiming timing;
      device_chains_.at(step.chain)->run(sidecars_, request_flags_, profile ? &timing : nullptr);
      if (profile) report_phase(step.chain, step.out, timing, 0.0);
      break;
    }

    case Opcode::RunBoundChain: {
      last_int8_runner_ = nullptr; last_int8_buffer_.clear();
      std::vector<const float*> inputs; std::vector<size_t> versions;
      for (const auto& name : step.inputs) {
        inputs.push_back(buffer(name).data()); versions.push_back(buffer_versions_[name]);
      }
      const bool profile = std::getenv("POLIMA_PROFILE_PHASES") != nullptr;
      RunTiming timing;
      auto& chain = *bound_chains_.at(step.chain);
      chain.run(inputs, versions, out.data(), profile ? &timing : nullptr);
      graph_checked_ = chain.output_elements() == out.size();
      graph_first_nonfinite_ = chain.first_nonfinite();
      if (profile) report_phase("bound_chain", step.out, timing, 0.0);
      break;
    }

    case Opcode::RunElfChain: {
      last_int8_runner_ = nullptr;
      last_int8_buffer_.clear();
      const bool profile = std::getenv("POLIMA_PROFILE_PHASES") != nullptr;
      RunTiming timing;
      auto& chain = *chains_.at(step.chain);
      chain.run(buffer(step.inputs.front()).data(), out.data(), profile ? &timing : nullptr);
      graph_checked_ = chain.output_checked();
      graph_first_nonfinite_ = chain.first_nonfinite();
      if (profile) report_phase(step.chain, step.out, timing, 0.0);
      break;
    }

    case Opcode::Pack: {
      // Zero first: the stem graph reads 601x512 but only ~307k of it is
      // written, and the latent token must be zero (EncoderStemLayer builds it
      // from torch.zeros).
      // Only the ranges no part writes need the zeros; filling them alone
      // gives the same buffer without writing every value twice. A pack that
      // reads its own output fills everything first.
      bool reads_itself = false;
      for (const auto& part : step.parts)
        reads_itself = reads_itself || (!part.sidecar && part.source == step.out);
      if (reads_itself) {
        std::fill(out.begin(), out.end(), 0.0f);
      } else {
        std::vector<std::pair<size_t, size_t>> covered;
        covered.reserve(step.parts.size());
        for (const auto& part : step.parts)
          covered.emplace_back(std::min(part.destination_offset, out.size()),
                               std::min(part.destination_offset + part.count, out.size()));
        std::sort(covered.begin(), covered.end());
        size_t cursor = 0;
        for (const auto& [begin, end] : covered) {
          if (begin > cursor) std::fill(out.begin() + cursor, out.begin() + begin, 0.0f);
          cursor = std::max(cursor, end);
        }
        if (cursor < out.size()) std::fill(out.begin() + cursor, out.end(), 0.0f);
      }
      for (const auto& part : step.parts) {
        // A part comes either from a live buffer or from a constants/ sidecar.
        // Sections fixed for a checkpoint (such as a language embedding) ship
        // with the bundle instead of crossing the wire every inference.
        const auto& source = part.sidecar ? sidecars_.get(part.source) : buffer(part.source);
        if (part.destination_offset + part.count > out.size())
          throw std::runtime_error("pack part overruns buffer '" + step.out + "'");
        size_t source_offset = part.source_offset;
        if (part.source_stride) {
          if (request_flags_ >= part.selector_count)
            throw std::runtime_error(
                "request flags " + std::to_string(request_flags_) +
                " outside task catalog [0, " + std::to_string(part.selector_count) + ")");
          source_offset += static_cast<size_t>(request_flags_) * part.source_stride;
        }
        if (source_offset + part.count > source.size())
          throw std::runtime_error("pack part reads past source '" + part.source + "'");
        if (reads_itself)
          std::copy_n(source.begin() + source_offset, part.count,
                      out.begin() + part.destination_offset);
        else
          kernels::copy_bytes(out.data() + part.destination_offset,
                              source.data() + source_offset, part.count * sizeof(float));
      }
      break;
    }

    case Opcode::Slice: {
      const auto& source = read_buffer(step);
      const size_t count = step.count ? step.count : out.size();
      if (step.offset + count > source.size())
        throw std::runtime_error("slice overruns its source on step " + step.raw);
      std::copy_n(source.begin() + step.offset, count, out.begin());
      break;
    }

    case Opcode::GatherStrided: {
      // ACT's 16-wide action head is padded for MLA alignment; the host takes
      // the policy's real action width from every 16 values.
      //
      // The scatter direction is supported too: the destination stride is
      // configurable and defaults to `take` (which is ACT's contiguous case).
      // A source stride of 0 broadcasts one span into every block, which is how
      // a shared time embedding reaches every action token.
      const auto& source = read_buffer(step);
      const size_t destination_stride = step.destination_stride ? step.destination_stride
                                                                : step.take;
      if (step.clear) std::fill(out.begin(), out.end(), 0.0f);
      for (size_t index = 0; index < step.count; ++index) {
        const size_t read_at = index * step.stride;
        const size_t write_at = index * destination_stride + step.destination_offset;
        if (read_at + step.take > source.size() || write_at + step.take > out.size())
          throw std::runtime_error("gather_strided overruns on step " + step.raw);
        std::copy_n(source.begin() + read_at, step.take, out.begin() + write_at);
      }
      break;
    }

    case Opcode::PixelUnshuffle: {
      // Eagle's channel fold, between the vision chain and the connector: a
      // grid x grid map of C channels becomes (grid/f)^2 tokens of C*f^2,
      // matching torch.pixel_unshuffle's (c*f^2 + dy*f + dx) channel order.
      // The connector is the one Eagle graph the chain cannot swallow, because
      // this reshape has to happen between post_norm and it.
      const auto& source = read_buffer(step);
      const size_t height = step.height, width = step.width;
      const size_t channels = step.channels, factor = step.factor;
      if (!height || !width || !channels || !factor ||
          height % factor || width % factor)
        throw std::runtime_error("pixel_unshuffle needs dimensions divisible by factor on step " +
                                 step.raw);
      const size_t out_height = height / factor;
      const size_t out_width = width / factor;
      const size_t out_channels = channels * factor * factor;
      if (source.size() != height * width * channels ||
          out.size() != out_height * out_width * out_channels)
        throw std::runtime_error("pixel_unshuffle shape mismatch on step " + step.raw);
      for (size_t row = 0; row < out_height; ++row)
        for (size_t column = 0; column < out_width; ++column)
          for (size_t dy = 0; dy < factor; ++dy)
            for (size_t dx = 0; dx < factor; ++dx) {
              const size_t read_at =
                  ((row * factor + dy) * width + (column * factor + dx)) * channels;
              const size_t write_at =
                  (row * out_width + column) * out_channels + dy * factor + dx;
              for (size_t channel = 0; channel < channels; ++channel)
                out[write_at + channel * factor * factor] = source[read_at + channel];
            }
      break;
    }

    case Opcode::Scale: {
      const auto& source = read_buffer(step);
      for (size_t index = 0; index < out.size(); ++index) out[index] = source[index] * step.scalar;
      break;
    }

    case Opcode::SincosState: {
      const auto& values = buffers_.at(step.source);
      size_t total = 0;
      for (size_t width : step.group_widths) {
        if (!width || width > values.size() - total)
          throw std::runtime_error("sincos_state invalid group widths");
        total += width;
      }
      if (total != values.size() || total > out.size() / 2)
        throw std::runtime_error("sincos_state shape mismatch");
      std::fill(out.begin(), out.end(), 0.0f);
      size_t input_offset = 0, output_offset = 0;
      for (size_t width : step.group_widths) {
        for (size_t i = 0; i < width; ++i) {
          out[output_offset + i] = std::sin(values[input_offset + i]);
          out[output_offset + width + i] = std::cos(values[input_offset + i]);
        }
        input_offset += width;
        output_offset += 2 * width;
      }
      break;
    }
    case Opcode::ClipUnit: {
      const auto& source = read_buffer(step);
      if (source.size() != out.size())
        throw std::runtime_error("clip_unit shape mismatch on step " + step.raw);
      for (size_t index = 0; index < out.size(); ++index)
        out[index] = std::clamp(source[index], -1.0f, 1.0f);
      break;
    }

    case Opcode::MatVec: {
      const auto& x = read_buffer(step);
      const auto& w = sidecars_.get(step.weights);
      const size_t rows = step.rows ? step.rows : out.size();
      const size_t cols = step.cols ? step.cols : x.size();
      for (size_t row = 0; row < rows; ++row) {
        float total = step.bias.empty() ? 0.0f : sidecars_.get(step.bias)[row];
        const float* weights = w.data() + row * cols;
        for (size_t col = 0; col < cols; ++col) total += weights[col] * x[col];
        out[row] = total;
      }
      break;
    }

    case Opcode::SincosTime: {
      // Geometric period sweep from min_period to max_period, as an angular
      // frequency: period = min * (max/min)^(i/(half-1)), angle = t * 2pi/period.
      //
      // The details matter and are easy to get subtly wrong: the exponent divides by
      // half-1 (so the last entry lands exactly on max_period, not one step
      // short), and the angle carries the 2pi. Getting either wrong yields a
      // plausible-looking embedding and a policy that acts at the wrong phase
      // of the denoising schedule.
      const float t = step.scalar;
      const size_t half = out.size() / 2;
      if (half == 0) break;
      const float min_period = step.min_period > 0.0f ? step.min_period : 1.0f;
      const float max_period = step.max_period > 0.0f ? step.max_period : min_period;
      const float span = max_period / min_period;
      const float divisor = half > 1 ? static_cast<float>(half - 1) : 1.0f;
      for (size_t index = 0; index < half; ++index) {
        const float fraction = static_cast<float>(index) / divisor;
        const float period = min_period * std::pow(span, fraction);
        const float angle = t * (2.0f * static_cast<float>(M_PI) / period);
        out[index] = std::sin(angle);
        out[half + index] = std::cos(angle);
      }
      break;
    }

    case Opcode::Euler: {
      const auto& velocity = read_buffer(step);
      const float dt = step.scalar;
      for (size_t index = 0; index < out.size(); ++index) out[index] -= dt * velocity[index];
      break;
    }

    case Opcode::Normalize:
    case Opcode::Denormalize: {
      const auto& source = read_buffer(step);
      const auto& mean = sidecars_.get(step.mean);
      const auto& deviation = sidecars_.get(step.std_dev);
      for (size_t index = 0; index < out.size(); ++index) {
        const size_t stat = index % mean.size();
        if (step.opcode == Opcode::Normalize) {
          const float value = (source[index] - mean[stat]) / deviation[stat];
          out[index] = step.clip ? std::clamp(value, step.clip_min, step.clip_max) : value;
        } else {
          // GR00T clips the normalized value before scaling it back.
          const float value = step.clip ? std::clamp(source[index], step.clip_min, step.clip_max)
                                        : source[index];
          out[index] = value * deviation[stat] + mean[stat];
        }
      }
      break;
    }
  }
}

float* Plan::input_buffer(const std::string& name) { return buffer(name).data(); }

void Plan::prepare_parts(const Step& step, size_t expected_elements) {
  // Every input buffer is passed as its own part: BF16 graphs convert each
  // straight into the MLA buffer, INT8 graphs quantize each against its own
  // segment, and neither joins them into a temporary first.
  parts_.clear();
  size_t offset = 0;
  for (const auto& name : step.inputs) {
    const auto& source = buffer(name);
    const size_t index = parts_.size();
    parts_.push_back({source.data(), source.size(), token(name),
                      index < step.shared_inputs.size() && step.shared_inputs[index]});
    offset += source.size();
  }
  if (offset != expected_elements)
    throw std::runtime_error("joined graph input size mismatch: " + step.graph);
}

std::vector<std::string> Plan::written_buffers(const Step& step) const {
  std::vector<std::string> names{step.out};
  // A device program also writes every host buffer it downloads into.
  if (step.opcode == Opcode::RunDeviceChain)
    for (const auto& download : device_chains_.at(step.chain)->downloads())
      names.push_back(download.host);
  return names;
}

void Plan::finish_step(const Step& step) {
  // run_dit_chain and run_bound_chain skip re-uploading inputs whose version is unchanged.
  const bool tracks_versions = !dit_chains_.empty() || !bound_chains_.empty();
  if (step.opcode == Opcode::RunDeviceChain) {
    for (const auto& download : device_chains_.at(step.chain)->downloads()) {
      if (tracks_versions) ++buffer_versions_[download.host];
      buffer_tokens_.erase(download.host);
    }
  }
  if (tracks_versions) ++buffer_versions_[step.out];
  buffer_tokens_.erase(step.out);
  // Accelerator corruption or an incorrect DRAM layout must become a failed
  // request, never a motor command. ELF outputs are the useful diagnostic
  // boundaries; the final result also covers host-side arithmetic.
  if (step.opcode == Opcode::RunElf)
    require_finite_checked("graph " + step.graph, buffer(step.out), graph_checked_,
                           graph_first_nonfinite_);
  if (step.opcode == Opcode::RunElfChain || step.opcode == Opcode::RunDiTChain ||
      step.opcode == Opcode::RunBoundChain)
    require_finite_checked("graph chain " + step.chain, buffer(step.out), graph_checked_,
                           graph_first_nonfinite_);
}

void Plan::run_twin_block(const TwinBlock& block) {
  if (!mla_worker_) mla_worker_ = std::make_unique<MlaWorker>();
  struct Lane {
    size_t next = 0, end = 0;
    const Step* pending = nullptr;  // submitted to the MLA, not yet decoded
    uint64_t ticket = 0;
    Runner* runner = nullptr;
    SharedRunnerChain* chain = nullptr;
    Runner* last_int8 = nullptr;    // this lane's raw INT8 hand-off source
    std::string last_int8_buffer;
  };
  Lane lanes[2];
  lanes[0].next = block.begin;
  lanes[0].end = lanes[1].next = block.begin + block.length;
  lanes[1].end = block.begin + 2 * block.length;
  // Sequential execution would report the earliest failing step; so does this.
  std::exception_ptr error;
  size_t error_step = steps_.size();
  const auto fail = [&](const Step& step) {
    const size_t index = static_cast<size_t>(&step - steps_.data());
    if (index < error_step) {
      error_step = index;
      error = std::current_exception();
    }
  };
  const bool int8_handoff = !std::getenv("POLIMA_DISABLE_INT8_HANDOFF");

  const auto complete = [&](Lane& lane) {
    const Step& step = *lane.pending;
    lane.pending = nullptr;
    try {
      mla_worker_->wait(lane.ticket);
      auto& out = buffer(step.out);
      if (lane.runner) {
        lane.runner->download();
        lane.runner->decode(out.data());
        graph_checked_ = lane.runner->output_checked();
        graph_first_nonfinite_ = lane.runner->first_nonfinite();
        lane.last_int8 = lane.runner->raw_int8_output() ? lane.runner : nullptr;
      } else {
        lane.chain->download();
        lane.chain->decode(out.data());
        graph_checked_ = lane.chain->output_checked();
        graph_first_nonfinite_ = lane.chain->first_nonfinite();
        lane.last_int8 = nullptr;
      }
      lane.last_int8_buffer = lane.last_int8 ? step.out : std::string{};
      finish_step(step);
    } catch (...) {
      fail(step);
    }
  };

  // Decodes the lane's pending graph, runs its host steps, and submits its
  // next graph to the MLA worker.
  const auto advance = [&](Lane& lane, bool twin) {
    if (lane.pending) complete(lane);
    while (!error && lane.next < lane.end) {
      const Step& step = steps_[lane.next++];
      try {
        if (step.opcode == Opcode::RunElf) {
          Runner& runner = twin ? *twin_runners_.at(step.graph) : *runners_.at(step.graph);
          const int8_t* raw_first = nullptr;
          if (lane.last_int8 && int8_handoff && !step.inputs.empty() &&
              step.inputs.front() == lane.last_int8_buffer &&
              runner.can_reuse_first_int8_from(*lane.last_int8))
            raw_first = lane.last_int8->raw_int8_output();
          prepare_parts(step, runner.input_elements());
          runner.encode(parts_, raw_first, &quantized_inputs_);
          runner.upload();
          lane.runner = &runner;
          lane.chain = nullptr;
          lane.pending = &step;
          lane.ticket = mla_worker_->submit([&runner] { runner.execute(); });
          return;
        }
        if (step.opcode == Opcode::RunElfChain) {
          auto& chain = *chains_.at(step.chain);
          chain.encode(buffer(step.inputs.front()).data());
          chain.upload();
          lane.runner = nullptr;
          lane.chain = &chain;
          lane.pending = &step;
          lane.ticket = mla_worker_->submit([&chain] { chain.execute(); });
          return;
        }
        graph_checked_ = false;
        run_step(step);
        finish_step(step);
        if (step.out == lane.last_int8_buffer) {
          lane.last_int8 = nullptr;
          lane.last_int8_buffer.clear();
        }
      } catch (...) {
        fail(step);
      }
    }
  };

  for (;;) {
    advance(lanes[0], false);
    advance(lanes[1], true);
    if (lanes[0].pending || lanes[1].pending) continue;
    if (error || (lanes[0].next >= lanes[0].end && lanes[1].next >= lanes[1].end)) break;
  }
  // The step after the block sees what it would see sequentially: lane 1 ran last.
  last_int8_runner_ = error ? nullptr : lanes[1].last_int8;
  last_int8_buffer_ = last_int8_runner_ ? lanes[1].last_int8_buffer : std::string{};
  if (error) std::rethrow_exception(error);
}

uint64_t Plan::token(const std::string& name) {
  auto found = buffer_tokens_.find(name);
  if (found != buffer_tokens_.end()) return found->second;
  const uint64_t fresh = ++next_token_;
  buffer_tokens_.emplace(name, fresh);
  return fresh;
}

const std::vector<float>& Plan::execute() { return execute(0); }

const std::vector<float>& Plan::execute(uint32_t request_flags) {
  request_flags_ = request_flags;
  last_int8_runner_ = nullptr;
  last_int8_buffer_.clear();
  timings_.clear();
  buffer_versions_.clear();
  // Request inputs were written outside the plan: every buffer gets a fresh
  // reuse token on first read. Tokens are never reused across requests.
  buffer_tokens_.clear();
  quantized_inputs_.clear();
  for(auto& item:dit_chains_)item.second->begin_request();
  for (auto& item : bound_chains_) item.second->begin_request();
  if (!dump_stages_.empty()) std::filesystem::create_directories(dump_stages_);
  // Diagnostics need one step at a time; so does an explicit opt-out.
  // POLIMA_OVERLAP_DUMP=1 keeps overlap under --dump-stages and dumps a
  // block's buffers after the block: exact for steps that write their own
  // buffer, the block's final value for a buffer several steps rewrite.
  const bool overlap = (dump_stages_.empty() || std::getenv("POLIMA_OVERLAP_DUMP")) &&
                       !collect_timings_ && !std::getenv("POLIMA_PROFILE_PHASES") &&
                       !std::getenv("POLIMA_DISABLE_OVERLAP");
  const auto dump = [&](size_t step_index) {
    if (dump_stages_.empty()) return;
    const auto& step = steps_[step_index];
    std::string number = std::to_string(step_index);
    number.insert(0, 3 - std::min<size_t>(3, number.size()), '0');
    write_f32(dump_stages_ / (number + "_" + opcode_name(step.opcode) + "_" + step.out + ".f32"),
              buffer(step.out));
  };
  size_t next_block = 0;
  for (size_t step_index = 0; step_index < steps_.size(); ++step_index) {
    if (next_block < twin_blocks_.size() && twin_blocks_[next_block].begin == step_index) {
      const auto& block = twin_blocks_[next_block++];
      if (overlap) {
        run_twin_block(block);
        for (size_t k = 0; k < 2 * block.length; ++k) dump(step_index + k);
        step_index += 2 * block.length - 1;
        continue;
      }
    }
    const auto& step = steps_[step_index];
    const auto started = Clock::now();
    graph_checked_ = false;
    graph_first_nonfinite_ = 0;
    uint64_t constant_token = 0;
    if (step.constant_pack && RuntimeOptions::get().input_reuse) {
      // Only this step writes the buffer, and only from sidecars, so once it
      // has run for these request flags the buffer already holds the result.
      auto done = constant_packs_.find(step.out);
      if (done == constant_packs_.end() || done->second.first != request_flags_) {
        constant_packs_.erase(step.out);
        run_step(step);
        done = constant_packs_.emplace(step.out, std::make_pair(request_flags_, ++next_token_)).first;
      }
      constant_token = done->second.second;
    } else {
      run_step(step);
    }
    finish_step(step);
    if (constant_token) buffer_tokens_[step.out] = constant_token;
    dump(step_index);
    if (collect_timings_)
      timings_.push_back(std::string(opcode_name(step.opcode)) + ":" +
                         (step.graph.empty() ? step.out : step.graph) + "=" +
                         std::to_string(elapsed_ms(started)));
  }
  require_finite("result " + result_, buffer(result_));
  return buffer(result_);
}

const std::vector<float>& Plan::execute(const std::map<std::string, const float*>& inputs) {
  for (const auto& [name, elements] : wire_.request_tensors) {
    auto supplied = inputs.find(name);
    if (supplied == inputs.end())
      throw std::runtime_error("missing input tensor '" + name + "'");
    auto& target = buffer(name);
    if (target.size() != elements)
      throw std::runtime_error("buffer '" + name + "' size disagrees with the wire description");
    std::memcpy(target.data(), supplied->second, elements * sizeof(float));
  }

  return execute();
}

}  // namespace polima
