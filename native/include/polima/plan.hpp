// The execution-plan interpreter.
//
// This is what makes one binary serve every bundle. act_llima.cpp hard-codes
// ACT's pipeline in C++ -- six Runners named in a constructor initializer list,
// packing offsets as literals, a 16->6 unpad loop. Change the policy and you
// rebuild the binary on the board.
//
// Here that pipeline is plan.json:
//
//   run_elf        cam0   <- vision_backbone(image0)
//   run_elf        cam1   <- vision_backbone(image1)
//   pack           packed <- state@0, cam0@512, cam1@154112
//   run_elf        hidden <- encoder_layer_00_stem(packed)
//   run_elf        hidden <- encoder_layer_01(hidden)   [x3]
//   run_elf        padded <- decoder_action_tail(hidden)
//   gather_strided actions<- padded stride 16 take 6 count 100
//
// ACT needs exactly three opcodes: run_elf, pack, gather_strided. SmolVLA uses
// all ten, including a 10-iteration flow-matching loop that the plan emits
// unrolled -- so the interpreter still has no control flow of its own.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "polima/runner.hpp"
#include "polima/dit_chain.hpp"
#include "polima/device_chain.hpp"
#include "polima/bound_chain.hpp"
#include "polima/sidecar.hpp"

namespace polima {

enum class Opcode {
  RunElf,
  RunElfChain,
  RunDiTChain,
  RunDeviceChain,
  RunBoundChain,   // named persistent bindings (GR00T DiT)
  Pack,
  Slice,
  GatherStrided,
  PixelUnshuffle,
  Scale,
  ClipUnit,
  MatVec,
  SincosTime,
  SincosState,
  Euler,
  Normalize,
  Denormalize,
};

Opcode parse_opcode(const std::string& name);
const char* opcode_name(Opcode opcode);

struct PackPart {
  std::string source;
  size_t destination_offset = 0;
  size_t count = 0;
  // Optional request-flags-indexed source window. GR00T stores one flattened
  // prompt per trained task in a single sidecar.
  size_t source_offset = 0;
  size_t source_stride = 0;
  size_t selector_count = 0;
  // Read from a constants/ sidecar rather than a plan buffer. SmolVLA's prefix
  // is built from two of these -- the empty-image and language embeddings are
  // fixed for a checkpoint, so they ship with the bundle instead of being
  // recomputed or sent over the wire every inference.
  bool sidecar = false;
};

struct Step {
  Opcode opcode;
  std::string out;
  std::string graph;                 // run_elf
  std::vector<std::string> graphs;   // run_elf_chain
  std::string chain;                 // internal SharedRunnerChain key
  std::vector<std::string> prelude;  // run_bound_chain: run once per input change
  std::vector<std::string> binds;    // run_bound_chain: logical name of each in[]
  std::string carry;                 // run_bound_chain: carried (ping-pong) binding
  std::string source;                // args["src"] -- slice, gather_strided, scale, ...
  std::vector<std::string> inputs;   // args["in"]  -- run_elf and friends
  std::vector<PackPart> parts;       // pack
  std::vector<size_t> group_widths;  // sincos_state in checkpoint order
  size_t offset = 0;                 // slice
  size_t stride = 0;                 // gather_strided: source stride; 0 broadcasts
  size_t take = 0;
  size_t destination_stride = 0;     // gather_strided: defaults to `take`
  size_t destination_offset = 0;
  bool clear = false;                // zero the output before scattering
  // pack whose parts are all sidecars, writing a buffer nothing else writes:
  // after the first request its output already holds exactly these values.
  bool constant_pack = false;
  // run_elf: per input, whether other run_elf steps read the same contents
  // (no write in between), so one quantized copy can serve them all.
  std::vector<bool> shared_inputs;
  size_t count = 0;
  size_t size = 0;
  float scalar = 0.0f;               // scale, euler dt, sincos timestep
  float min_period = 0.0f;           // sincos_time
  float max_period = 0.0f;
  std::string weights;               // matvec / normalize
  std::string bias;
  std::string mean;
  std::string std_dev;
  bool clip = false;                 // normalize/denormalize: clamp into [clip_min, clip_max]
  float clip_min = 0.0f;             //   normalize clamps its result, denormalize its input,
  float clip_max = 0.0f;             //   which is where GR00T's min/max scaling clips
  size_t rows = 0;
  size_t cols = 0;
  size_t height = 0;                 // pixel_unshuffle: input token-grid height
  size_t width = 0;                  // pixel_unshuffle: input token-grid width
  size_t channels = 0;               // pixel_unshuffle: input channels
  size_t factor = 0;                 // pixel_unshuffle: fold factor
  std::string raw;                   // original JSON, for error messages
};

struct WireDescription {
  uint32_t magic = 0;
  uint32_t version = 1;
  int default_port = 0;
  std::vector<std::pair<std::string, size_t>> request_tensors;  // name -> elements
  size_t response_elements = 0;
};

// A bundle's plan.json plus the ELF inventory from bundle.json, loaded and
// wired into live Runners with every buffer allocated up front.
class Plan {
 public:
  Plan(const std::filesystem::path& bundle_root, bool verbose = false);

  // Runs the whole plan. `inputs` must supply every wire request tensor by name.
  const std::vector<float>& execute(const std::map<std::string, const float*>& inputs);

  // Run against whatever is already in the plan's own input buffers.
  // Callers that can write straight into `input_buffer()` should use this: it
  // avoids copying the request payload a second time, which for ACT is 7.4 MB
  // (two 480x640x3 images) on every single inference.
  const std::vector<float>& execute();
  const std::vector<float>& execute(uint32_t request_flags);

  // Writable pointer to a declared buffer, for filling inputs in place.
  float* input_buffer(const std::string& name);

  const WireDescription& wire() const { return wire_; }
  const std::string& bundle_id() const { return bundle_id_; }
  const std::string& policy() const { return policy_; }
  //: The bundle's own pass/fail bar for `check`, written by the packer from
  //: the policy spec. Absent in bundles packed before smoke thresholds were
  //: per-policy, which keep ACT's historical 0.999 / 0.01.
  double smoke_cosine_min() const { return smoke_cosine_min_; }
  double smoke_mean_abs_max() const { return smoke_mean_abs_max_; }
  size_t result_elements() const { return buffer_sizes_.at(result_); }
  const std::vector<std::string>& stage_timings() const { return timings_; }
  // Timing builds a string per step, which on SmolVLA's 82-step plan is real
  // work on the request path -- so it is off unless asked for, and `stages`
  // turns it on for exactly one run rather than requiring a restart.
  void set_collect_timings(bool on) { collect_timings_ = on; }
  void set_dump_stages(const std::filesystem::path& directory) { dump_stages_ = directory; }

 private:
  std::vector<float>& buffer(const std::string& name);
  const std::vector<float>& buffer(const std::string& name) const;
  // Resolves a step's single read buffer: args["src"] if present, else in[0].
  const std::vector<float>& read_buffer(const Step& step) const;
  void run_step(const Step& step);

  // Declared first, so it is destroyed LAST. Members are torn down in reverse
  // declaration order, and this used to sit at the bottom -- which disconnected
  // the MLA runtime before a single Runner had called model_.free().
  std::shared_ptr<RuntimeGuard> guard_;

  std::filesystem::path root_;
  bool verbose_;
  bool collect_timings_ = false;
  std::filesystem::path dump_stages_;
  std::string bundle_id_;
  std::string policy_;
  double smoke_cosine_min_ = 0.999;
  double smoke_mean_abs_max_ = 0.01;
  std::string result_;
  WireDescription wire_;

  std::map<std::string, std::vector<float>> buffers_;
  std::map<std::string, size_t> buffer_sizes_;
  std::map<std::string, std::unique_ptr<Runner>> runners_;
  std::map<std::string, std::unique_ptr<SharedRunnerChain>> chains_;
  std::map<std::string, std::unique_ptr<DiTRunnerChain>> dit_chains_;
  // run_device_chain: plan-wide device buffers, then the programs bound to
  // them (declared after the arena so programs are freed first).
  std::unique_ptr<DeviceArena> device_arena_;
  std::map<std::string, std::unique_ptr<DeviceChain>> device_chains_;
  std::map<std::string, std::unique_ptr<BoundRunnerChain>> bound_chains_;
  std::map<std::string, size_t> buffer_versions_;
  std::vector<Step> steps_;
  // Reuse tokens: a buffer's token changes whenever a step writes it and at
  // every request; graphs skip re-converting an input whose token they have
  // already uploaded.
  uint64_t token(const std::string& name);
  std::map<std::string, uint64_t> buffer_tokens_;
  uint64_t next_token_ = 0;
  std::vector<InputPart> parts_;
  QuantizedInputCache quantized_inputs_;
  // Set by graph steps: whether their decode already checked every output
  // value, and the first non-finite index it found.
  bool graph_checked_ = false;
  size_t graph_first_nonfinite_ = 0;
  // Constant packs already materialised: output buffer -> (request flags they
  // were packed for, reuse token). Survives across requests, so graphs keep
  // the converted copy of these inputs too.
  std::map<std::string, std::pair<uint32_t, uint64_t>> constant_packs_;
  Sidecars sidecars_;
  std::vector<std::string> timings_;
  uint32_t request_flags_ = 0;
  // Raw codes are valid only until another RunElf or a write to their plan buffer.
  Runner* last_int8_runner_ = nullptr;
  std::string last_int8_buffer_;

  // Two adjacent step sequences that do the same work on independent data
  // (GR00T's two cameras). They run interleaved: while the MLA executes one
  // lane's graph on the worker thread, this thread decodes the other lane's
  // previous output and encodes its next input. Lane 1 uses its own instance
  // of every graph (same ELF, separate MLA buffers), so nothing it writes can
  // reach lane 0. Each step computes exactly what it computes sequentially.
  struct TwinBlock {
    size_t begin = 0;   // first step of lane 0
    size_t length = 0;  // steps per lane; lane 1 starts at begin + length
  };
  bool lanes_match(size_t begin, size_t length) const;
  void run_twin_block(const TwinBlock& block);
  // Bookkeeping every step gets after it has written its output.
  void finish_step(const Step& step);
  // Every plan buffer a step writes: its output, plus a device program's downloads.
  std::vector<std::string> written_buffers(const Step& step) const;
  void prepare_parts(const Step& step, size_t expected_elements);
  std::vector<TwinBlock> twin_blocks_;
  std::map<std::string, std::unique_ptr<Runner>> twin_runners_;
  std::unique_ptr<MlaWorker> mla_worker_;

};

}  // namespace polima
