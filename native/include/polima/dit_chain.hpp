// Experimental multi-input BF16 DiT chain, enabled only by run_dit_chain plans.
//
// Inputs are converted straight into the MLA buffers' CPU mappings, the 16
// graphs go to the MLA runtime as one queued submission (llima
// add_to_queue/run_queue), and the output decode also verifies that every
// value is finite. See runner.hpp for the switches that restore the earlier
// per-graph run() and staged copies.
#pragma once
#include "polima/runner.hpp"
#include <array>

namespace polima {
class DiTRunnerChain {
 public:
  DiTRunnerChain(const std::string& name, const std::vector<SharedStage>& stages,
                 const std::array<size_t, 5>& counts) : counts_(counts) {
    if (stages.size() < 2 || counts[3] != counts[4])
      throw std::runtime_error("Invalid DiT chain geometry");
    for (const auto n : counts) if (!n) throw std::runtime_error("Empty DiT input");
    for (const auto& s : stages)
      if (s.input_elements != counts[0]+counts[1]+counts[2]+counts[3] ||
          s.output_elements != counts[0]) throw std::runtime_error("DiT stage size mismatch");
    for (size_t i=0;i<7;++i) {
      const size_t count = i<5 ? counts[i] : counts[0];
      buffers_.push_back(std::make_unique<DeviceBuffer>(name+"_"+std::to_string(i),count));
      if(i<5) encoded_[i].resize(count);
    }
    raw_.resize(counts[0]);
    for(size_t i=0;i<stages.size();++i) {
      const size_t source=i==0 ? 0 : (i%2 ? 5 : 6), dest=i%2 ? 6 : 5;
      const auto slice=[&](size_t n){return llima::MLABufferSlice(&buffers_[n]->storage);};
      models_.push_back(std::make_unique<llima::MLAModelWithBuffer>(stages[i].elf,
          std::vector<llima::MLABufferSlice>{slice(source),slice(1),slice(2),slice(i%2 ? 4 : 3)},
          std::vector<llima::MLABufferSlice>{slice(dest)}));
      models_.back()->load();
      names_.push_back(stages[i].name);
    }
    final_=stages.size()%2 ? 5 : 6;
    segments_=queue_segments(stages);
  }
  ~DiTRunnerChain(){for(auto& model:models_)model->free();models_.clear();}
  // External request writes are not plan operations; invalidate at every request.
  void begin_request(){valid_.fill(false);}
  void run(const std::array<const float*,5>& inputs,
           const std::array<size_t,5>& versions, float* output, RunTiming* timing=nullptr) {
    using Clock=std::chrono::steady_clock;
    const auto& options=RuntimeOptions::get();
    if(timing) *timing=RunTiming{};
    for(size_t i=0;i<5;++i) {
      // Hidden and timestep always refresh. Shared inputs refresh on any plan write,
      // changed source buffer, or new request; no values leak across observations.
      if(i>=2 && valid_[i] && sources_[i]==inputs[i] && versions_[i]==versions[i])continue;
      auto t=Clock::now();
      detail::write_bf16(buffers_[i]->storage,encoded_[i],inputs[i],0,counts_[i],false);
      if(timing)timing->encode_ms+=detail::since_ms(t);
      t=Clock::now();
      if(options.staged_io)buffers_[i]->storage.upload(encoded_[i].data());
      else buffers_[i]->storage.flush_cache();
      if(timing)timing->upload_ms+=detail::since_ms(t);
      valid_[i]=true;sources_[i]=inputs[i];versions_[i]=versions[i];
    }
    auto t=Clock::now();
    run_models(models_,timing,&names_,segments_);
    if(timing)timing->execute_ms=detail::since_ms(t);
    t=Clock::now();
    if(options.staged_io)buffers_[final_]->storage.download(raw_.data());
    else buffers_[final_]->storage.invalidate_cache();
    if(timing)timing->download_ms=detail::since_ms(t);
    t=Clock::now();
    const auto* raw=options.staged_io ? raw_.data()
        : static_cast<const uint16_t*>(buffers_[final_]->storage.get_virtual_addr());
    first_nonfinite_=kernels::bf16_to_f32_checked(raw,output,counts_[0]);
    if(timing)timing->decode_ms=detail::since_ms(t);
  }
  // Every output value is verified by the decode; counts_[0] means all finite.
  size_t first_nonfinite() const { return first_nonfinite_; }
  size_t output_elements() const { return counts_[0]; }
 private:
  struct DeviceBuffer {
    DeviceBuffer(const std::string& name,size_t count):storage(name,{1,count},"bfloat16",false){storage.allocate();storage.clear();}
    ~DeviceBuffer(){storage.free();}
    llima::MLABuffer storage;
  };
  std::array<size_t,5> counts_,versions_{};
  std::array<const float*,5> sources_{};
  std::array<bool,5> valid_{};
  std::array<std::vector<uint16_t>,5> encoded_;
  std::vector<uint16_t> raw_;
  std::vector<std::unique_ptr<DeviceBuffer>> buffers_;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models_;
  std::vector<std::string> names_;
  size_t final_=0;
  size_t first_nonfinite_=0;
  std::vector<size_t> segments_;
};
}
