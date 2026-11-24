"""Pi0.5 (Physical Intelligence) -- PaliGemma VLM + flow-matching action expert.

## Graph structure

Pi0.5 is SigLIP So400m/14 (27 layers) feeding a Gemma-2B language model (18
layers) whose KV cache conditions a Gemma-300M action expert (18 layers), which
integrates a 50 x 32 action chunk over ten Euler steps. PoLiMa cuts it into:

  vision        SigLIP + projector, one camera per run          [1,3,224,224] -> [1,1,256,2048]
  gemma_AA_BB   two Gemma-2B layers each (nine graphs, once)     hidden, key bias -> hidden + K/V
  expert        the whole action expert for one Euler step       cache, x, cond, rope, bias -> velocity

The language model's output hidden state is never used -- only its KV cache --
so the prefix pass is nine graphs run once, and the denoise loop is one graph run
ten times. `polima.policies.pi05.modules` computes the same function as
LeRobot's `PI05Pytorch` with static shapes; export verifies it against LeRobot.

## Language and state

Pi0.5 has no state projection: the normalized state is discretized into 256 bins
and written into the prompt ("Task: ..., State: 12 200 ...;\\nAction: "). The
prompt therefore changes every request, and the client tokenizes it and sends
its embedding rows (`polima.policies.pi05.client`). See `runtime` for the wire.

## Geometry

The registered spec is `lerobot/pi05_base`'s: three 224x224 cameras
(base, left wrist, right wrist) and 32-wide state and action. A fine-tuned
checkpoint is specialized from its config.json (`profile.specialize_spec`):
camera count, the real state/action width, and for the SO-101 its joints and
camera roles.
"""
from __future__ import annotations

from polima.policies.pi05.profile import Geometry, build_spec
from polima.policies.registry import register_policy

PI05_SPEC = build_spec(Geometry.base())
register_policy(PI05_SPEC)
