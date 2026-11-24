"""The exported graphs: fixed-shape wrappers around `modules`.

Every graph boundary is rank-4 NHWC `[1, 1, rows, width]` and compiled with
`--external-dram-layout HWC`, so the bare runtime reads and writes plain BF16
rows with no tessellation transform (the contract SmolVLA and the GR00T
language partitions already use on Modalix).

Geometry for a checkpoint with C cameras and a 200-token prompt:

    prefix tokens   P = 256*C + 200          (968 for pi05_base's three cameras)
    vision_k        image [1,3,224,224] NCHW          -> tokens [1,1,256,2048]
    gemma_AA_BB     hidden [1,1,P,2048], bias [1,1,1,P] -> [1,1,P*8 + 4*P... ] (see PrefixBlock)
    expert          per Euler step, see ExpertStep
"""
from __future__ import annotations

import torch
from torch import nn

from polima.policies.pi05.modules import (
    HEAD_DIM,
    ExpertLayer,
    Linear,
    PrefixLayer,
    gelu_tanh,
    rms,
    rope_tables,
)

PREFIX_WIDTH = 2048
EXPERT_WIDTH = 1024
ROW = HEAD_DIM  # every packed boundary is [.., rows, 256]


def prefix_output_rows(tokens: int, layers: int) -> int:
    """Rows of a prefix block's output: the hidden state as 8 rows of 256 per token,
    then K and V of each layer (one row of 256 per token)."""
    return tokens * (PREFIX_WIDTH // ROW) + 2 * layers * tokens


class PrefixBlock(nn.Module):
    """Gemma-2B layers [first, first+count): (hidden, key bias) -> (hidden, K/V per layer).

    Positions are 0..P-1: every image token is valid and the valid text tokens
    come first, so a valid token's RoPE position is its index. Padded text
    tokens get a MASK bias as keys; their own outputs are never read.
    The output is one tensor so the runtime downloads one buffer:
    rows [0, 8P) are the hidden state, then K_first, V_first, K_first+1, ...
    """

    def __init__(self, weights: dict, first: int, count: int, tokens: int):
        super().__init__()
        self.layers = nn.ModuleList(PrefixLayer(weights, first + i) for i in range(count))
        self.tokens = tokens
        cos, sin = rope_tables(torch.arange(tokens))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = hidden.reshape(1, self.tokens, PREFIX_WIDTH)
        b = bias.reshape(1, 1, 1, self.tokens)
        cache = []
        for layer in self.layers:
            x, k, v = layer(x, b, self.cos, self.sin)
            cache += [k.reshape(1, self.tokens, ROW), v.reshape(1, self.tokens, ROW)]
        rows = [x.reshape(1, self.tokens * PREFIX_WIDTH // ROW, ROW), *cache]
        return torch.cat(rows, dim=1).reshape(1, 1, -1, ROW)


class ExpertStep(nn.Module):
    """One Euler step's velocity: all 18 action-expert layers plus the action
    projections, over the 50-token chunk.

    Inputs (each its own MLA input, fed in the compiler's input order):
      cache   [1,1,2*L*P,256]  prefix K/V, layer by layer (K then V)
      x       [1,1,50,32]      the noisy action chunk x_t
      cond    [1,1,1,1024]     adaRMS conditioning silu(time_mlp(emb(t)))
      rope    [1,1,100,256]    cos rows 0..49 then sin rows for positions n+0..n+49,
                               n = number of valid prefix tokens
      bias    [1,1,1,P+50]     0 for visible keys, MASK for padded prefix keys
    Output: velocity [1,1,50,32].
    """

    def __init__(self, weights: dict, layers: int, tokens: int, chunk: int = 50, action_lane: int = 32):
        super().__init__()
        self.layers = nn.ModuleList(ExpertLayer(weights, i) for i in range(layers))
        self.tokens, self.chunk, self.lane = tokens, chunk, action_lane
        self.action_in = Linear(weights["action_in_proj.weight"], weights["action_in_proj.bias"])
        self.action_out = Linear(weights["action_out_proj.weight"], weights["action_out_proj.bias"])
        p = "paligemma_with_expert.gemma_expert.model.norm.dense"
        self.final = Linear(weights[f"{p}.weight"], weights[f"{p}.bias"])

    def forward(self, cache, x, cond, rope, bias):
        p, n = self.tokens, self.chunk
        # Every reshape keeps a leading axis of 1: ModelSDK loads graphs with a
        # flexible batch size and rejects nodes whose first axis is not the batch.
        kv = cache.reshape(1, 2 * len(self.layers), p, HEAD_DIM)
        c = cond.reshape(1, 1, EXPERT_WIDTH)
        r = rope.reshape(1, 1, 2 * n, HEAD_DIM)
        cos, sin = r[:, :, :n], r[:, :, n:]
        b = bias.reshape(1, 1, 1, p + n)
        h = self.action_in(x.reshape(1, n, self.lane))
        for index, layer in enumerate(self.layers):
            h = layer(h, c, kv[:, 2 * index:2 * index + 1], kv[:, 2 * index + 1:2 * index + 2], b, cos, sin)
        scale, shift, _ = self.final(c).chunk(3, dim=-1)
        h = rms(h) * (1.0 + scale) + shift
        return self.action_out(h).reshape(1, 1, n, self.lane)


# --------------------------------------------------------------- NCHW expert
#
# GR00T N1.7's DiT rewrite applied to Pi0.5's action expert (policies/groot/dit.py):
# activations as [1, C, 1, L], every Linear as a 1x1 Conv, heads split along the
# channel axis, attention as two Einsums with the softmax over axis 1, and the
# adaRMS modulation (which depends only on the Euler step) computed on the host.
# The prefix K/V input is [1, 256, 2*layers, P] NCHW: with an external HWC DRAM
# layout that is [layer][token][256] in memory -- what LLiMa's pre graphs write.


def _conv(linear: Linear):
    layer = nn.Conv2d(linear.weight.shape[1], linear.weight.shape[0], 1, bias=linear.bias is not None)
    layer.weight = nn.Parameter(linear.weight.detach().clone()[:, :, None, None], requires_grad=False)
    if linear.bias is not None:
        layer.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
    return layer


def _rms_channels(x: torch.Tensor) -> torch.Tensor:  # RMSNorm over C of [1, C, 1, L]
    """Written as ModelSDK's fusible pattern, Mul(x,x) -> ReduceMean -> Add -> Sqrt -> Div ->
    Mul(weight), so it compiles to one RMSNorm kernel. As x * rsqrt(...) it does not fuse,
    and the MLA computes rsqrt through an INT8 lookup (relative error ~6% per expert step)."""
    norm = x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + 1e-6)
    return norm * torch.ones(1, x.shape[1], 1, 1, dtype=x.dtype)


def _rotate_half_channels(x: torch.Tensor) -> torch.Tensor:  # rotate_half over C of [1, D, H, L]
    half = x.shape[1] // 2
    return torch.cat((-x[:, half:], x[:, :half]), dim=1)


class ExpertLayerNchw(nn.Module):
    def __init__(self, layer: ExpertLayer):
        super().__init__()
        a = layer.attn
        self.q, self.k, self.v, self.o = _conv(a.q), _conv(a.k), _conv(a.v), _conv(a.o)
        self.gate, self.up, self.down = _conv(layer.mlp.gate), _conv(layer.mlp.up), _conv(layer.mlp.down)
        self.scale = a.scale

    def heads(self, t):  # [1, H*D, 1, L] -> [1, D, H, L]
        return torch.cat(torch.split(t, HEAD_DIM, dim=1), dim=2)

    def forward(self, x, mod1, mod2, prefix_k, prefix_v, mask, cos, sin):
        c = x.shape[1]
        h = _rms_channels(x) * (1.0 + mod1[:, :c]) + mod1[:, c:2 * c]
        q = self.heads(self.q(h))                                  # [1, 256, 8, L]
        k, v = self.k(h), self.v(h)                                # [1, 256, 1, L]
        q = q * cos + _rotate_half_channels(q) * sin
        k = k * cos + _rotate_half_channels(k) * sin
        keys = torch.cat((prefix_k, k), dim=3).expand(-1, -1, 8, -1)    # [1, 256, 8, S]
        values = torch.cat((prefix_v, v), dim=3).expand(-1, -1, 8, -1)
        s = torch.einsum("nchw,nchq->nqhw", q * self.scale, keys) + mask  # [1, S, 8, L]
        o = torch.einsum("nchw,nqhc->nqhw", torch.softmax(s, dim=1), values)  # [1, 256, 8, L]
        x = x + self.o(torch.cat(torch.split(o, 1, dim=2), dim=1)) * mod1[:, 2 * c:]
        h = _rms_channels(x) * (1.0 + mod2[:, :c]) + mod2[:, c:2 * c]
        return x + self.down(gelu_tanh(self.gate(h)) * self.up(h)) * mod2[:, 2 * c:]


class ExpertStepNchw(nn.Module):
    """One Euler step's velocity, NCHW form with host adaRMS.

    Inputs: x [1,32,1,50], ada [1, (2*layers+1)*3072, 1, 1] (per-step modulation,
    `ada_vectors`), cache [1, 256, 2*layers, P], rope [1, 512, 1, 50] (cos then
    sin), mask [1, P+50, 1, 1]. Output: velocity [1, 32, 1, 50].
    """

    def __init__(self, step: ExpertStep):
        super().__init__()
        self.layers = nn.ModuleList(ExpertLayerNchw(layer) for layer in step.layers)
        self.action_in, self.action_out = _conv(step.action_in), _conv(step.action_out)
        self.width = EXPERT_WIDTH

    def forward(self, x, ada, cache, rope, mask):
        m = 3 * self.width
        return self.velocity(x, [ada[:, j * m:(j + 1) * m] for j in range(2 * len(self.layers) + 1)],
                             cache, rope, mask)

    def velocity(self, x, mods, cache, rope, mask):
        """`mods`: each norm's [1, 3072, 1, 1] [scale | shift | gate], layer by layer, final last."""
        c = self.width
        cos, sin = rope[:, :HEAD_DIM], rope[:, HEAD_DIM:]
        h = self.action_in(x)
        for i, layer in enumerate(self.layers):
            h = layer(h, mods[2 * i], mods[2 * i + 1],
                      cache[:, :, 2 * i:2 * i + 1], cache[:, :, 2 * i + 1:2 * i + 2], mask, cos, sin)
        h = _rms_channels(h) * (1.0 + mods[-1][:, :c]) + mods[-1][:, c:2 * c]
        return self.action_out(h)


class ExpertEulerNchw(nn.Module):
    """ExpertStepNchw with the Euler update folded in: returns x - dt * velocity, in x's
    own [1, 32, 1, 50] layout, so one step's output is the next step's input and the
    denoising loop stays on the device (runtime.build_llima_plan, denoise="device").

    Its adaRMS input is [1, 3072, norms, 1] -- one row per norm -- with the same HWC bytes
    as ExpertStepNchw's [1, norms*3072, 1, 1]: as its own MLA binding, the flat
    113664-channel vector trips ModelSDK 2.1.3's large-tensor input handling."""

    def __init__(self, step: ExpertStepNchw, dt: float):
        super().__init__()
        self.step, self.dt = step, dt

    def forward(self, x, ada, cache, rope, mask):
        mods = [ada[:, :, j:j + 1] for j in range(ada.shape[2])]
        return x - self.dt * self.step.velocity(x, mods, cache, rope, mask)


def euler_ada(ada: torch.Tensor) -> torch.Tensor:
    """ExpertStepNchw's [1, norms*3072, 1, 1] adaRMS as ExpertEulerNchw's [1, 3072, norms, 1]
    (identical HWC bytes, so the ada_XX sidecars serve both)."""
    return ada.reshape(-1, 3 * EXPERT_WIDTH).T.reshape(1, 3 * EXPERT_WIDTH, -1, 1)


@torch.no_grad()
def ada_vectors(step: ExpertStep, cond: torch.Tensor) -> torch.Tensor:
    """The per-step adaRMS modulation: every layer's two norms, then the final norm,
    each [scale | shift | gate] of 3*1024 -- [1, (2*layers+1)*3072, 1, 1]."""
    c = cond.reshape(1, EXPERT_WIDTH)
    parts = []
    for layer in step.layers:
        parts += [layer.mod1(c), layer.mod2(c)]
    parts.append(step.final(c))
    return torch.cat(parts, dim=1).reshape(1, -1, 1, 1)


def to_nchw_inputs(cache, x, cond, rope, bias, step: ExpertStep, tokens: int):
    """The NCHW expert's inputs from ExpertStep's (NHWC) inputs."""
    layers = len(step.layers)
    n = rope.shape[2] // 2
    return (x.reshape(1, n, -1).permute(0, 2, 1).reshape(1, -1, 1, n),
            ada_vectors(step, cond),
            cache.reshape(2 * layers, tokens, HEAD_DIM).permute(2, 0, 1).reshape(1, HEAD_DIM, 2 * layers, tokens),
            rope.reshape(2, n, HEAD_DIM).permute(0, 2, 1).reshape(1, 2 * HEAD_DIM, 1, n),
            bias.reshape(1, -1, 1, 1))
