"""Pi0.5's network as static, explicit PyTorch modules for ONNX export.

LeRobot's `PI05Pytorch` runs through HF Gemma/PaliGemma classes with a dynamic
KV cache, 4-D boolean masks built from cumsums, and float64 time embeddings.
None of that survives an NPU compiler. These modules compute the same function
from the same checkpoint tensors, with every shape fixed and everything that
depends only on geometry (RoPE tables, positions) baked in as constants:

  * `Gemma` layers take the attention bias as an input row (0 for a visible key,
    MASK for a padded one) instead of a mask built from a cumsum.
  * The prefix pass returns each layer's K and V, so the denoise graphs take the
    cache as an explicit input rather than appending to a DynamicCache.
  * The action expert takes its adaRMS conditioning vector as an input; the
    vector depends only on the (fixed) Euler timestep, so the host precomputes
    one per step.

Numerics follow LeRobot's `pi_gemma.py` / `modeling_pi05.py` exactly; the
export verifies each graph against the LeRobot reference.

Torch only (no transformers): the weights are read straight from the
checkpoint's model.safetensors by `polima.policies.pi05.weights`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

#: Additive bias for a key that must not be attended to. LeRobot uses
#: -2.38e38, which overflows BF16 and every INT8 scale; -1e4 already drives the
#: softmax weight below 1e-4000 relative to any visible key.
MASK = -1.0e4
ROPE_THETA = 10_000.0
RMS_EPS = 1e-6
HEAD_DIM = 256
HEADS = 8


def rope_tables(positions: torch.Tensor, head_dim: int = HEAD_DIM,
                theta: float = ROPE_THETA) -> tuple[torch.Tensor, torch.Tensor]:
    """HF Gemma rotary embedding: cos/sin of shape [T, head_dim] for integer positions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class Linear(nn.Module):
    """A plain linear layer whose weight comes from a checkpoint tensor."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.weight = nn.Parameter(weight.float().contiguous(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.float().contiguous(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class GemmaMLP(nn.Module):
    def __init__(self, w: dict, prefix: str):
        super().__init__()
        self.gate = Linear(w[f"{prefix}.gate_proj.weight"])
        self.up = Linear(w[f"{prefix}.up_proj.weight"])
        self.down = Linear(w[f"{prefix}.down_proj.weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(gelu_tanh(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Gemma attention with one K/V head shared by eight query heads."""

    def __init__(self, w: dict, prefix: str):
        super().__init__()
        self.q = Linear(w[f"{prefix}.q_proj.weight"])
        self.k = Linear(w[f"{prefix}.k_proj.weight"])
        self.v = Linear(w[f"{prefix}.v_proj.weight"])
        self.o = Linear(w[f"{prefix}.o_proj.weight"])
        self.scale = HEAD_DIM ** -0.5

    def project(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x [1, T, width] -> q [1, H, T, D], k [1, 1, T, D], v [1, 1, T, D] (RoPE on q and k).

        Activations keep their batch axis throughout: ModelSDK's graph surgery
        mis-folds a reshape that adds axes right after a dense layer.
        """
        t = x.shape[1]
        q = self.q(x).reshape(1, t, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).reshape(1, t, 1, HEAD_DIM).transpose(1, 2)
        v = self.v(x).reshape(1, t, 1, HEAD_DIM).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """q [1, H, T, D], k/v [1, 1, S, D], bias [1, 1, 1, S] -> [1, T, H*D].

        Both matmuls are rank-4 batched products with the one K/V head expanded
        to all eight query heads (HF's repeat_kv): ModelSDK maps a matmul whose
        right operand is rank 2 to a dense layer and requires that operand to be
        a constant weight.
        """
        s = k.shape[2]
        k = k.expand(1, HEADS, s, HEAD_DIM)
        v = v.expand(1, HEADS, s, HEAD_DIM)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale + bias
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # [1, H, T, D]
        return out.transpose(1, 2).reshape(1, q.shape[2], HEADS * HEAD_DIM)


class PrefixLayer(nn.Module):
    """One PaliGemma (Gemma-2B) decoder layer, plain RMSNorm, returning its K/V."""

    def __init__(self, w: dict, index: int):
        super().__init__()
        p = f"paligemma_with_expert.paligemma.model.language_model.layers.{index}"
        self.attn = Attention(w, f"{p}.self_attn")
        self.mlp = GemmaMLP(w, f"{p}.mlp")
        self.norm1 = nn.Parameter(1.0 + w[f"{p}.input_layernorm.weight"].float(), requires_grad=False)
        self.norm2 = nn.Parameter(1.0 + w[f"{p}.post_attention_layernorm.weight"].float(), requires_grad=False)

    def forward(self, x, bias, cos, sin):
        q, k, v = self.attn.project(rms(x) * self.norm1, cos, sin)
        x = x + self.attn.o(self.attn.attend(q, k, v, bias))
        x = x + self.mlp(rms(x) * self.norm2)
        return x, k, v


class ExpertLayer(nn.Module):
    """One action-expert (Gemma-300M) layer: adaRMS norms, gated residuals, and
    attention over [prefix K/V ; own K/V]."""

    def __init__(self, w: dict, index: int):
        super().__init__()
        p = f"paligemma_with_expert.gemma_expert.model.layers.{index}"
        self.attn = Attention(w, f"{p}.self_attn")
        self.mlp = GemmaMLP(w, f"{p}.mlp")
        self.mod1 = Linear(w[f"{p}.input_layernorm.dense.weight"], w[f"{p}.input_layernorm.dense.bias"])
        self.mod2 = Linear(w[f"{p}.post_attention_layernorm.dense.weight"],
                           w[f"{p}.post_attention_layernorm.dense.bias"])

    @staticmethod
    def ada(x, modulation):
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return rms(x) * (1.0 + scale) + shift, gate

    def forward(self, x, cond, prefix_k, prefix_v, bias, cos, sin):
        h, gate = self.ada(x, self.mod1(cond))
        q, k, v = self.attn.project(h, cos, sin)
        keys = torch.cat((prefix_k, k), dim=2)
        values = torch.cat((prefix_v, v), dim=2)
        x = x + self.attn.o(self.attn.attend(q, keys, values, bias)) * gate
        h, gate = self.ada(x, self.mod2(cond))
        return x + self.mlp(h) * gate


def time_embedding(t: float, width: int = 1024, min_period: float = 4e-3,
                   max_period: float = 4.0) -> torch.Tensor:
    """LeRobot `create_sinusoidal_pos_embedding` for one scalar timestep, in float64."""
    fraction = torch.linspace(0.0, 1.0, width // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    angle = (2 * math.pi / period) * float(t)
    return torch.cat([torch.sin(angle), torch.cos(angle)]).float()


# ----------------------------------------------------------------- SigLIP

VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
VISION_HEADS = 16


class LayerNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.float(), requires_grad=False)
        self.bias = nn.Parameter(bias.float(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class SiglipLayer(nn.Module):
    """Pre-LN SigLIP encoder layer (So400m/14: width 1152, 16 heads of 72, MLP 4304)."""

    def __init__(self, w: dict, index: int):
        super().__init__()
        p = f"{VISION}.encoder.layers.{index}"
        self.ln1 = LayerNorm(w[f"{p}.layer_norm1.weight"], w[f"{p}.layer_norm1.bias"])
        self.ln2 = LayerNorm(w[f"{p}.layer_norm2.weight"], w[f"{p}.layer_norm2.bias"])
        a = f"{p}.self_attn"
        self.q = Linear(w[f"{a}.q_proj.weight"], w[f"{a}.q_proj.bias"])
        self.k = Linear(w[f"{a}.k_proj.weight"], w[f"{a}.k_proj.bias"])
        self.v = Linear(w[f"{a}.v_proj.weight"], w[f"{a}.v_proj.bias"])
        self.o = Linear(w[f"{a}.out_proj.weight"], w[f"{a}.out_proj.bias"])
        self.fc1 = Linear(w[f"{p}.mlp.fc1.weight"], w[f"{p}.mlp.fc1.bias"])
        self.fc2 = Linear(w[f"{p}.mlp.fc2.weight"], w[f"{p}.mlp.fc2.bias"])

    def forward(self, x):  # x [1, T, 1152]
        _, t, width = x.shape
        head = width // VISION_HEADS
        h = self.ln1(x)
        q = self.q(h).reshape(1, t, VISION_HEADS, head).transpose(1, 2)
        k = self.k(h).reshape(1, t, VISION_HEADS, head).transpose(1, 2)
        v = self.v(h).reshape(1, t, VISION_HEADS, head).transpose(1, 2)
        weights = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * head ** -0.5, dim=-1)
        x = x + self.o(torch.matmul(weights, v).transpose(1, 2).reshape(1, t, width))
        return x + self.fc2(gelu_tanh(self.fc1(self.ln2(x))))


PATCH = 14
GRID = 16
PATCH_VALUES = 3 * PATCH * PATCH  # 588


def patchify(image: torch.Tensor) -> torch.Tensor:
    """[1, 3, 224, 224] -> [1, 256, 588]: row-major 14x14 patches, each laid out (c, y, x),
    the order Conv2d(3, 1152, 14, stride 14) reads them in."""
    return (image.reshape(1, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5)
            .reshape(1, GRID * GRID, PATCH_VALUES))


class Vision(nn.Module):
    """SigLIP So400m/14 at 224x224 plus the PaliGemma projector: patches [1, 256, 588] -> [1, 256, 2048].

    The client cuts the [-1, 1] image into patches (`patchify`), so the patch
    embedding is a dense layer: ModelSDK cannot detessellate a 3-channel input.
    """

    def __init__(self, w: dict, first: int = 0, count: int = 27, stem: bool = True, head: bool = True):
        super().__init__()
        self.stem, self.head = stem, head
        if stem:
            e = f"{VISION}.embeddings"
            self.patch = Linear(w[f"{e}.patch_embedding.weight"].reshape(1152, PATCH_VALUES),
                                w[f"{e}.patch_embedding.bias"])
            self.position = nn.Parameter(w[f"{e}.position_embedding.weight"].float(), requires_grad=False)
        self.layers = nn.ModuleList(SiglipLayer(w, first + i) for i in range(count))
        if head:
            self.post = LayerNorm(w[f"{VISION}.post_layernorm.weight"], w[f"{VISION}.post_layernorm.bias"])
            p = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
            self.project = Linear(w[f"{p}.weight"], w[f"{p}.bias"])

    def forward(self, x):
        if self.stem:
            x = self.patch(x) + self.position
        for layer in self.layers:
            x = layer(x)
        if self.head:
            x = self.project(self.post(x))
        return x
