"""SmolVLA checkpoint -> the four fixed-shape Modalix ONNX graphs.

This is the PoLiMa-native form of the proven standalone exporters under
``SmolVLA/scripts``.  The public graph boundaries match ``runtime.py``:

    vision(512x512 image) -> 64x960 tokens
    prefix(241x960 embeddings) -> packed K/V cache
    suffix(50x[32 action + 720 time]) -> 50x720 embeddings
    denoise(packed cache + suffix) -> 50x32 velocity

The exporter deliberately keeps prefix construction, sinusoidal time features,
and Euler integration on the host/runtime side.  Only learned tensor operations
are compiled into ELFs.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from polima.policies.smolvla.runtime import (
    ACTION_LANE,
    CHUNK,
    DENOISE_STEPS,
    DENOISE_TOKENS,
    DENOISE_WIDTH,
    EXPERT_HIDDEN,
    HIDDEN,
    IMAGE_SCALE,
    LANGUAGE_TOKENS,
    PREFIX_TOKENS,
    STATE_TOKEN_INDEX,
    SUFFIX_TOKEN,
)

EXTRA_OBSERVATION_KEYS = ("task",)
# BF16 compilation has no activation scales.  One trace supplies the export
# examples, fixture, sidecars, and verification reference; tracing all eight
# dataset samples would repeat the 16-layer denoise loop for no compile benefit.
TRACE_FIRST_ONLY = True
FIXTURE_FILE = "smolvla_fixture.npz"


# ------------------------------------------------------------- environment/load


def _lerobot_source(explicit: str | Path | None = None) -> Path | None:
    """Optional development override; installed LeRobot is the default."""
    return Path(explicit).resolve() if explicit else None


def _prepare_imports(lerobot_src: str | Path | None = None) -> None:
    source = _lerobot_source(lerobot_src)
    if source and str(source) not in sys.path:
        sys.path.insert(0, str(source))

    # The provisioned compiler environment intentionally combines the torch
    # stack used for export with a newer Transformers/PyAV.  These two narrow
    # compatibility aliases are the same no-op adaptations used by the working
    # legacy controller; neither changes model arithmetic.
    try:
        import transformers.utils as transformers_utils

        if not hasattr(transformers_utils, "torch_compilable_check"):
            transformers_utils.torch_compilable_check = lambda function: function
    except ImportError:
        pass
    try:
        import av

        if not hasattr(av, "option"):
            av.option = types.SimpleNamespace(Option=object)
    except ImportError:
        pass


def _real_image_keys(checkpoint: Path, policy) -> list[str]:
    """Names after the checkpoint's rename processor, excluding empty slots."""
    manifest = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    for step in manifest.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            values = list((step.get("config", {}).get("rename_map") or {}).values())
            if values:
                return values
    return [key for key in policy.config.image_features if "empty_camera" not in key][:2]


def load_policy(checkpoint: str | Path, lerobot_src: str | Path | None = None):
    _prepare_imports(lerobot_src)
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint).resolve()
    config = SmolVLAConfig.from_pretrained(checkpoint, local_files_only=True)
    # Export and ONNX verification are CPU operations.  Checkpoints commonly
    # retain device="cuda" from training; overriding that runtime preference
    # avoids requiring CUDA (or even probing for it) in PoLiMa's host venv.
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True
    ).cpu().eval()
    config = policy.config
    image_keys = _real_image_keys(checkpoint, policy)

    problems: list[str] = []
    if config.type != "smolvla":
        problems.append(f"type={config.type}, expected smolvla")
    if config.chunk_size != CHUNK or config.n_action_steps != CHUNK:
        problems.append(f"chunk/action steps must both be {CHUNK}")
    if config.max_state_dim != ACTION_LANE or config.max_action_dim != ACTION_LANE:
        problems.append(f"state/action lanes must both be {ACTION_LANE}")
    if config.num_steps != DENOISE_STEPS:
        problems.append(f"num_steps must be {DENOISE_STEPS}")
    if tuple(config.resize_imgs_with_padding or ()) != (512, 512):
        problems.append("resize_imgs_with_padding must be [512, 512]")
    if config.tokenizer_max_length != LANGUAGE_TOKENS:
        problems.append(f"tokenizer_max_length must be {LANGUAGE_TOKENS}")
    if config.empty_cameras != 1 or len(image_keys) != 2:
        problems.append(f"expected two real cameras plus one empty slot, got {image_keys}")
    if config.num_vlm_layers != 16:
        problems.append("the cache layout requires 16 VLM layers")
    if config.add_image_special_tokens:
        problems.append("image special tokens change the fixed 241-token prefix")
    if not config.use_cache:
        problems.append("compiled denoising requires prefix cache support")
    if problems:
        raise ValueError(
            f"unsupported SmolVLA checkpoint {checkpoint}:\n  - " + "\n  - ".join(problems)
        )
    return policy, image_keys


# --------------------------------------------------------------------- modules


def vision_position_ids(side: int, per_side: int, dtype: torch.dtype) -> torch.Tensor:
    """SmolVLM's patch position ids for a fully valid side x side patch grid.

    Transformers computes the fractional patch coordinates in the *pixel dtype*
    (``arange(n, dtype=pixel_values.dtype) / n * (1 - 1e-6)``). In BF16, which is
    how LeRobot runs SmolVLA's frozen vision tower and what the policy was trained
    on, ``1 - 1e-6`` rounds to 1 and the 32 x 32 grid maps to positions 0..31; in
    FP32 every coordinate lands just below its bucket and the grid shifts to
    0, 0, 1, ..., 30 (vision features at cosine 0.88 of the BF16 ones). So the
    ids must be computed in the dtype the checkpoint's vision tower runs in."""
    boundaries = torch.arange(1 / per_side, 1.0, 1 / per_side)
    coordinates = torch.arange(side, dtype=dtype) / torch.tensor(side) * (1 - 1e-6)
    buckets = torch.bucketize(coordinates, boundaries, right=True)
    return (buckets[:, None] * per_side + buckets).flatten().unsqueeze(0)


class VisionTower(nn.Module):
    def __init__(self, model, position_dtype: torch.dtype | None = None):
        super().__init__()
        self.vision = model.vlm_with_expert.get_vlm_model().vision_model
        self.connector = model.vlm_with_expert.get_vlm_model().connector
        side = 512 // self.vision.patch_size
        position_ids = vision_position_ids(
            side, self.vision.embeddings.num_patches_per_side, position_dtype or self.vision.dtype
        )
        self.register_buffer("position_ids", position_ids, persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # The input is always a fully-valid 512x512 square.  Express the
        # resulting position ids statically; Transformers' general padded-image
        # path builds them with an in-place boolean scatter that Torch 2.3
        # incorrectly exports as float Gather indices.
        image = image.to(dtype=self.vision.dtype)
        embeddings = self.vision.embeddings
        patches = embeddings.patch_embedding(image).flatten(2).transpose(1, 2)
        hidden = patches + embeddings.position_embedding(self.position_ids)
        hidden = self.vision.encoder(
            inputs_embeds=hidden, attention_mask=None
        ).last_hidden_state
        hidden = self.vision.post_layernorm(hidden)
        return self.connector(hidden)


def fusible_rms(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A Llama RMSNorm (``weight``, ``variance_epsilon``) over the last axis, written as
    ModelSDK's fusible pattern Mul(x,x) -> ReduceMean -> Add -> Sqrt -> Div -> Mul(weight).

    Transformers writes it as x * rsqrt(mean(x^2) + eps), which ModelSDK does not fuse:
    rsqrt becomes an INT8 lookup between a quantize/dequantize pair (about 6% relative
    error per step in Pi0.5's expert). Compile with --no-simplify, or the simplifier
    rewrites the pattern."""
    x = x.float()
    normed = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    return normed * norm.weight.float()


class PrefixCache(nn.Module):
    """Static 16-layer VLM prefill producing 32 packed K/V buffers."""

    def __init__(self, model, language_tokens: int = 10):
        super().__init__()
        owner = model.vlm_with_expert
        self.layers = owner.get_vlm_model().text_model.layers
        self.num_heads = owner.num_attention_heads
        self.num_kv_heads = owner.num_key_value_heads
        self.head_dim = owner.config.text_config.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        if len(self.layers) != 16:
            raise ValueError(f"expected 16 VLM layers, got {len(self.layers)}")

        pad = torch.zeros(1, PREFIX_TOKENS, dtype=torch.bool)
        pad[:, :128] = True
        pad[:, 192:192 + language_tokens] = True
        pad[:, STATE_TOKEN_INDEX] = True
        attention_types = torch.zeros(1, PREFIX_TOKENS, dtype=torch.bool)
        attention_types[:, STATE_TOKEN_INDEX] = True
        blocks = torch.cumsum(attention_types, dim=1)
        mask = (blocks[:, None, :] <= blocks[:, :, None]) & (
            pad[:, None, :] & pad[:, :, None]
        )
        self.register_buffer(
            "attention_bias",
            torch.where(mask[:, None], torch.tensor(0.0), torch.tensor(-1.0e9)),
            persistent=False,
        )
        positions = torch.cumsum(pad, dim=1) - 1
        half = self.head_dim // 2
        exponents = (2.0 / self.head_dim) * torch.arange(half, dtype=torch.float32)
        radians = positions[..., None].float() / (10000**exponents)[None, None, :]
        self.register_buffer("rope_sin", torch.sin(radians)[..., None, :], persistent=False)
        self.register_buffer("rope_cos", torch.cos(radians)[..., None, :], persistent=False)

    def rope(self, value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        dtype = value.dtype
        first, second = value.float().split(half, dim=-1)
        return torch.cat(
            [first * self.rope_cos - second * self.rope_sin,
             second * self.rope_cos + first * self.rope_sin], dim=-1
        ).to(dtype)

    def attention(self, query, key, value):
        batch, length = query.shape[:2]
        key = key[:, :, :, None, :].expand(
            batch, length, self.num_kv_heads, self.groups, self.head_dim
        ).reshape(batch, length, self.num_heads, self.head_dim)
        value = value[:, :, :, None, :].expand(
            batch, length, self.num_kv_heads, self.groups, self.head_dim
        ).reshape(batch, length, self.num_heads, self.head_dim)
        scores = torch.matmul(
            query.float().transpose(1, 2), key.float().transpose(1, 2).transpose(2, 3)
        )
        probability = nn.functional.softmax(
            scores * (self.head_dim**-0.5) + self.attention_bias, dim=-1
        ).to(value.dtype)
        output = torch.matmul(probability, value.permute(0, 2, 1, 3))
        return output.permute(0, 2, 1, 3).reshape(batch, length, -1)

    def forward(self, prefix_embeddings: torch.Tensor) -> torch.Tensor:
        hidden = prefix_embeddings.reshape(1, PREFIX_TOKENS, HIDDEN)
        caches = []
        for layer in self.layers:
            normed = fusible_rms(layer.input_layernorm, hidden).to(layer.self_attn.q_proj.weight.dtype)
            query = layer.self_attn.q_proj(normed).view(1, PREFIX_TOKENS, -1, self.head_dim)
            key = layer.self_attn.k_proj(normed).view(1, PREFIX_TOKENS, -1, self.head_dim)
            value = layer.self_attn.v_proj(normed).view(1, PREFIX_TOKENS, -1, self.head_dim)
            query, key = self.rope(query), self.rope(key)
            caches.extend([
                key.transpose(1, 2).reshape(1, -1),
                value.transpose(1, 2).reshape(1, -1),
            ])
            attended = self.attention(query, key, value).to(layer.self_attn.o_proj.weight.dtype)
            residual = layer.self_attn.o_proj(attended) + hidden.to(attended.dtype)
            hidden = layer.mlp(fusible_rms(layer.post_attention_layernorm, residual)) + residual
        return torch.stack(caches, dim=1).unsqueeze(1)


class SuffixProjection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.action_in = model.action_in_proj
        self.mlp_in = model.action_time_mlp_in
        self.mlp_out = model.action_time_mlp_out

    def forward(self, suffix_input: torch.Tensor) -> torch.Tensor:
        packed = suffix_input.reshape(1, CHUNK, SUFFIX_TOKEN)
        action = self.action_in(packed[:, :, :ACTION_LANE])
        hidden = nn.functional.silu(self.mlp_in(torch.cat([action, packed[:, :, ACTION_LANE:]], 2)))
        return self.mlp_out(hidden).unsqueeze(1)


class DenoiseCore(nn.Module):
    """The 16-layer action expert with explicit prefix-cache tensors."""

    def __init__(self, model, language_tokens: int = 10):
        super().__init__()
        owner = model.vlm_with_expert
        self.expert = owner.lm_expert
        self.self_every = owner.self_attn_every_n_layers
        self.num_heads = owner.num_attention_heads
        self.num_kv_heads = owner.num_key_value_heads
        self.head_dim = owner.config.text_config.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        self.prefix_len = PREFIX_TOKENS
        self.suffix_len = CHUNK

        prefix_pad = torch.zeros(1, PREFIX_TOKENS, dtype=torch.bool)
        prefix_pad[:, :128] = True
        prefix_pad[:, 192:192 + language_tokens] = True
        prefix_pad[:, STATE_TOKEN_INDEX] = True
        suffix_causal = torch.tril(torch.ones(1, CHUNK, CHUNK, dtype=torch.bool))
        prefix_mask = prefix_pad[:, None, :].expand(1, CHUNK, PREFIX_TOKENS)
        self.register_buffer(
            "self_attention_bias",
            self._bias(torch.cat([prefix_mask, suffix_causal], dim=2)),
            persistent=False,
        )
        self.register_buffer("cross_attention_bias", self._bias(prefix_mask), persistent=False)
        valid_prefix = 128 + language_tokens + 1
        self._register_rope("cross", torch.arange(CHUNK))
        self._register_rope("self", torch.arange(valid_prefix, valid_prefix + CHUNK))

    @staticmethod
    def _bias(mask):
        return torch.where(mask[:, None], torch.tensor(0.0), torch.tensor(-1.0e9))

    def _register_rope(self, prefix, positions):
        half = self.head_dim // 2
        timescale = 10000 ** ((2.0 / self.head_dim) * torch.arange(half).float())
        radians = positions[None, :, None, None].float() / timescale[None, None, None, :]
        self.register_buffer(f"{prefix}_rope_sin", torch.sin(radians), persistent=False)
        self.register_buffer(f"{prefix}_rope_cos", torch.cos(radians), persistent=False)

    def _rope(self, value, prefix):
        half = value.shape[-1] // 2
        dtype = value.dtype
        first, second = value.float().split(half, -1)
        sin, cos = getattr(self, f"{prefix}_rope_sin"), getattr(self, f"{prefix}_rope_cos")
        return torch.cat([first * cos - second * sin, second * cos + first * sin], -1).to(dtype)

    def _attention(self, bias, query, key, value):
        batch, length = query.shape[0], key.shape[1]
        key = key[:, :, :, None, :].expand(
            batch, length, self.num_kv_heads, self.groups, self.head_dim
        ).reshape(batch, length, self.num_heads, self.head_dim)
        value = value[:, :, :, None, :].expand(
            batch, length, self.num_kv_heads, self.groups, self.head_dim
        ).reshape(batch, length, self.num_heads, self.head_dim)
        weights = torch.matmul(query.float().transpose(1, 2), key.float().transpose(1, 2).transpose(2, 3))
        probability = nn.functional.softmax(weights * (self.head_dim**-0.5) + bias, -1).to(value.dtype)
        output = torch.matmul(probability, value.permute(0, 2, 1, 3))
        return output.permute(0, 2, 1, 3).reshape(batch, CHUNK, -1)

    def forward(self, suffix_embeddings: torch.Tensor, *cache_tensors) -> torch.Tensor:
        hidden = suffix_embeddings
        for index, layer in enumerate(self.expert.layers):
            cache_k = cache_tensors[index * 2].transpose(1, 2)
            cache_v = cache_tensors[index * 2 + 1].transpose(1, 2)
            normed = fusible_rms(layer.input_layernorm, hidden).to(layer.self_attn.q_proj.weight.dtype)
            query = layer.self_attn.q_proj(normed).view(1, CHUNK, -1, self.head_dim)
            if self.self_every > 0 and index % self.self_every == 0:
                key = layer.self_attn.k_proj(normed).view(1, CHUNK, -1, self.head_dim)
                value = layer.self_attn.v_proj(normed).view(1, CHUNK, -1, self.head_dim)
                query, key = self._rope(query, "self"), self._rope(key, "self")
                key, value = torch.cat([cache_k, key], 1), torch.cat([cache_v, value], 1)
                attended = self._attention(self.self_attention_bias, query, key, value)
            else:
                flat_k = cache_k.to(layer.self_attn.k_proj.weight.dtype).reshape(1, PREFIX_TOKENS, -1)
                flat_v = cache_v.to(layer.self_attn.v_proj.weight.dtype).reshape(1, PREFIX_TOKENS, -1)
                key = layer.self_attn.k_proj(flat_k).view(1, PREFIX_TOKENS, -1, self.head_dim)
                value = layer.self_attn.v_proj(flat_v).view(1, PREFIX_TOKENS, -1, self.head_dim)
                attended = self._attention(
                    self.cross_attention_bias, self._rope(query, "cross"), key, value
                )
            residual = layer.self_attn.o_proj(
                attended.to(layer.self_attn.o_proj.weight.dtype)
            ) + hidden
            hidden = layer.mlp(fusible_rms(layer.post_attention_layernorm, residual)) + residual
        return fusible_rms(self.expert.norm, hidden)


class DenoiseExpert(nn.Module):
    def __init__(self, model, language_tokens: int = 10):
        super().__init__()
        self.core = DenoiseCore(model)
        self.action_out = model.action_out_proj

    def forward(self, denoise_input: torch.Tensor) -> torch.Tensor:
        packed = denoise_input.reshape(1, DENOISE_TOKENS, DENOISE_WIDTH)
        cache = packed[:, :1205].reshape(1, 32, 77120)
        suffix = packed[:, 1205:].reshape(1, -1)[:, :CHUNK * EXPERT_HIDDEN]
        suffix = suffix.reshape(1, CHUNK, EXPERT_HIDDEN)
        caches = tuple(cache[:, index].reshape(1, 5, PREFIX_TOKENS, 64) for index in range(32))
        hidden = self.core(suffix, *caches)
        return self.action_out(hidden.float())


class VisionConnector(nn.Module):
    """SmolVLM's connector on SigLIP's output tokens, with the prefix's image scale folded in:
    tokens [1,1,1024,768] (post_layernorm) -> pixel shuffle x4 -> projection -> x sqrt(960)
    -> [1,1,64,960]. Pairs with a vision tower compiled without its connector (LLiMa)."""

    def __init__(self, model):
        super().__init__()
        self.connector = model.vlm_with_expert.get_vlm_model().connector

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        width = tokens.shape[-1]
        hidden = self.connector(tokens.reshape(1, -1, width))
        return (hidden * IMAGE_SCALE).reshape(1, 1, -1, HIDDEN)


#: Rows of the device-resident K/V cache as the denoise step reads it: the prefix's
#: 32 x 77120 output (16 layers x K/V x 5 heads x 241 tokens x 64) as 1205 rows of 2048.
#: Same bytes; rows keep each binding's channel count small for ModelSDK.
CACHE_ROWS = 1205


def time_embedding(timestep: float) -> torch.Tensor:
    """SmolVLA's sinusoidal time features (min period 0.004, max 4.0): [720]."""
    fraction = torch.arange(EXPERT_HIDDEN // 2, dtype=torch.float32) / (EXPERT_HIDDEN // 2 - 1)
    period = 0.004 * torch.pow(torch.tensor(1000.0), fraction)
    angle = timestep * (2.0 * torch.pi / period)
    return torch.cat([torch.sin(angle), torch.cos(angle)])


class DenoiseStep(nn.Module):
    """One Euler step on the device: x [1,1,50,32], tbias [1,1,1,720], cache [1,1,1205,2048]
    -> x - dt * velocity [1,1,50,32].

    Folds the suffix graph in. Its time MLP input, mlp_in(cat[action, time]), splits into
    W_action @ action + (W_time @ time + b): the second term depends only on the step, so
    the host precomputes it as `step_time_bias(k)` and the graph adds it. The K/V cache is
    the prefix graph's output, left in a device buffer for all ten steps."""

    def __init__(self, model, core: DenoiseCore):
        super().__init__()
        self.core = core
        self.action_in = model.action_in_proj
        width = model.action_time_mlp_in.weight.shape[0]
        self.register_buffer("w_action", model.action_time_mlp_in.weight[:, :width].detach().float(),
                             persistent=False)
        self.mlp_out = model.action_time_mlp_out
        self.action_out = model.action_out_proj

    def forward(self, x: torch.Tensor, tbias: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        actions = x.reshape(1, CHUNK, ACTION_LANE)
        hidden = nn.functional.linear(self.action_in(actions), self.w_action) + tbias.reshape(1, 1, -1)
        suffix = self.mlp_out(nn.functional.silu(hidden))
        layers = cache.reshape(1, 32, -1)
        caches = tuple(layers[:, index].reshape(1, 5, PREFIX_TOKENS, 64) for index in range(32))
        velocity = self.action_out(self.core(suffix, *caches).float())
        return (actions - velocity * (1.0 / DENOISE_STEPS)).reshape(1, 1, CHUNK, ACTION_LANE)


def step_time_bias(model, step: int) -> torch.Tensor:
    """The per-step constant of DenoiseStep's time MLP: W_time @ time(t_k) + b, [720]."""
    mlp_in = model.action_time_mlp_in
    width = mlp_in.weight.shape[0]
    time = time_embedding(1.0 - step / DENOISE_STEPS)
    return nn.functional.linear(time, mlp_in.weight[:, width:].float(), mlp_in.bias.float()).detach()


def _keep_native_positions(embeddings: nn.Module, dtype: torch.dtype) -> None:
    """After promoting the vision tower to FP32, keep LeRobot's own forward (the
    reference that trace() checks the exported graphs against) on the position ids
    of the dtype it originally ran in, rather than FP32's shifted ones."""
    if dtype == torch.float32:
        return

    def forward(pixel_values, patch_attention_mask):
        side = pixel_values.shape[-1] // embeddings.patch_size
        if pixel_values.shape[-2] != pixel_values.shape[-1] or not bool(patch_attention_mask.all()):
            raise ValueError("SmolVLA export expects fully valid square images")
        patches = embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        ids = vision_position_ids(side, embeddings.num_patches_per_side, dtype).to(pixel_values.device)
        return patches + embeddings.position_embedding(ids.expand(pixel_values.shape[0], -1))

    embeddings.forward = forward


def build_modules(policy) -> dict[str, nn.Module]:
    model = policy.model
    vlm = model.vlm_with_expert.get_vlm_model()
    # Patch position ids depend on the dtype LeRobot runs the vision tower in
    # (see vision_position_ids); record it before promoting to FP32.
    vision_dtype = vlm.vision_model.dtype
    # The legacy production exporters explicitly promote these submodules.
    # Doing it once here also makes ONNX Runtime verification deterministic on CPU.
    vlm.vision_model.float()
    # Torch 2.3's ONNX symbolic for scaled_dot_product_attention cannot encode
    # the Python float scale used by this Transformers release.  Eager attention
    # is the same QK-softmax-V arithmetic expressed as primitive tensor ops and
    # is what the compiler can import.
    vlm.vision_model.config._attn_implementation = "eager"
    vlm.connector.float()
    vlm.text_model.float()
    model.vlm_with_expert.lm_expert.float()
    _keep_native_positions(vlm.vision_model.embeddings, vision_dtype)
    return {
        "_policy": policy,
        "vision": VisionTower(model, position_dtype=vision_dtype).eval(),
        "prefix": PrefixCache(model).eval(),
        "suffix": SuffixProjection(model).eval(),
        "denoise": DenoiseExpert(model).eval(),
    }


# ---------------------------------------------------------------- tracing/pack


def _suffix_input(actions: torch.Tensor, timestep: float) -> torch.Tensor:
    fraction = torch.arange(EXPERT_HIDDEN // 2, dtype=torch.float32) / (EXPERT_HIDDEN // 2 - 1)
    period = 0.004 * torch.pow(torch.tensor(1000.0), fraction)
    angle = timestep * (2.0 * torch.pi / period)
    time = torch.cat([torch.sin(angle), torch.cos(angle)])
    packed = torch.zeros(1, 1, CHUNK, SUFFIX_TOKEN)
    packed[0, 0, :, :ACTION_LANE] = actions.reshape(CHUNK, ACTION_LANE)
    packed[0, 0, :, ACTION_LANE:] = time
    return packed


def _denoise_input(cache: torch.Tensor, suffix: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(1, 1, DENOISE_TOKENS, DENOISE_WIDTH)
    packed[:, :, :1205] = cache.reshape(1, 1, 1205, DENOISE_WIDTH)
    packed[:, :, 1205:].reshape(1, -1)[:, :CHUNK * EXPERT_HIDDEN] = suffix.reshape(1, -1)
    return packed


def trace(modules: dict[str, nn.Module], sample: dict, image_keys: Sequence[str]) -> dict:
    policy = modules["_policy"]
    state_dim = int(sample["observation.state"].shape[-1])
    action_dim = int(policy.config.action_feature.shape[0])
    if not 0 < state_dim <= ACTION_LANE or not 0 < action_dim <= ACTION_LANE:
        raise ValueError(
            f"SmolVLA sample dimensions {state_dim}/{action_dim} do not fit "
            f"the {ACTION_LANE}-wide model lane"
        )
    with torch.inference_mode():
        images, masks = policy.prepare_images(sample)
        state_lane = policy.prepare_state(sample)
        tokens = sample["observation.language.tokens"]
        language_mask = sample["observation.language.attention_mask"]
        valid_language = int(language_mask.sum())
        if not 0 < valid_language <= LANGUAGE_TOKENS:
            raise ValueError(
                f"SmolVLA task must occupy 1..{LANGUAGE_TOKENS} language tokens, "
                f"got {valid_language}"
            )
        # The attention masks and suffix RoPE positions are constants in the
        # exported graphs, so specialize both modules to this deployment task.
        modules["prefix"] = PrefixCache(policy.model, valid_language).eval()
        modules["denoise"] = DenoiseExpert(policy.model, valid_language).eval()
        prefix_reference, pad_mask, _ = policy.model.embed_prefix(
            images, masks, tokens, language_mask, state=state_lane
        )
        expected_pad = torch.zeros_like(pad_mask)
        expected_pad[:, :128] = True
        expected_pad[:, 192:192 + valid_language] = True
        expected_pad[:, STATE_TOKEN_INDEX] = True
        if not torch.equal(pad_mask.bool(), expected_pad.bool()):
            raise ValueError("compiled prefix mask does not match the processed task mask")

        vision = [modules["vision"](images[index]) for index in range(2)]
        empty = prefix_reference[:, 128:192]
        language = prefix_reference[:, 192:240]
        state_embedding = policy.model.state_proj(state_lane)
        prefix = torch.zeros(1, PREFIX_TOKENS, HIDDEN)
        prefix[:, :64] = vision[0] * IMAGE_SCALE
        prefix[:, 64:128] = vision[1] * IMAGE_SCALE
        prefix[:, 128:192] = empty
        prefix[:, 192:240] = language
        prefix[:, STATE_TOKEN_INDEX] = state_embedding
        if not torch.allclose(prefix, prefix_reference.float(), atol=1e-5, rtol=1e-5):
            block_diffs = {
                "vision0": float((prefix[:, :64] - prefix_reference[:, :64]).abs().max()),
                "vision1": float((prefix[:, 64:128] - prefix_reference[:, 64:128]).abs().max()),
                "empty": float((prefix[:, 128:192] - prefix_reference[:, 128:192]).abs().max()),
                "language": float((prefix[:, 192:240] - prefix_reference[:, 192:240]).abs().max()),
                "state": float((prefix[:, 240:] - prefix_reference[:, 240:]).abs().max()),
            }
            raise ValueError(
                "runtime prefix packing does not reproduce SmolVLA embed_prefix: "
                f"{block_diffs}"
            )
        cache = modules["prefix"](prefix.unsqueeze(1))

        generator = torch.Generator(device="cpu").manual_seed(20260805)
        actions = torch.randn(1, CHUNK, ACTION_LANE, generator=generator)
        noise = actions.clone()
        suffix_inputs, denoise_inputs, velocities = [], [], []
        for index in range(DENOISE_STEPS):
            suffix_input = _suffix_input(actions, 1.0 - index / DENOISE_STEPS)
            suffix = modules["suffix"](suffix_input)
            denoise_input = _denoise_input(cache, suffix)
            velocity = modules["denoise"](denoise_input)
            suffix_inputs.append(suffix_input)
            denoise_inputs.append(denoise_input)
            velocities.append(velocity)
            actions = actions - velocity * (1.0 / DENOISE_STEPS)

    return {
        "prepared_images": images[:2],
        "state_lane": state_lane,
        "prefix": prefix,
        "cache": cache,
        "noise": noise,
        "suffix_inputs": suffix_inputs,
        "denoise_inputs": denoise_inputs,
        "velocities": velocities,
        "normalized_action": actions[:, :, :action_dim],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "empty_image_embedding": empty,
        "language_embedding": language,
        "state_project_weight": policy.model.state_proj.weight.detach().float(),
        "state_project_bias": policy.model.state_proj.bias.detach().float(),
    }


# ---------------------------------------------------------------------- export


def _export_graph(module, example, path: Path, input_name: str, output_name: str,
                  *, infer_shapes: bool = False) -> Path:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    module.cpu().eval()
    with torch.inference_mode():
        torch.onnx.export(
            module, (example,), str(path), input_names=[input_name],
            output_names=[output_name], opset_version=17,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(path))
    if infer_shapes:
        model = onnx.shape_inference.infer_shapes(onnx.load(path))
        model.ir_version = min(model.ir_version, 8)
        onnx.save(model, path)
    return path


def export_all(output: Path, modules: dict[str, nn.Module], samples: list[dict],
               traces: list[dict], image_keys: Sequence[str]) -> list[Path]:
    del samples, image_keys
    first = traces[0]
    onnx_dir = output / "onnx"
    calibration_dir = output / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    # These are activations at the actual graph boundaries, not synthetic
    # normally-distributed tensors.  Pure BF16 compiles do not fit scales and
    # therefore ignore their values, but split BF16/INT8 graphs require real
    # samples.  Keep every Euler step: the denoiser's range changes over the
    # trajectory and calibrating only t=1 makes later steps drift.
    torch.cat(first["prepared_images"][:2], dim=0).detach().cpu().numpy().astype("<f4").tofile(
        calibration_dir / "vision.f32"
    )
    first["prefix"].unsqueeze(1).detach().cpu().numpy().astype("<f4").tofile(
        calibration_dir / "prefix.f32"
    )
    torch.cat(first["suffix_inputs"], dim=0).detach().cpu().numpy().astype("<f4").tofile(
        calibration_dir / "suffix.f32"
    )
    torch.cat(first["denoise_inputs"], dim=0).detach().cpu().numpy().astype("<f4").tofile(
        calibration_dir / "denoise.f32"
    )
    return [
        _export_graph(modules["vision"], first["prepared_images"][0],
                      onnx_dir / "vision.onnx", "image", "image_tokens"),
        _export_graph(modules["prefix"], first["prefix"].unsqueeze(1),
                      onnx_dir / "prefix.onnx", "prefix_embeddings", "cache"),
        _export_graph(modules["suffix"], first["suffix_inputs"][0],
                      onnx_dir / "suffix.onnx", "suffix_input", "suffix_output"),
        _export_graph(modules["denoise"], first["denoise_inputs"][0],
                      onnx_dir / "denoise.onnx", "denoise_input", "velocity"),
    ]


# ---------------------------------------------------------------- fixtures/data


def write_fixtures(output: Path, sample: dict, traced: dict,
                   image_keys: Sequence[str], postprocessor) -> None:
    del sample, image_keys
    normalized = traced["normalized_action"].detach().cpu()
    raw_steps = [postprocessor(step.unsqueeze(0)).squeeze(0) for step in normalized[0]]
    raw = torch.stack(raw_steps)
    np.savez(
        output / FIXTURE_FILE,
        image0=traced["prepared_images"][0].numpy(),
        image1=traced["prepared_images"][1].numpy(),
        state=traced["state_lane"][..., :traced["state_dim"]].numpy(),
        noise=traced["noise"].numpy(),
        normalized_action=normalized.numpy(),
        action=raw.numpy(),
        prefix_embeddings=traced["prefix"].numpy(),
    )
    constants = output / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    for name in (
        "empty_image_embedding", "language_embedding",
        "state_project_weight", "state_project_bias",
    ):
        np.asarray(traced[name], dtype="<f4").tofile(constants / name)


def write_normalization(checkpoint: str | Path, image_keys: Sequence[str], output: Path) -> Path:
    del image_keys
    from polima.export.normalization import from_lerobot_checkpoint

    arrays = from_lerobot_checkpoint(checkpoint, ())
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    constants = output.parent / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    for name in ("state_mean", "state_std", "action_mean", "action_std"):
        np.asarray(arrays[name], dtype="<f4").tofile(constants / name)
    return output


# ---------------------------------------------------------------- verification


def verify_chain(onnx_dir: Path, fixture_path: Path, report_path: Path,
                 atol: float = 1e-3, rtol: float = 1e-2,
                 stage_dir: Path | None = None) -> dict:
    import onnxruntime as ort

    def session(name):
        return ort.InferenceSession(str(onnx_dir / f"{name}.onnx"),
                                    providers=["CPUExecutionProvider"])

    def dump(name, value):
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            np.asarray(value, dtype="<f4").tofile(stage_dir / f"{name}.f32")

    fixture = np.load(fixture_path)
    vision_session = session("vision")
    vision = [
        vision_session.run(None, {"image": fixture[f"image{i}"].astype(np.float32)})[0]
        for i in range(2)
    ]
    for index, value in enumerate(vision):
        dump(f"vision_{index:02d}_output", value)
    prefix = fixture["prefix_embeddings"].astype(np.float32).copy()
    prefix[:, :64] = vision[0] * IMAGE_SCALE
    prefix[:, 64:128] = vision[1] * IMAGE_SCALE
    cache = session("prefix").run(None, {"prefix_embeddings": prefix[:, None]})[0]
    dump("prefix_00_output", cache)
    actions = fixture["noise"].astype(np.float32).copy()
    suffix_session, denoise_session = session("suffix"), session("denoise")
    for index in range(DENOISE_STEPS):
        suffix_input = _suffix_input(
            torch.from_numpy(actions), 1.0 - index / DENOISE_STEPS
        ).numpy()
        suffix = suffix_session.run(None, {"suffix_input": suffix_input})[0]
        denoise_input = np.zeros((1, 1, DENOISE_TOKENS, DENOISE_WIDTH), np.float32)
        denoise_input[:, :, :1205] = cache.reshape(1, 1, 1205, DENOISE_WIDTH)
        denoise_input[:, :, 1205:].reshape(1, -1)[:, :CHUNK * EXPERT_HIDDEN] = suffix.reshape(1, -1)
        velocity = denoise_session.run(None, {"denoise_input": denoise_input})[0]
        dump(f"suffix_{index:02d}_output", suffix)
        dump(f"denoise_{index:02d}_output", velocity)
        actions -= velocity * (1.0 / DENOISE_STEPS)
    reference = fixture["normalized_action"].astype(np.float32)
    action = actions[:, :, :reference.shape[-1]]
    difference = np.abs(action - reference)
    report = {
        "ok": bool(np.allclose(action, reference, atol=atol, rtol=rtol)),
        "shape": list(action.shape),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "atol": atol,
        "rtol": rtol,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def validate_checkpoint(checkpoint: str | Path) -> None:
    load_policy(checkpoint)


__all__ = [
    "DenoiseExpert",
    "PrefixCache",
    "SuffixProjection",
    "VisionTower",
    "build_modules",
    "export_all",
    "load_policy",
    "trace",
    "validate_checkpoint",
    "verify_chain",
    "write_fixtures",
    "write_normalization",
]
