"""GR00T N1.6's torch side: checkpoint -> 45 ONNX graphs, calibration, fixtures.

Ported from the two scripts that produced the build tree PoLiMa adopts:

    GR00T-N1.6/scripts/export_groot_modalix_eagle.py    the 26-stage Eagle cut
    GR00T-N1.6/scripts/export_groot_modalix_action.py   the 19-graph action cut

This is the only GR00T module that imports torch. It runs in the `groot-n1.6`
conda environment; `polima.policies.groot` and `.runtime` stay importable in the
compiler venv and on the board, which is why the spec names these entry points
as dotted strings rather than holding the callables.

## Why GR00T supplies its own sampler

`polima.export.samples.load` builds calibration observations through
`lerobot.policies.make_pre_post_processors`, which needs a LeRobot policy
object. A GR00T checkpoint is a `transformers` AutoModel with its own
AutoProcessor over a LeRobot *v2.1* dataset, so that path does not apply. The
export driver therefore prefers a graphs module's own `load_samples` when it
defines one; GR00T does, ACT and SmolVLA do not.

## Why the cut lands where it does

Neither half fits the compiler whole. The Eagle backbone is emitted as 26
stages -- patch embedding, 14 SigLIP layer pairs, post-norm, the connector, 8
Qwen layer pairs and a fused output norm -- and the action head as 19: two
projectors, sixteen DiT block pairs, and the decoder tail. Two rewrites in here
are not cosmetic:

* `VisionPostNorm` and `OutputNorm` compute LayerNorm/RMSNorm as matmuls
  against a constant `ones/N` vector rather than calling the reduction
  operators, which Model Compiler cannot place on Modalix.
* `ActionProject` takes a *precomputed* tau encoding instead of a scalar
  timestep, because the sinusoidal path emits scalar Cast/Expand nodes that the
  compiler also rejects. The four encodings are constants of the fixed schedule
  and ship as sidecars.
"""

from __future__ import annotations

import gc
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from polima.policies.groot import runtime as rt

#: Every trained instruction is traced. The ELFs remain shared, but export
#: needs one prompt embedding (and one PyTorch/ONNX check) per catalog entry.
TRACE_FIRST_ONLY = False

DEFAULT_MODEL = "nvidia/GR00T-N1.6-3B"
DEFAULT_REVISION = "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"

# The public GR1 v2.0 datasets encode annotation values in tasks.jsonl too.
# These are not language commands and must never become prompt calibration.
IGNORED_TASK_INSTRUCTIONS = frozenset({"valid"})

#: The GR1 prompt's image-token id, and the vocabulary it indexes. Both are
#: checked rather than assumed: a checkpoint whose tokenizer moved would
#: otherwise scatter vision features over the wrong sequence positions.
IMAGE_TOKEN_ID = 151669
EMBODIMENT_ID = 20


# ------------------------------------------------------------------ modules


class StateProject(nn.Module):
    def __init__(self, head, embodiment_id: int = EMBODIMENT_ID):
        super().__init__()
        self.encoder = head.state_encoder
        self.register_buffer("embodiment", torch.tensor([embodiment_id], dtype=torch.long))

    def forward(self, state):
        return self.encoder(state, self.embodiment)


class ActionProject(nn.Module):
    """The action lane plus this step's tau encoding.

    The four timestep values are fixed by the Euler schedule, so their
    sinusoidal encodings are computed once by the host and arrive as a tensor.
    Computing them here would emit scalar Cast/Expand operators that Model
    Compiler cannot place on Modalix.
    """

    def __init__(self, head, embodiment_id: int = EMBODIMENT_ID):
        super().__init__()
        self.encoder = head.action_encoder
        self.position_embedding = head.position_embedding
        self.register_buffer("embodiment", torch.tensor([embodiment_id], dtype=torch.long))

    def forward(self, actions, tau_embedding):
        action_embedding = self.encoder.W1(actions, self.embodiment)
        features = torch.cat((action_embedding, tau_embedding), dim=-1)
        features = self.encoder.W2(features, self.embodiment)
        features = features * torch.sigmoid(features)
        features = self.encoder.W3(features, self.embodiment)
        positions = torch.arange(rt.CHUNK, dtype=torch.long, device=actions.device)
        return features + self.position_embedding(positions).unsqueeze(0)


class DiTBlockPair(nn.Module):
    """One cross-attention block followed by one self-attention block."""

    def __init__(self, dit, start: int):
        super().__init__()
        if start % 2 or start < 0 or start >= 2 * rt.BLOCK_PAIRS:
            raise ValueError(f"pair must begin at an even block, got {start}")
        self.cross_block = dit.transformer_blocks[start]
        self.self_block = dit.transformer_blocks[start + 1]

    def forward(self, hidden, backbone_features, additive_mask, temb):
        hidden = self.cross_block(
            hidden,
            encoder_hidden_states=backbone_features,
            encoder_attention_mask=additive_mask,
            temb=temb,
        )
        return self.self_block(hidden, temb=temb)


class ActionTail(nn.Module):
    def __init__(self, head, embodiment_id: int = EMBODIMENT_ID):
        super().__init__()
        dit = head.model
        self.norm_out = dit.norm_out
        self.proj_out_1 = dit.proj_out_1
        self.proj_out_2 = dit.proj_out_2
        self.decoder = head.action_decoder
        self.register_buffer("embodiment", torch.tensor([embodiment_id], dtype=torch.long))

    def forward(self, hidden, temb):
        shift, scale = self.proj_out_1(torch.nn.functional.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        hidden = self.proj_out_2(hidden)
        decoded = self.decoder(hidden, self.embodiment)
        return decoded[:, -rt.CHUNK:]


class VisionPatch(nn.Module):
    def __init__(self, embeddings, image_height: int, image_width: int):
        super().__init__()
        self.patch = int(embeddings.patch_size)
        self.projection = embeddings.patch_embedding
        out_channels = self.projection.weight.shape[0]
        with torch.no_grad():
            position = embeddings.position_embedding.weight.reshape(
                embeddings.position_embedding_size,
                embeddings.position_embedding_size,
                out_channels,
            )
            spatial = torch.tensor([[image_height // self.patch, image_width // self.patch]])
            position = embeddings.resize_positional_embeddings(position, spatial)
        self.register_buffer("position", position)

    def forward(self, patches):
        return self.projection(patches) + self.position


class VisionPair(nn.Module):
    def __init__(self, layers, patch_grid: tuple[int, int]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        height, width = patch_grid
        self._meta = [{"img_idx": 0, "patch_hw": (height, width),
                       "win_xy": (0, 0), "win_hw": (height, width)}]
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden, output_attentions=False, rope_freqs_cis=None,
                           win_meta_list=self._meta, windows_attn=False)[0]
        return hidden


class VisionPostNorm(nn.Module):
    """LayerNorm as matmuls.

    `mean` and `var` reductions do not place on Modalix; a matmul against a
    constant `ones/N` column computes the same statistic in an operator the
    compiler does support.
    """

    def __init__(self, norm):
        super().__init__()
        features = int(norm.weight.numel())
        self.register_buffer("weight", norm.weight.detach().reshape(1, 1, features))
        self.register_buffer("bias", norm.bias.detach().reshape(1, 1, features))
        self.register_buffer("ones", torch.ones((features, 1)) / float(features))
        self.epsilon = float(norm.eps)

    def forward(self, hidden):
        mean = torch.matmul(hidden, self.ones)
        centered = hidden - mean
        variance = torch.matmul(centered * centered, self.ones)
        normalized = centered / torch.sqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class VisionConnector(nn.Module):
    def __init__(self, connector):
        super().__init__()
        self.connector = connector

    def forward(self, hidden):
        return self.connector(hidden)


class QwenPair(nn.Module):
    def __init__(self, layers, cos, sin):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, hidden, attention_mask):
        # Public boundaries are rank-4 NHWC so ModelSDK can keep HWC buffers
        # device-resident while accepting a separate task-selected mask.
        hidden = hidden.squeeze(1)
        attention_mask = attention_mask.permute(0, 3, 1, 2)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask,
                           position_embeddings=(self.cos, self.sin),
                           use_cache=False, output_attentions=False)[0]
        return hidden.unsqueeze(1)


class OutputNorm(nn.Module):
    """Qwen's final RMSNorm fused with the action head's input LayerNorm.

    Fusing them saves a whole ELF -- and a round trip -- for two elementwise
    passes over the same 116x2048 tensor. Both reductions are matmuls, for the
    reason VisionPostNorm gives.
    """

    def __init__(self, qwen_norm, action_vlln):
        super().__init__()
        features = int(qwen_norm.weight.numel())
        self.register_buffer("qwen_weight", qwen_norm.weight.detach().reshape(1, 1, features))
        self.register_buffer("action_weight", action_vlln.weight.detach().reshape(1, 1, features))
        self.register_buffer("action_bias", action_vlln.bias.detach().reshape(1, 1, features))
        self.register_buffer("ones", torch.ones((features, 1)) / float(features))
        self.qwen_epsilon = float(qwen_norm.variance_epsilon)
        self.action_epsilon = float(action_vlln.eps)

    def forward(self, hidden):
        rms_variance = torch.matmul(hidden * hidden, self.ones)
        hidden = hidden / torch.sqrt(rms_variance + self.qwen_epsilon)
        hidden = hidden * self.qwen_weight
        mean = torch.matmul(hidden, self.ones)
        centered = hidden - mean
        variance = torch.matmul(centered * centered, self.ones)
        normalized = centered / torch.sqrt(variance + self.action_epsilon)
        return normalized * self.action_weight + self.action_bias


# --------------------------------------------------------------- host helpers


def patchify(pixels: torch.Tensor, patch: int) -> torch.Tensor:
    """(N, 3, H, W) -> (N, H/p * W/p, p*p*3).

    The live client runs the same fold before the wire, which is what keeps an
    image reshape off the board. See `patchify_for_wire`.
    """
    batch, channels, height, width = pixels.shape
    patches = pixels.reshape(batch, channels, height // patch, patch, width // patch, patch)
    return patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch * patch * channels)


def patchify_for_wire(image: np.ndarray) -> np.ndarray:
    """Client-side preprocessor: an HWC uint8 frame -> the wire's patch tensor.

    Named by `RobotSpec.image_preprocessor`, so the robot client resolves it
    without importing torch.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB frame, got {array.shape}")
    height, width = array.shape[:2]
    if height % rt.PATCH or width % rt.PATCH:
        raise ValueError(
            f"GR00T frame dimensions must be divisible by patch size {rt.PATCH}, "
            f"got {array.shape[:2]}"
        )
    patch = rt.PATCH
    rows, columns = height // patch, width // patch
    folded = array.reshape(rows, patch, columns, patch, 3).transpose(0, 2, 1, 3, 4)
    return folded.reshape(rows * columns, rt.PATCH_CHANNELS)


def pixel_unshuffle_tokens(
    hidden: torch.Tensor, grid: int, factor: int, width: int | None = None
) -> torch.Tensor:
    """Fold an HxW token grid exactly as the runtime pixel-unshuffle opcode."""
    height = grid
    width = height if width is None else width
    batch, _, channels = hidden.shape
    folded = hidden.transpose(1, 2).reshape(batch, channels, height, width)
    folded = torch.nn.functional.pixel_unshuffle(folded, factor)
    return folded.flatten(2).transpose(1, 2).contiguous()


def additive_masks(image_mask: torch.Tensor, attention_mask: torch.Tensor):
    """The two attention masks the DiT alternates between, as additive biases."""
    image_valid = image_mask & attention_mask
    text_valid = (~image_mask) & attention_mask
    zero = torch.zeros((), dtype=torch.float32, device=image_mask.device)
    blocked = torch.full((), -10000.0, dtype=torch.float32, device=image_mask.device)
    return (torch.where(image_valid, zero, blocked),
            torch.where(text_valid, zero, blocked))


def qwen_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Four-dimensional causal mask with padded key positions blocked."""

    sequence = int(attention_mask.shape[-1])
    causal = torch.tril(
        torch.ones((sequence, sequence), dtype=torch.bool, device=attention_mask.device)
    )
    valid = causal.unsqueeze(0).unsqueeze(0) & attention_mask.bool()[:, None, None, :]
    zero = torch.zeros((), dtype=torch.float32, device=attention_mask.device)
    blocked = torch.full((), -10000.0, dtype=torch.float32, device=attention_mask.device)
    return torch.where(valid, zero, blocked)


def image_token_starts(
    input_ids: torch.Tensor,
    image_count: int,
    tokens_per_image: int = rt.CONNECTOR_WIDTH,
) -> tuple[int, ...]:
    """Validate and locate one contiguous token window per Eagle image."""
    positions = np.flatnonzero((input_ids[0] == IMAGE_TOKEN_ID).detach().cpu().numpy())
    expected = image_count * tokens_per_image
    if len(positions) != expected:
        raise ValueError(
            f"GR00T prompt needs {expected} image tokens for {image_count} images, "
            f"found {len(positions)}"
        )
    chunks = tuple(
        positions[index:index + tokens_per_image]
        for index in range(0, expected, tokens_per_image)
    )
    starts = tuple(int(chunk[0]) for chunk in chunks)
    if any(not np.array_equal(chunk, np.arange(start, start + tokens_per_image))
           for start, chunk in zip(starts, chunks, strict=True)):
        raise ValueError(f"GR00T image-token windows are not contiguous: starts={starts}")
    return starts


#: `dynamo` only exists from torch 2.5. GR00T exports under 2.7, where the flag
#: is needed -- afe's importer is validated against opset 17 and the legacy
#: tracer, and the dynamo exporter emits a different (valid, unsupported) graph.
#: Older torch has no dynamo path at all, so omitting it there is the same
#: request, and it keeps this module runnable in PoLiMa's own host venv.
_EXPORT_TAKES_DYNAMO = "dynamo" in inspect.signature(torch.onnx.export).parameters


def _export_onnx(module: nn.Module, inputs: tuple, path: Path, names, outputs) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.cpu().float().eval()
    cpu_inputs = tuple(item.detach().cpu().float() for item in inputs)
    extra = {"dynamo": False} if _EXPORT_TAKES_DYNAMO else {}
    with torch.inference_mode():
        expected = module(*cpu_inputs)
        torch.onnx.export(module, cpu_inputs, str(path), input_names=list(names),
                          output_names=list(outputs), opset_version=17,
                          do_constant_folding=True, **extra)
    return expected.detach().cpu().numpy().astype(np.float32)


def _save_calibration(directory: Path, name: str, **tensors) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{name}.npz",
             **{key: np.asarray(value, dtype=np.float32) for key, value in tensors.items()})


# ------------------------------------------------------ the driver's contract


def load_policy(
    checkpoint: str | Path,
    lerobot_src: str | Path | None = None,
    profile: Mapping[str, Any] | None = None,
):
    """(model, image_keys) for a GR00T checkpoint.

    `lerobot_src` is accepted and ignored: GR00T loads through `transformers`,
    not through a LeRobot source tree. The signature matches ACT's and
    SmolVLA's so `polima.export.driver` needs no branch.
    """
    # Importing registers GR00T's custom AutoModel/AutoProcessor classes.
    import gr00t.model  # noqa: F401
    from transformers import AutoModel

    from polima.policies.groot.profile import checkpoint_metadata

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GR00T export requires CUDA for the checkpoint BF16 FlashAttention "
            "reference pass, but torch.cuda.is_available() is false"
        )
    # Eagle-Block2A hard-wires FlashAttention for both its vision and language
    # encoders. Loading on CPU succeeds, but its first attention call cannot;
    # keep the reference pass on CUDA. Individual export modules are moved to
    # CPU/FP32 by _export_onnx after every task reference has been collected.
    model = AutoModel.from_pretrained(str(checkpoint)).eval().cuda()
    embodiment_tag = (profile or {}).get("embodiment_tag")
    return model, checkpoint_metadata(
        checkpoint, embodiment_tag=embodiment_tag
    )["camera_keys"]


def load_samples(policy, checkpoint: str | Path, observation_keys: Sequence[str],
                 dataset_root: str | Path | None = None, count: int = 8,
                 lerobot_src: str | Path | None = None,
                 profile: Mapping[str, Any] | None = None):
    """GR00T's own calibration draw, in place of `polima.export.samples.load`.

    Returns the same `(samples, postprocessor, root)` triple the generic path
    does. The postprocessor is None: GR00T denormalizes on the board from the
    statistics `write_normalization` extracts, so nothing downstream needs one.
    """
    import gr00t.model  # noqa: F401 - registers custom AutoProcessor classes
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from transformers import AutoProcessor

    from polima.policies.groot.profile import checkpoint_metadata


    root = Path(dataset_root) if dataset_root else _dataset_from_checkpoint(checkpoint)
    embodiment_tag = (profile or {}).get("embodiment_tag")
    if root.is_file():
        return _load_manifest_samples(
            policy, checkpoint, root, count=count, embodiment_tag=embodiment_tag
        )
    metadata = checkpoint_metadata(checkpoint, root, embodiment_tag)
    info_path = root / "meta" / "info.json"
    modality_path = root / "meta" / "modality.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"GR00T calibration dataset has no valid {info_path}") from error
    if str(info.get("codebase_version")) not in {"v2.0", "v2.1"}:
        raise ValueError(
            f"GR00T calibration dataset must be LeRobot v2.0 or v2.1, got "
            f"{info.get('codebase_version')!r} at {root}"
        )
    if not modality_path.is_file():
        raise FileNotFoundError(
            f"GR00T calibration dataset is missing {modality_path}; copy the "
            "robot-specific modality JSON after v3-to-v2.1 conversion"
        )
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    processor.eval()
    tag = EmbodimentTag(metadata["embodiment_tag"])
    modalities = processor.get_modality_configs()[tag.value]
    # The indexed-frame loader does not implement pyav; torchcodec is the
    # native GR00T backend and is validated by the environment installer.
    dataset = LeRobotEpisodeLoader(
        dataset_path=root,
        modality_configs=modalities,
        video_backend="torchcodec",
    )
    catalog = _task_catalog(dataset)
    episode_tasks = _episode_task_ids(dataset, catalog)

    # A calibration count is a minimum for language-conditioned policies: one
    # representative per trained instruction is mandatory. Remaining slots
    # are spread over the recording instead of taking adjacent episodes.
    selected: list[tuple[int, int]] = []
    for task in catalog:
        matches = [
            episode
            for episode, task_ids in episode_tasks.items()
            if task["dataset_id"] in task_ids
        ]
        if not matches:
            raise ValueError(
                f"GR00T dataset task {task['dataset_id']} ({task['instruction']!r}) "
                "has no episode"
            )
        selected.append((matches[len(matches) // 2], int(task["id"])))

    target = min(max(int(count), len(catalog)), len(dataset))
    evenly_spaced = np.linspace(0, len(dataset) - 1, num=target, dtype=int)
    selected_episodes = {episode for episode, _ in selected}
    for episode in evenly_spaced:
        episode = int(episode)
        if len(selected) >= target:
            break
        if episode in selected_episodes:
            continue
        task_ids = episode_tasks.get(episode, ())
        if not task_ids:
            continue
        runtime_id = next(
            int(task["id"]) for task in catalog if task["dataset_id"] == task_ids[0]
        )
        selected.append((episode, runtime_id))
        selected_episodes.add(episode)

    samples = []
    for episode, task_id in selected:
        trajectory = dataset[episode]
        step = extract_step_data(
            trajectory, 0, modality_configs=modalities, embodiment_tag=tag
        )
        # GR00T's loader may randomly choose among an episode's task labels.
        # Catalog representatives must be exact and deterministic.
        instruction = str(catalog[task_id]["instruction"])
        step.text = instruction
        message = {"type": MessageType.EPISODE_STEP.value, "content": step}
        raw_state = np.concatenate([
            np.asarray(step.states[key], dtype=np.float32).reshape(-1)
            for key in modalities["state"].modality_keys
        ])
        inputs = processor.collator([processor([message])])["inputs"]
        inputs["_polima_raw_state"] = raw_state
        inputs["_polima_task_id"] = task_id
        inputs["_polima_task"] = instruction
        inputs["_polima_task_catalog"] = catalog
        samples.append(inputs)
    _left_pad_prompt_samples(samples, processor.processor.tokenizer.pad_token_id)
    return samples, None, root


def _left_pad_prompt_samples(samples, pad_token_id: int) -> None:
    """Give every language task one static sequence and image-token window.

    Left padding shifts every real token by one constant offset, preserving
    RoPE-relative distances. Padding keys are excluded by attention_mask, and
    the action head also masks them from cross-attention.
    """

    if pad_token_id is None:
        raise ValueError("GR00T tokenizer has no pad token id")
    lengths = {int(sample["input_ids"].shape[-1]) for sample in samples}
    if not lengths:
        raise ValueError("GR00T export received no processor samples")
    target = max(lengths)
    for sample in samples:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        padding = target - int(input_ids.shape[-1])
        if padding <= 0:
            continue
        pad_ids = torch.full(
            (*input_ids.shape[:-1], padding),
            int(pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        pad_mask = torch.zeros(
            (*attention_mask.shape[:-1], padding),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        sample["input_ids"] = torch.cat((pad_ids, input_ids), dim=-1)
        sample["attention_mask"] = torch.cat((pad_mask, attention_mask), dim=-1)


def _load_manifest_samples(
    policy,
    checkpoint: str | Path,
    manifest_path: Path,
    *,
    count: int,
    embodiment_tag: str | None,
    split: str = "calibration",
):
    """Load an explicit split, keeping task IDs fixed by the calibration catalog."""

    import gr00t.model  # noqa: F401 - registers custom AutoProcessor classes
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from transformers import AutoProcessor

    from polima.policies.groot.calibration import load_manifest
    from polima.policies.groot.profile import checkpoint_metadata

    manifest = load_manifest(manifest_path)
    if embodiment_tag and manifest.embodiment_tag != embodiment_tag:
        raise ValueError(
            f"GR00T manifest embodiment {manifest.embodiment_tag!r} does not match "
            f"the selected checkpoint embodiment {embodiment_tag!r}"
        )
    selected = manifest.select(split)
    if int(count) != len(selected):
        raise ValueError(
            f"GR00T sparse manifest declares {len(selected)} {split} observations; "
            f"pass --calibration-samples {len(selected)}, not {count}"
        )

    processor = AutoProcessor.from_pretrained(str(checkpoint))
    processor.eval()
    tag = EmbodimentTag(manifest.embodiment_tag)
    modalities = processor.get_modality_configs()[tag.value]

    datasets = {}
    for observation in selected:
        if observation.dataset in datasets:
            continue
        checkpoint_metadata(
            checkpoint, observation.root, embodiment_tag=manifest.embodiment_tag
        )
        datasets[observation.dataset] = LeRobotEpisodeLoader(
            dataset_path=observation.root,
            modality_configs=modalities,
            video_backend="torchcodec",
        )

    instructions = list(dict.fromkeys(item.instruction for item in manifest.select("calibration")))
    if any(item.instruction not in instructions for item in selected):
        raise ValueError("evaluation instruction is absent from the compiled task catalog")
    catalog = [
        {"id": index, "dataset_id": index, "instruction": instruction}
        for index, instruction in enumerate(instructions)
    ]
    task_ids = {task["instruction"]: int(task["id"]) for task in catalog}

    samples = []
    trajectories = {}
    for observation in selected:
        identity = (observation.dataset, observation.episode_index)
        if identity not in trajectories:
            trajectories[identity] = datasets[observation.dataset][
                observation.episode_index
            ]
        step = extract_step_data(
            trajectories[identity],
            observation.frame_index,
            modality_configs=modalities,
            embodiment_tag=tag,
        )
        step.text = observation.instruction
        message = {"type": MessageType.EPISODE_STEP.value, "content": step}
        raw_state = np.concatenate([
            np.asarray(step.states[key], dtype=np.float32).reshape(-1)
            for key in modalities["state"].modality_keys
        ])
        inputs = processor.collator([processor([message])])["inputs"]
        inputs["_polima_raw_state"] = raw_state
        inputs["_polima_task_id"] = task_ids[observation.instruction]
        inputs["_polima_task"] = observation.instruction
        inputs["_polima_task_catalog"] = catalog
        inputs["_polima_dataset"] = observation.dataset
        inputs["_polima_episode_index"] = observation.episode_index
        inputs["_polima_frame_index"] = observation.frame_index
        samples.append(inputs)
    _left_pad_prompt_samples(samples, processor.processor.tokenizer.pad_token_id)
    return samples, None, manifest.path


def _task_catalog(dataset) -> list[dict[str, Any]]:
    """Stable runtime ids for every instruction declared by the dataset."""
    raw = getattr(dataset, "tasks_map", None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("GR00T calibration dataset has no task catalog")

    def dataset_id(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    ordered = sorted(
        raw.items(),
        key=lambda item: (str(type(dataset_id(item[0]))), dataset_id(item[0])),
    )
    filtered = [
        (raw_id, str(raw_instruction).strip())
        for raw_id, raw_instruction in ordered
        if str(raw_instruction).strip().lower() not in IGNORED_TASK_INSTRUCTIONS
    ]
    catalog = []
    instructions: set[str] = set()
    for runtime_id, (raw_id, instruction) in enumerate(filtered):
        if not instruction:
            raise ValueError(f"GR00T dataset task {raw_id!r} has an empty instruction")
        if instruction in instructions:
            raise ValueError(f"duplicate GR00T task instruction {instruction!r}")
        instructions.add(instruction)
        catalog.append({
            "id": runtime_id,
            "dataset_id": dataset_id(raw_id),
            "instruction": instruction,
        })
    return catalog


def _episode_task_ids(
    dataset, catalog: Sequence[dict[str, Any]]
) -> dict[int, tuple[Any, ...]]:
    """Normalize GR00T loader metadata into episode -> dataset task ids."""
    known_by_text = {
        str(task["instruction"]): task["dataset_id"] for task in catalog
    }
    known_ids = {str(task["dataset_id"]): task["dataset_id"] for task in catalog}
    metadata = getattr(dataset, "episodes_metadata", None)
    if not isinstance(metadata, Sequence):
        raise TypeError("GR00T calibration dataset has no episode task metadata")
    result = {}
    for episode, entry in enumerate(metadata):
        values = entry.get("tasks", ()) if isinstance(entry, dict) else ()
        normalized = []
        for value in values:
            key = str(value)
            if key in known_ids:
                normalized.append(known_ids[key])
            elif key in known_by_text:
                normalized.append(known_by_text[key])
        result[episode] = tuple(normalized)
    return result


def _dataset_from_checkpoint(checkpoint: str | Path) -> Path:
    """The dataset a GR00T run trained on, from its own experiment config."""
    for name in ("experiment_cfg/metadata.json", "train_config.json", "config.json"):
        path = Path(checkpoint) / name
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset_path", "dataset_root"):
            if config.get(key):
                return Path(config[key]).resolve()
        dataset = config.get("dataset") or {}
        if dataset.get("root"):
            return Path(dataset["root"]).resolve()
    raise FileNotFoundError(
        f"cannot infer the dataset for {checkpoint}; pass --dataset-root explicitly"
    )


def build_modules(policy) -> dict[str, Any]:
    """Split the checkpoint into the pieces every later stage indexes by name."""
    backbone = policy.backbone.model
    return {
        "model": policy,
        "head": policy.action_head,
        "eagle": backbone,
        "vision": backbone.vision_model.vision_model,
        "language": backbone.language_model.model,
        "connector": backbone.mlp1,
        "vlln": policy.action_head.vlln,
    }


def _trace_action_prefix(head, backbone, action_inputs):
    """Project reference inputs without leaking inference tensors to autograd."""
    # prepare_input and the backbone reference pass run under inference mode.
    # Their outputs therefore remain inference tensors after that context exits.
    # The embodiment-conditioned layers index trainable weights with embodiment;
    # if autograd is re-enabled, PyTorch tries to save that inference tensor for
    # backward and raises. This is a reference-only pass, so keep both action
    # projections in inference mode as well.
    with torch.inference_mode():
        backbone_features = head.vlln(backbone.backbone_features.float()).float()
        state = action_inputs.state.float()
        embodiment = action_inputs.embodiment_id.long().reshape(-1)
        if embodiment.numel() != 1:
            raise ValueError(
                f"GR00T export requires one embodiment id, got {embodiment.tolist()}"
            )
        state_features = head.state_encoder(state, embodiment)
    return backbone_features, state, embodiment, state_features


def trace(modules: dict, sample, image_keys) -> dict:
    """One reference pass, recording every tensor a later stage needs.

    The four denoise steps are traced together because each graph's calibration
    set is exactly those four activations -- tau is bucketed at 0/250/500/750
    and the pipeline never sees anything else.
    """
    model, head = modules["model"], modules["head"]
    device = model.device
    metadata = {
        key: value for key, value in sample.items() if key.startswith("_polima_")
    }
    model_sample = {
        key: value for key, value in sample.items() if not key.startswith("_polima_")
    }
    # The fine-tuned top Qwen layers are deliberately stored in FP32 while
    # Eagle attention is FlashAttention, which only accepts FP16/BF16. GR00T
    # training supplies this cast through CUDA autocast; reproduce that exact
    # execution mode for the reference pass.
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        backbone_inputs, action_inputs = model.prepare_input(model_sample)
        backbone = model.backbone(backbone_inputs)

    sequence = int(backbone_inputs["input_ids"].shape[1])
    image_shapes = tuple(
        (int(value.shape[-2]), int(value.shape[-1]))
        for value in backbone_inputs["pixel_values"]
    )
    if len(set(image_shapes)) != 1:
        raise ValueError(
            "GR00T shared vision ELF requires every camera to have one processed "
            f"shape, got {image_shapes}"
        )
    image_height, image_width = image_shapes[0]
    if image_height % rt.PATCH or image_width % rt.PATCH:
        raise ValueError(
            f"GR00T processed image {image_shapes[0]} is not divisible by patch "
            f"size {rt.PATCH}"
        )
    patch_grid = (image_height // rt.PATCH, image_width // rt.PATCH)
    if any(value % rt.UNSHUFFLE for value in patch_grid):
        raise ValueError(
            f"GR00T patch grid {patch_grid} is not divisible by pixel-unshuffle "
            f"factor {rt.UNSHUFFLE}"
        )
    connector_width = (
        patch_grid[0] // rt.UNSHUFFLE
    ) * (
        patch_grid[1] // rt.UNSHUFFLE
    )
    starts = image_token_starts(
        backbone_inputs["input_ids"],
        len(backbone_inputs["pixel_values"]),
        connector_width,
    )

    # The action head (including vlln) is stored in FP32 in this checkpoint.
    backbone_features, state, embodiment, state_features = _trace_action_prefix(
        head, backbone, action_inputs
    )
    generator = torch.Generator(device=device).manual_seed(123)
    actions = torch.randn((1, rt.CHUNK, rt.ACTION_LANE), generator=generator,
                          dtype=torch.float32, device=device)
    image_additive, text_additive = additive_masks(
        backbone.image_mask, backbone.backbone_attention_mask
    )

    steps = []
    with torch.inference_mode():
        for index in range(rt.DENOISE_STEPS):
            timestep = torch.tensor(
                [rt.TIMESTEP_BUCKETS[index]], dtype=torch.float32, device=device
            )
            action_features = head.action_encoder(actions, timestep.long(), embodiment)
            positions = torch.arange(rt.CHUNK, device=device)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
            hidden = torch.cat((state_features, action_features), dim=1)
            temb = head.model.timestep_encoder(timestep.long())
            pair_inputs = []
            for start in range(0, 2 * rt.BLOCK_PAIRS, 2):
                mask = text_additive if start % 4 == 0 else image_additive
                pair_inputs.append(hidden.detach().cpu().numpy())
                hidden = DiTBlockPair(head.model, start)(
                    hidden, backbone_features, mask, temb)
            velocity = ActionTail(head, int(embodiment.item()))(hidden, temb)
            actions = actions + np.float32(rt.DT) * velocity
            steps.append({
                "timestep": timestep.detach().cpu().numpy(),
                "temb": temb.detach().cpu().numpy(),
                "tau": head.action_encoder.pos_encoding(
                    torch.full(
                        (1, rt.CHUNK),
                        float(rt.TIMESTEP_BUCKETS[index]),
                        device=device,
                    )
                ).detach().cpu().numpy(),
                "pair_inputs": pair_inputs,
                "tail_input": hidden.detach().cpu().numpy(),
                "velocity": velocity.detach().cpu().numpy(),
                "actions": actions.detach().cpu().numpy(),
            })

    # The prompt's token embeddings, recorded here rather than during the Eagle
    # export so that `write_fixtures` does not need the language model again.
    input_ids = backbone_inputs["input_ids"]
    with torch.inference_mode():
        prompt_embedding = modules["language"].embed_tokens(input_ids).float()

    return {
        "task_id": int(metadata.get("_polima_task_id", 0)),
        "task": str(metadata.get("_polima_task", "")),
        "task_catalog": list(metadata.get("_polima_task_catalog", ())),
        "sequence": sequence,
        "image_shape": image_shapes[0],
        "patch_grid": patch_grid,
        "vision_width": patch_grid[0] * patch_grid[1],
        "connector_width": connector_width,
        "image_token_starts": starts,
        "embodiment_id": int(embodiment.item()),
        "backbone_inputs": backbone_inputs,
        "prompt_embedding": prompt_embedding.detach().cpu().numpy(),
        "state": state.detach().cpu().numpy(),
        "raw_state": metadata.get("_polima_raw_state"),
        "backbone_features": backbone_features.detach().cpu().numpy(),
        "image_additive_mask": image_additive.detach().cpu().numpy(),
        "text_additive_mask": text_additive.detach().cpu().numpy(),
        "qwen_attention_mask": qwen_attention_mask(
            backbone.backbone_attention_mask
        ).detach().cpu().numpy(),
        # The seed the first step started from, recovered by undoing its update.
        "noise": steps[0]["actions"] - np.float32(rt.DT) * steps[0]["velocity"],
        "final_action": steps[-1]["actions"],
        "steps": steps,
    }



def specialize_spec_from_trace(spec, reference: dict, image_keys):
    """Bind prompt sequence/image windows discovered by the real processor."""
    from polima.policies.groot.profile import specialize_geometry

    return specialize_geometry(spec, reference)

def _catalog_references(traces: Sequence[dict]) -> list[dict]:
    """One validated trace per runtime task id, in catalog order."""
    if not traces:
        raise ValueError("GR00T export produced no traces")
    catalog = list(traces[0].get("task_catalog") or ())
    if not catalog:
        raise ValueError("GR00T export trace has no task catalog")
    expected_ids = list(range(len(catalog)))
    if [int(task.get("id", -1)) for task in catalog] != expected_ids:
        raise ValueError(f"GR00T runtime task ids must be contiguous: {catalog}")

    representatives: dict[int, dict] = {}
    for trace_value in traces:
        if list(trace_value.get("task_catalog") or ()) != catalog:
            raise ValueError("GR00T task catalog changed between calibration samples")
        task_id = int(trace_value.get("task_id", -1))
        if task_id in expected_ids:
            representatives.setdefault(task_id, trace_value)
    missing = [task_id for task_id in expected_ids if task_id not in representatives]
    if missing:
        raise ValueError(f"GR00T export has no calibration trace for task id(s) {missing}")

    references = [representatives[task_id] for task_id in expected_ids]
    first = references[0]
    geometry = (
        int(first["sequence"]),
        tuple(first["image_token_starts"]),
        tuple(first["image_shape"]),
        tuple(first["patch_grid"]),
        int(first["connector_width"]),
    )
    for reference in references[1:]:
        candidate = (
            int(reference["sequence"]),
            tuple(reference["image_token_starts"]),
            tuple(reference["image_shape"]),
            tuple(reference["patch_grid"]),
            int(reference["connector_width"]),
        )
        if candidate != geometry:
            raise ValueError(
                "GR00T task prompts do not share one processor layout: "
                f"task 0 has sequence/windows {geometry}, task {reference['task_id']} "
                f"has {candidate}. One fixed-shape ONNX graph set requires shared geometry."
            )
        if not np.array_equal(
            reference["image_additive_mask"], first["image_additive_mask"]
        ):
            raise ValueError(
                f"GR00T task {reference['task_id']} has a different image mask"
            )
        for mask_name in ("text_additive_mask", "qwen_attention_mask"):
            if np.asarray(reference[mask_name]).shape != np.asarray(first[mask_name]).shape:
                raise ValueError(
                    f"GR00T task {reference['task_id']} has an invalid {mask_name} shape"
                )
    return references


def export_all(build_dir: Path, modules: dict, samples, traces, image_keys) -> list[Path]:
    """Write ONNX, all calibration traces, and one fixture per trained task."""
    build_dir = Path(build_dir)
    onnx_dir, calibration = build_dir / "onnx", build_dir / "calibration"
    references = _catalog_references(traces)
    # write_fixtures is called later with traces[0]; preserve the validated
    # catalog set there so it can emit every prompt and task fixture.
    traces[0]["_task_references"] = references
    written = _export_eagle(
        onnx_dir, calibration, modules, references, calibration_traces=traces
    )
    written += _export_action(
        onnx_dir, calibration, modules, references, calibration_traces=traces
    )
    return written


def _export_eagle(
    onnx_dir: Path,
    calibration: Path,
    modules: dict,
    references: Sequence[dict],
    *,
    calibration_traces: Sequence[dict] | None = None,
) -> list[Path]:
    vision = modules["vision"]
    language = modules["language"]
    first_reference = references[0]
    camera_count = len(first_reference["backbone_inputs"]["pixel_values"])
    patch_grid = tuple(int(value) for value in first_reference["patch_grid"])
    vision_width = int(first_reference["vision_width"])
    connector_width = int(first_reference["connector_width"])
    calibration_traces = list(calibration_traces or references)
    if any(len(reference["backbone_inputs"]["pixel_values"]) != camera_count
           for reference in calibration_traces):
        raise ValueError("GR00T camera count changed between calibration traces")

    pixels = [
        value.cpu().float()
        for reference in calibration_traces
        for value in reference["backbone_inputs"]["pixel_values"]
    ]
    written: list[Path] = []

    def export_shared(module, values, path, names, outputs):
        """Export one shared graph and evaluate every balanced calibration value."""
        first = torch.from_numpy(_export_onnx(module, (values[0],), path, names, outputs))
        result = [first]
        module = module.cpu().float().eval()
        with torch.inference_mode():
            result.extend(module(value.cpu().float()).detach().cpu() for value in values[1:])
        return result

    patch_module = VisionPatch(
        vision.embeddings, int(pixels[0].shape[-2]), int(pixels[0].shape[-1])
    )
    patches = [patchify(value, patch_module.patch) for value in pixels]
    _save_calibration(
        calibration, "eagle_vision_patch",
        patches=np.stack([value.numpy() for value in patches], axis=0),
    )
    path = onnx_dir / "eagle_vision_patch.onnx"
    hidden = export_shared(patch_module, patches, path, ("patches",), ("output",))
    written.append(path)

    layers = list(vision.encoder.layers)
    for start in range(0, len(layers), 2):
        end = min(start + 2, len(layers))
        name = f"eagle_vision_{start:02d}_{end - 1:02d}"
        _save_calibration(
            calibration, name,
            hidden=np.stack([value.numpy() for value in hidden], axis=0),
        )
        path = onnx_dir / f"{name}.onnx"
        hidden = export_shared(
            VisionPair(layers[start:end], patch_grid), hidden, path,
            ("hidden",), ("output",),
        )
        written.append(path)

    _save_calibration(
        calibration, "eagle_vision_post_norm",
        hidden=np.stack([value.numpy() for value in hidden], axis=0),
    )
    path = onnx_dir / "eagle_vision_post_norm.onnx"
    hidden = export_shared(
        VisionPostNorm(vision.post_layernorm), hidden, path,
        ("hidden",), ("output",),
    )
    written.append(path)

    if any(int(value.shape[1]) != vision_width for value in hidden):
        raise ValueError(
            f"GR00T vision export expected {vision_width} patch tokens for grid "
            f"{patch_grid}"
        )
    folded = [
        pixel_unshuffle_tokens(
            value, patch_grid[0], rt.UNSHUFFLE, width=patch_grid[1]
        )
        for value in hidden
    ]
    _save_calibration(
        calibration, "eagle_vision_connector",
        hidden=np.stack([value.numpy() for value in folded], axis=0),
    )
    path = onnx_dir / "eagle_vision_connector.onnx"
    image_tokens = export_shared(
        VisionConnector(modules["connector"]), folded, path,
        ("hidden",), ("output",),
    )
    written.append(path)

    # Vision export above produces CPU tensors, and every Qwen pair is exported
    # on CPU below. Move the shared language module before constructing prompt
    # embeddings so its embedding/rotary buffers and their inputs cannot straddle
    # CUDA and CPU.
    language = language.cpu().float().eval()
    hidden_values = []
    rotary = []
    for reference_index, reference in enumerate(calibration_traces):
        backbone_inputs = reference["backbone_inputs"]
        input_ids = backbone_inputs["input_ids"].cpu()
        image_positions = input_ids == IMAGE_TOKEN_ID
        expected_tokens = camera_count * connector_width
        if int(image_positions.sum()) != expected_tokens:
            raise ValueError(
                f"task {reference['task_id']} expected {expected_tokens} image tokens, "
                f"got {int(image_positions.sum())}"
            )
        with torch.inference_mode():
            token_embeddings = language.embed_tokens(input_ids).float()
            begin = reference_index * camera_count
            joined = torch.cat(image_tokens[begin:begin + camera_count], dim=1)
            token_embeddings[image_positions] = joined.reshape(-1, joined.shape[-1])
            positions = torch.arange(
                int(reference["sequence"]), dtype=torch.long
            ).unsqueeze(0)
            rotary.append(language.rotary_emb(token_embeddings, positions))
        hidden_values.append(token_embeddings)

    cos, sin = rotary[0]
    for task_id, (other_cos, other_sin) in enumerate(rotary[1:], start=1):
        if not torch.equal(cos, other_cos) or not torch.equal(sin, other_sin):
            raise ValueError(f"GR00T rotary constants differ for task {task_id}")

    attention_masks = [
        torch.from_numpy(reference["qwen_attention_mask"]).permute(0, 2, 3, 1)
        for reference in calibration_traces
    ]
    layers = list(language.layers)
    for start in range(0, len(layers), 2):
        end = min(start + 2, len(layers))
        name = f"eagle_qwen_{start:02d}_{end - 1:02d}"
        hidden_inputs = [value.unsqueeze(1) for value in hidden_values]
        _save_calibration(
            calibration, name,
            hidden=np.stack([value.detach().cpu().numpy() for value in hidden_inputs]),
            attention_mask=np.stack([
                value.detach().cpu().numpy() for value in attention_masks
            ]),
        )
        path = onnx_dir / f"{name}.onnx"
        module = QwenPair(layers[start:end], cos, sin).cpu().float().eval()
        first = torch.from_numpy(_export_onnx(
            module,
            (hidden_inputs[0], attention_masks[0]),
            path,
            ("hidden", "attention_mask"),
            ("output",),
        )).squeeze(1)
        next_hidden = [first]
        with torch.inference_mode():
            next_hidden.extend(
                module(hidden.cpu().float(), mask.cpu().float()).squeeze(1).detach().cpu()
                for hidden, mask in zip(
                    hidden_inputs[1:], attention_masks[1:], strict=True
                )
            )
        hidden_values = next_hidden
        written.append(path)

    _save_calibration(
        calibration, "eagle_output_norm",
        hidden=np.stack([value.detach().cpu().numpy() for value in hidden_values]),
    )
    path = onnx_dir / "eagle_output_norm.onnx"
    _export_onnx(
        OutputNorm(language.norm, modules["vlln"]),
        (hidden_values[0],),
        path,
        ("hidden",),
        ("output",),
    )
    written.append(path)
    return written


def _export_action(
    onnx_dir: Path,
    calibration: Path,
    modules: dict,
    references: Sequence[dict],
    *,
    calibration_traces: Sequence[dict] | None = None,
) -> list[Path]:
    head = modules["head"].cpu().float()
    first = references[0]
    written: list[Path] = []
    embodiment_id = int(first["embodiment_id"])
    calibration_traces = list(calibration_traces or references)
    if any(
        int(reference["embodiment_id"]) != embodiment_id
        for reference in calibration_traces
    ):
        raise ValueError("GR00T embodiment id changed between calibration traces")

    state = torch.from_numpy(first["state"])
    noise = torch.from_numpy(first["noise"])
    tau = torch.from_numpy(first["steps"][0]["tau"])

    _save_calibration(
        calibration, "state_project",
        state=np.stack([reference["state"] for reference in calibration_traces]),
    )
    path = onnx_dir / "state_project.onnx"
    _export_onnx(
        StateProject(head, embodiment_id), (state,), path,
        ("state",), ("state_features",),
    )
    written.append(path)

    action_inputs = []
    tau_inputs = []
    for reference in calibration_traces:
        steps = reference["steps"]
        action_inputs.extend([reference["noise"], *(step["actions"] for step in steps[:-1])])
        tau_inputs.extend(step["tau"] for step in steps)
    _save_calibration(
        calibration, "action_project",
        actions=np.stack(action_inputs),
        tau_embedding=np.stack(tau_inputs),
    )
    path = onnx_dir / "action_project.onnx"
    _export_onnx(
        ActionProject(head, embodiment_id), (noise, tau), path,
        ("actions", "tau_embedding"), ("action_features",),
    )
    written.append(path)

    first_steps = first["steps"]
    for pair, start in enumerate(range(0, 2 * rt.BLOCK_PAIRS, 2)):
        name = f"dit_blocks_{start:02d}_{start + 1:02d}"
        pair_rows = [
            (reference, step)
            for reference in calibration_traces
            for step in reference["steps"]
        ]
        mask_name = "text_additive_mask" if pair % 2 == 0 else "image_additive_mask"
        _save_calibration(
            calibration, name,
            hidden=np.stack([step["pair_inputs"][pair] for _, step in pair_rows]),
            backbone_features=np.stack(
                [reference["backbone_features"] for reference, _ in pair_rows]
            ),
            additive_mask=np.stack(
                [reference[mask_name] for reference, _ in pair_rows]
            ),
            temb=np.stack([step["temb"] for _, step in pair_rows]),
        )
        path = onnx_dir / f"{name}.onnx"
        _export_onnx(
            DiTBlockPair(head.model, start),
            (
                torch.from_numpy(first_steps[0]["pair_inputs"][pair]),
                torch.from_numpy(first["backbone_features"]),
                torch.from_numpy(first[mask_name]),
                torch.from_numpy(first_steps[0]["temb"]),
            ),
            path,
            ("hidden", "backbone_features", "additive_mask", "temb"),
            ("hidden_out",),
        )
        written.append(path)
        gc.collect()

    _save_calibration(
        calibration, "action_tail",
        hidden=np.stack([
            step["tail_input"]
            for reference in calibration_traces
            for step in reference["steps"]
        ]),
        temb=np.stack([
            step["temb"]
            for reference in calibration_traces
            for step in reference["steps"]
        ]),
    )
    path = onnx_dir / "action_tail.onnx"
    _export_onnx(
        ActionTail(head, embodiment_id),
        (
            torch.from_numpy(first_steps[0]["tail_input"]),
            torch.from_numpy(first_steps[0]["temb"]),
        ),
        path,
        ("hidden", "temb"),
        ("velocity",),
    )
    written.append(path)
    return written
def write_fixtures(
    build_dir: Path, sample, reference: dict, image_keys, postprocessor
) -> Path:
    """Write indexed prompt sidecars and one reference fixture per task."""
    build_dir = Path(build_dir)
    constants = build_dir / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    references = list(reference.get("_task_references") or (reference,))
    references.sort(key=lambda value: int(value.get("task_id", 0)))
    catalog = list(reference.get("task_catalog") or ())
    if catalog and len(references) != len(catalog):
        raise ValueError(
            f"GR00T fixture has {len(references)} task traces for {len(catalog)} tasks"
        )

    prompts = []
    starts_by_task = []
    for task_reference in references:
        backbone_inputs = task_reference["backbone_inputs"]
        input_ids = backbone_inputs["input_ids"].cpu()
        image_positions = (input_ids == IMAGE_TOKEN_ID)[0].numpy()
        prompt = np.array(task_reference["prompt_embedding"], dtype=np.float32)
        prompt[:, image_positions] = 0.0
        starts = image_token_starts(
            input_ids,
            len(backbone_inputs["pixel_values"]),
            int(task_reference["connector_width"]),
        )
        if starts != tuple(task_reference["image_token_starts"]):
            raise ValueError(
                f"task {task_reference.get('task_id')} image-token windows moved from "
                f"{task_reference['image_token_starts']} to {starts}"
            )
        prompts.append(prompt)
        starts_by_task.append(starts)

    def write(name: str, values) -> None:
        np.asarray(values, dtype="<f4").ravel().tofile(constants / name)

    write("prompt_embeddings", np.stack(prompts))
    write("image_additive_mask", references[0]["image_additive_mask"])
    write("text_additive_masks", np.stack([
        value["text_additive_mask"] for value in references
    ]))
    write("qwen_attention_masks", np.stack([
        value["qwen_attention_mask"] for value in references
    ]))
    for index, step in enumerate(references[0]["steps"]):
        write(f"tau_embedding_{index}", step["tau"])
        write(f"timestep_embedding_{index}", step["temb"])

    if catalog:
        (build_dir / "task_catalog.json").write_text(
            json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def save_fixture(path: Path, task_reference: dict, starts) -> None:
        backbone_inputs = task_reference["backbone_inputs"]
        fixture_values = {
            f"patches{index}": patchify(pixels.cpu().float(), rt.PATCH).numpy()
            for index, pixels in enumerate(backbone_inputs["pixel_values"])
        }
        if task_reference.get("raw_state") is not None:
            fixture_values["raw_state"] = np.asarray(task_reference["raw_state"], dtype=np.float32)
        steps = task_reference["steps"]
        np.savez(
            path,
            **fixture_values,
            task_id=np.asarray(task_reference.get("task_id", 0), dtype=np.int64),
            sequence=np.asarray(task_reference["sequence"], dtype=np.int64),
            image_shape=np.asarray(task_reference["image_shape"], dtype=np.int64),
            patch_grid=np.asarray(task_reference["patch_grid"], dtype=np.int64),
            connector_width=np.asarray(
                task_reference["connector_width"], dtype=np.int64
            ),
            image_token_starts=np.asarray(starts, dtype=np.int64),
            state=task_reference["state"],
            noise=task_reference["noise"],
            backbone_features=task_reference["backbone_features"],
            final_action=task_reference["final_action"],
            image_additive_mask=task_reference["image_additive_mask"],
            text_additive_mask=task_reference["text_additive_mask"],
            qwen_attention_mask=task_reference["qwen_attention_mask"],
            tau_embeddings=np.stack([step["tau"] for step in steps]),
            timestep_embeddings=np.stack([step["temb"] for step in steps]),
        )

    primary = build_dir / "groot_fixture.npz"
    save_fixture(primary, references[0], starts_by_task[0])
    for task_reference, starts in zip(references, starts_by_task, strict=True):
        task_id = int(task_reference.get("task_id", 0))
        save_fixture(
            build_dir / f"groot_fixture_task_{task_id:03d}.npz",
            task_reference,
            starts,
        )
    return primary


def write_normalization(
    checkpoint: str | Path,
    image_keys,
    out_path: Path,
    profile: Mapping[str, Any] | None = None,
) -> Path:
    """Write the exact checkpoint processor's affine state/action transforms.

    GR00T calls these arrays ``min/max`` statistics and maps them to [-1, 1].
    PoLiMa's generic normalize opcode is ``(x - mean) / std``, so the stored
    ``mean`` is the interval centre and ``std`` is half its range. Relative arm
    actions use the saved per-horizon ``relative_action`` block; absolute
    gripper actions use the ordinary action block. For sine/cosine state groups,
    the runtime uses raw angles directly; the affine state arrays are retained
    for compatibility but must not be applied to those groups.
    """
    from polima.policies.groot.profile import (
        _processor_config,
        _statistics,
        checkpoint_metadata,
    )

    checkpoint = Path(checkpoint)
    metadata = checkpoint_metadata(
        checkpoint, embodiment_tag=(profile or {}).get("embodiment_tag")
    )
    processor = _processor_config(checkpoint)
    kwargs = processor.get("processor_kwargs") or processor
    tag = metadata["embodiment_tag"]
    modality = kwargs["modality_configs"][tag]
    statistics = _statistics(checkpoint)[tag]
    use_percentiles = bool(kwargs.get("use_percentiles", False))
    use_relative = bool(kwargs.get("use_relative_action", False))
    low_key, high_key = ("q01", "q99") if use_percentiles else ("min", "max")

    def affine(block: dict, *, mean_std: bool) -> tuple[np.ndarray, np.ndarray]:
        if mean_std:
            center = np.asarray(block["mean"], dtype=np.float32)
            scale = np.asarray(block["std"], dtype=np.float32)
        else:
            low = np.asarray(block[low_key], dtype=np.float32)
            high = np.asarray(block[high_key], dtype=np.float32)
            center = (low + high) * np.float32(0.5)
            scale = (high - low) * np.float32(0.5)
        return center, np.where(np.abs(scale) > 0, scale, 1.0).astype(np.float32)

    state_cfg = modality["state"]
    state_mean_std = set(state_cfg.get("mean_std_embedding_keys") or ())
    state_pairs = [
        affine(statistics["state"][key], mean_std=key in state_mean_std)
        for key in metadata["state_keys"]
    ]
    state_mean = np.concatenate([pair[0].reshape(-1) for pair in state_pairs])
    state_std = np.concatenate([pair[1].reshape(-1) for pair in state_pairs])

    action_cfg = modality["action"]
    action_mean_std = set(action_cfg.get("mean_std_embedding_keys") or ())
    action_pairs = []
    for key, config in zip(
        metadata["action_keys"], action_cfg["action_configs"], strict=True
    ):
        relative = use_relative and str(config.get("rep", "")).lower() == "relative"
        section = "relative_action" if relative else "action"
        center, scale = affine(
            statistics[section][key], mean_std=key in action_mean_std
        )
        width = center.shape[-1]
        horizon = metadata["actions_per_chunk"]
        center = np.broadcast_to(center, (horizon, width)).copy()
        scale = np.broadcast_to(scale, (horizon, width)).copy()
        action_pairs.append((center, scale))
    action_mean = np.concatenate([pair[0] for pair in action_pairs], axis=-1)
    action_std = np.concatenate([pair[1] for pair in action_pairs], axis=-1)

    expected = (metadata["actions_per_chunk"], metadata["action_dim"])
    if state_mean.shape != (metadata["state_dim"],) or action_mean.shape != expected:
        raise ValueError(
            f"GR00T normalization shapes disagree with profile: "
            f"state={state_mean.shape}, action={action_mean.shape}, expected={expected}"
        )
    out_path = Path(out_path)
    np.savez(out_path, state_mean=state_mean, state_std=state_std,
             action_mean=action_mean, action_std=action_std)
    constants = out_path.parent / "constants"
    constants.mkdir(parents=True, exist_ok=True)
    for name, values in (("state_mean", state_mean), ("state_std", state_std),
                         ("action_mean", action_mean), ("action_std", action_std)):
        values.astype("<f4").tofile(constants / name)
    return out_path


def _verify_fixture(
    onnx_dir: Path,
    fixture_path: Path,
    report_path: Path,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    stage_dir: Path | None = None,
    sessions: dict[str, Any] | None = None,
    action_output: Path | None = None,
) -> dict:
    """Replay all 45 graphs under onnxruntime and compare against the trace.

    Run before anything is quantized, so a mismatch on the board afterwards is
    the compiler's or the runtime's, not the cut's.
    """
    import onnxruntime as ort

    onnx_dir = Path(onnx_dir)
    fixture = np.load(fixture_path)
    sessions = {} if sessions is None else sessions

    def run(name: str, feeds: dict) -> np.ndarray:
        if name not in sessions:
            sessions[name] = ort.InferenceSession(
                str(onnx_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        result = sessions[name].run(None, feeds)[0]
        if stage_dir is not None:
            Path(stage_dir).mkdir(parents=True, exist_ok=True)
            np.asarray(result, dtype="<f4").ravel().tofile(Path(stage_dir) / f"{name}.f32")
        return result

    patch_keys = sorted(
        (key for key in fixture.files if key.startswith("patches")),
        key=lambda key: int(key.removeprefix("patches")),
    )
    patch_grid = (
        tuple(int(value) for value in fixture["patch_grid"])
        if "patch_grid" in fixture.files
        else (rt.PATCH_SIDE, rt.PATCH_SIDE)
    )
    connector_width = (
        int(fixture["connector_width"])
        if "connector_width" in fixture.files
        else rt.CONNECTOR_WIDTH
    )
    image_tokens = []
    for patch_key in patch_keys:
        hidden = run(
            "eagle_vision_patch", {"patches": fixture[patch_key].astype(np.float32)}
        )
        for name in rt.vision_stage_names():
            hidden = run(name, {"hidden": hidden})
        hidden = run("eagle_vision_post_norm", {"hidden": hidden})
        folded = pixel_unshuffle_tokens(
            torch.from_numpy(hidden),
            patch_grid[0],
            rt.UNSHUFFLE,
            width=patch_grid[1],
        )
        image_tokens.append(run("eagle_vision_connector", {"hidden": folded.numpy()}))

    # The prompt half of the sequence is a constant; the fixture's recorded
    # backbone input carries it, with every image window replaced here.
    sequence = int(fixture["sequence"])
    starts = fixture["image_token_starts"].astype(int).tolist()
    language = np.zeros((1, sequence, rt.LANGUAGE_CHANNELS), dtype=np.float32)
    task_id = int(fixture["task_id"]) if "task_id" in fixture.files else 0
    prompt_values = np.fromfile(
        Path(fixture_path).parent / "constants" / "prompt_embeddings", dtype="<f4"
    ).reshape(-1, sequence, rt.LANGUAGE_CHANNELS)
    if not 0 <= task_id < len(prompt_values):
        raise ValueError(
            f"fixture task id {task_id} is outside {len(prompt_values)} prompt embeddings"
        )
    language += prompt_values[task_id:task_id + 1]
    for start, tokens in zip(starts, image_tokens, strict=True):
        language[:, start:start + connector_width] = tokens
    qwen_hidden = language[:, None, :, :]
    qwen_mask = fixture["qwen_attention_mask"].astype(np.float32)
    qwen_mask = np.transpose(qwen_mask, (0, 2, 3, 1))
    for name in rt.qwen_stage_names():
        qwen_hidden = run(name, {
            "hidden": qwen_hidden,
            "attention_mask": qwen_mask,
        })
    language = qwen_hidden[:, 0, :, :]
    backbone_features = run("eagle_output_norm", {"hidden": language})

    eagle_reference = fixture["backbone_features"].astype(np.float32)

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        left, right = left.ravel(), right.ravel()
        return float(
            np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        )

    eagle_cosine = cosine(backbone_features, eagle_reference)
    valid_tokens = np.logical_or(
        fixture["image_additive_mask"] == 0,
        fixture["text_additive_mask"] == 0,
    )[0]
    eagle_valid_cosine = cosine(
        backbone_features[:, valid_tokens, :],
        eagle_reference[:, valid_tokens, :],
    )

    state_features = run("state_project", {"state": fixture["state"].astype(np.float32)})
    actions = fixture["noise"].astype(np.float32).copy()
    per_step = []
    for index in range(rt.DENOISE_STEPS):
        action_features = run("action_project", {
            "actions": actions,
            "tau_embedding": fixture["tau_embeddings"][index].astype(np.float32),
        })
        temb = fixture["timestep_embeddings"][index].astype(np.float32)
        hidden = np.concatenate((state_features, action_features), axis=1)
        for pair, name in enumerate(rt.block_names()):
            mask = fixture["text_additive_mask" if pair % 2 == 0 else "image_additive_mask"]
            hidden = run(name, {
                "hidden": hidden,
                "backbone_features": backbone_features,
                "additive_mask": mask.astype(np.float32),
                "temb": temb,
            })
        velocity = run("action_tail", {"hidden": hidden, "temb": temb})
        actions = actions + np.float32(rt.DT) * velocity
        per_step.append({"step": index, "velocity_l2": float(np.linalg.norm(velocity))})

    if action_output is not None:
        np.save(action_output, actions)
    expected = fixture["final_action"].astype(np.float32)
    difference = np.abs(actions - expected)
    close = difference <= atol + rtol * np.abs(expected)
    action_cosine = cosine(actions, expected)
    close_fraction = float(close.mean())
    strict_allclose = bool(close.all())
    report = {
        "task_id": task_id,
        "ok": bool(close_fraction >= 0.99 and action_cosine >= 0.99),
        "strict_allclose": strict_allclose,
        "minimum_close_fraction": 0.99,
        "minimum_action_cosine_similarity": 0.99,
        "close_fraction": close_fraction,
        "action_cosine_similarity": action_cosine,
        "max_abs": float(difference.max()),
        "atol": atol,
        "rtol": rtol,
        "mean_abs": float(difference.mean()),
        "eagle_cosine_similarity": eagle_cosine,
        "eagle_valid_token_cosine_similarity": eagle_valid_cosine,
        "onnx_shape": list(actions.shape),
        "reference_shape": list(expected.shape),
        "steps": per_step,
    }
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    return report


def verify_chain(
    onnx_dir: Path,
    fixture_path: Path,
    report_path: Path,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    stage_dir: Path | None = None,
) -> dict:
    """Verify the shared ONNX graph set against every catalog instruction."""
    fixture_path = Path(fixture_path)
    task_fixtures = sorted(
        fixture_path.parent.glob("groot_fixture_task_*.npz")
    )
    if not task_fixtures:
        task_fixtures = [fixture_path]

    sessions: dict[str, Any] = {}
    reports = []
    for index, task_fixture in enumerate(task_fixtures):
        task_report = Path(report_path).with_name(
            f"{Path(report_path).stem}_task_{index:03d}.json"
        )
        reports.append(
            _verify_fixture(
                onnx_dir,
                task_fixture,
                task_report,
                atol=atol,
                rtol=rtol,
                stage_dir=stage_dir if index == 0 else None,
                sessions=sessions,
            )
        )

    aggregate = {
        "ok": all(report["ok"] for report in reports),
        "strict_allclose": all(report["strict_allclose"] for report in reports),
        "task_count": len(reports),
        "max_abs": max(report["max_abs"] for report in reports),
        "mean_abs": max(report["mean_abs"] for report in reports),
        "atol": atol,
        "rtol": rtol,
        "eagle_cosine_similarity": min(
            report["eagle_cosine_similarity"] for report in reports
        ),
        "eagle_valid_token_cosine_similarity": min(
            report["eagle_valid_token_cosine_similarity"] for report in reports
        ),
        "action_cosine_similarity": min(
            report["action_cosine_similarity"] for report in reports
        ),
        "close_fraction": min(report["close_fraction"] for report in reports),
        "onnx_shape": reports[0]["onnx_shape"],
        "reference_shape": reports[0]["reference_shape"],
        "tasks": reports,
    }
    Path(report_path).write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def validate_checkpoint(checkpoint: str | Path) -> list[str]:
    """Problems that would only surface deep inside the export otherwise."""
    checkpoint = Path(checkpoint)
    problems: list[str] = []
    required = (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    )
    for name in required:
        if not (checkpoint / name).is_file():
            problems.append(f"{checkpoint} has no {name}; checkpoint is incomplete")
    if not problems:
        try:
            from polima.policies.groot.profile import checkpoint_metadata

            checkpoint_metadata(checkpoint)
        except (OSError, TypeError, ValueError) as error:
            problems.append(str(error))
    return problems
