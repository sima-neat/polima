"""Read a Pi0.5 checkpoint's tensors without instantiating the LeRobot model."""
from __future__ import annotations

import json
from pathlib import Path

PREFIX_LAYERS = 18
EXPERT_LAYERS = 18
EMBED = "paligemma_with_expert.paligemma.lm_head.weight"  # tied with embed_tokens
#: transformers v5's GemmaTextScaledWordEmbedding multiplies every looked-up row
#: by sqrt(hidden) (a float32 scalar), and PiGemmaModel adds no other normalizer.
EMBED_SCALE = 2048 ** 0.5


def checkpoint_file(checkpoint: str | Path) -> Path:
    path = Path(checkpoint) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"{checkpoint}: no model.safetensors")
    return path


def config(checkpoint: str | Path) -> dict:
    data = json.loads((Path(checkpoint) / "config.json").read_text())
    if data.get("type") != "pi05":
        raise ValueError(f"{checkpoint}: config type {data.get('type')!r} is not pi05")
    return data


def stored_names(handle) -> dict[str, str]:
    """Canonical tensor name -> name stored in the file. Newer LeRobot saves prefix every
    tensor with `model.` (PI05Policy.model); LeRobot strips it on load, and so do we."""
    names = {}
    for name in handle.keys():  # noqa: SIM118 - a safetensors handle, not a dict
        lerobot = name.startswith(("model.paligemma_with_expert", "model.action_", "model.time_mlp"))
        names[name[len("model."):] if lerobot else name] = name
    return names


def load(checkpoint: str | Path, prefixes: tuple[str, ...] = ("",), skip: tuple[str, ...] = (EMBED,)) -> dict:
    """The checkpoint's float32 tensors (canonical names) that start with any of `prefixes`."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(checkpoint_file(checkpoint)), "pt") as handle:
        for name, stored in stored_names(handle).items():
            if name in skip or not name.startswith(prefixes) or name.endswith("embed_tokens.weight"):
                continue
            out[name] = handle.get_tensor(stored).float()
    return out


def _embedding(handle):
    """The token embedding: the tied lm_head (pi05_base), else embed_tokens."""
    names = stored_names(handle)
    for name in (EMBED, "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"):
        if name in names:
            return handle.get_tensor(names[name])
    raise KeyError("checkpoint has neither the PaliGemma lm_head nor embed_tokens")


def embedding_table(checkpoint: str | Path):
    """The 257152 x 2048 token embedding (the tied lm_head) as float32, already
    multiplied by EMBED_SCALE, so a row is exactly what the language model sees."""
    import torch
    from safetensors import safe_open

    with safe_open(str(checkpoint_file(checkpoint)), "pt") as handle:
        return _embedding(handle).float() * torch.tensor(EMBED_SCALE, dtype=torch.float32)


PALIGEMMA_BASE = "google/paligemma-3b-pt-224"
PALIGEMMA_FILES = ("config.json", "added_tokens.json", "generation_config.json", "preprocessor_config.json",
                   "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model")


def paligemma_dir(checkpoint: str | Path, out: str | Path) -> Path:
    """An HF PaliGemma folder holding Pi0.5's own vision and language weights, in the
    (pre-v5) weight layout LLiMa reads: what `llima-compile` turns into the prefill ELFs.

    The config and tokenizer come from google/paligemma-3b-pt-224 (the architecture
    Pi0.5 starts from); the vocabulary is Pi0.5's 257152 rows.
    """
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "paligemma_with_expert.paligemma."
    renames = ((prefix + "model.language_model.layers.", "language_model.model.layers."),
               (prefix + "model.language_model.norm.", "language_model.model.norm."),
               (prefix + "model.vision_tower.vision_model.", "vision_tower.vision_model."),
               (prefix + "model.multi_modal_projector.linear.", "multi_modal_projector.linear."))
    tensors = {}
    with safe_open(str(checkpoint_file(checkpoint)), "pt") as handle:
        # float32 throughout, as the config declares (fine-tunes are often saved in bf16)
        tensors["language_model.model.embed_tokens.weight"] = _embedding(handle).float().contiguous()
        for name, stored in stored_names(handle).items():
            if name.endswith("embed_tokens.weight"):
                continue
            for old, new in renames:
                if name.startswith(old):
                    tensors[new + name[len(old):]] = handle.get_tensor(stored).float().contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for name in PALIGEMMA_FILES:
        (out / name).write_bytes(Path(hf_hub_download(PALIGEMMA_BASE, name)).read_bytes())
    config = json.loads((out / "config.json").read_text())
    config["vocab_size"] = config["text_config"]["vocab_size"] = 257152
    config["text_config"].update(head_dim=256, hidden_activation="gelu_pytorch_tanh", rope_theta=10000.0,
                                 rms_norm_eps=1e-6, max_position_embeddings=8192)
    config["mm_tokens_per_image"] = 256
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out
