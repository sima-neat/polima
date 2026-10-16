"""Build a Pi0.5 request on the robot side (numpy only; runs on the DevKit).

Pi0.5's prompt carries the robot state as 256-bin tokens, so every request needs
the prompt tokenized and embedded. That happens here, mirroring LeRobot's pi05
pre-processor exactly:

  1. normalize the state (QUANTILES: 2(x-q01)/(q99-q01) - 1; identity without stats)
  2. "Task: {task}, State: {bins};\\nAction: " with bins = digitize(state, 256 bins) - 1
  3. PaliGemma tokenizer, right-padded to 200 tokens (BOS added)
  4. rows of the embedding table, already scaled by sqrt(2048)
  5. images: bilinear resize with centred zero padding to 224x224, then [-1, 1]

and adds what the plan cannot derive: the key bias row and the action tokens'
RoPE rows. The server returns normalized actions; `actions()` undoes the
normalization (and relative actions, when the checkpoint was trained with them).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE = 224
TOKENS = 200
WIDTH = 2048
CHUNK = 50
LANE = 32
HEAD_DIM = 256
MASK = -1.0e4
THETA = 10_000.0


def resize_with_pad(image: np.ndarray, size: int = IMAGE) -> np.ndarray:
    """HWC uint8 or float [0,1] -> CHW float32 in [-1, 1] (LeRobot resize_with_pad_torch)."""
    img = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    h, w = img.shape[:2]
    if (h, w) != (size, size):
        ratio = max(w / size, h / size)
        rh, rw = int(h / ratio), int(w / ratio)
        img = np.clip(_bilinear(img, rh, rw), 0.0, 1.0)
        top, left = (size - rh) // 2, (size - rw) // 2
        padded = np.zeros((size, size, img.shape[2]), np.float32)
        padded[top:top + rh, left:left + rw] = img
        img = padded
    return np.ascontiguousarray((img * 2.0 - 1.0).transpose(2, 0, 1))


def _bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """torch.nn.functional.interpolate(mode='bilinear', align_corners=False), no antialias."""
    def axis(n_in, n_out):
        src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
        src = np.maximum(src, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (src - i0).astype(np.float32)

    y0, y1, wy = axis(img.shape[0], out_h)
    x0, x1, wx = axis(img.shape[1], out_w)
    top = img[y0][:, x0] * (1 - wx)[None, :, None] + img[y0][:, x1] * wx[None, :, None]
    bottom = img[y1][:, x0] * (1 - wx)[None, :, None] + img[y1][:, x1] * wx[None, :, None]
    return top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]


def patchify(chw: np.ndarray) -> np.ndarray:
    """[3, 224, 224] -> [256, 588]: row-major 14x14 patches laid out (c, y, x)."""
    return np.ascontiguousarray(chw.reshape(3, 16, 14, 16, 14).transpose(1, 3, 0, 2, 4).reshape(256, 588))


def prompt(task: str, state: np.ndarray) -> str:
    """The Pi0.5 prompt for a normalized state (Pi05PrepareStateTokenizerProcessorStep)."""
    bins = np.digitize(np.asarray(state, np.float32), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "


def rope_rows(start: int, count: int = CHUNK) -> np.ndarray:
    """cos rows then sin rows for positions start..start+count-1, [2*count, 256]."""
    inv_freq = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.int64).astype(np.float32) / HEAD_DIM))
    freqs = np.arange(start, start + count, dtype=np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.concatenate([np.cos(emb), np.sin(emb)]).astype(np.float32)


@dataclass
class Stats:
    """Quantile statistics for state and action (absent for pi05_base)."""

    state_q01: np.ndarray | None = None
    state_q99: np.ndarray | None = None
    action_q01: np.ndarray | None = None
    action_q99: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> Stats:
        path = Path(path)
        if not path.is_file():
            return cls()
        with np.load(path) as data:
            return cls(**{k: data[k].astype(np.float32) for k in data.files
                          if k in ("state_q01", "state_q99", "action_q01", "action_q99")})

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_q01 is None:
            return state.astype(np.float32)
        span = np.where(self.state_q99 - self.state_q01 == 0, 1e-8, self.state_q99 - self.state_q01)
        return (2.0 * (state - self.state_q01) / span - 1.0).astype(np.float32)

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_q01 is None:
            return actions.astype(np.float32)
        span = np.where(self.action_q99 - self.action_q01 == 0, 1e-8, self.action_q99 - self.action_q01)
        return ((actions + 1.0) * span / 2.0 + self.action_q01).astype(np.float32)


class Pi05Client:
    """Request builder for one bundle.

    `root` holds client.json, tokenizer.json, embedding.f16 (257152 x 2048, scaled)
    and normalization_stats.npz -- the bundle's robot_client/pi05/ folder.
    """

    def __init__(self, root: str | Path, tokenizer=None):
        root = Path(root)
        self.config = json.loads((root / "client.json").read_text())
        self.cameras = int(self.config["cameras"])
        self.state_dim = int(self.config["state_dim"])
        self.action_dim = int(self.config["action_dim"])
        #: "llima": HWC images and a prefix padded to one 1024-token LLiMa group;
        #: "onnx": 14x14 patches and a 256*C + 200 token prefix.
        self.backend = self.config.get("backend", "onnx")
        self.prefix = int(self.config.get("prefix_tokens", 256 * self.cameras + TOKENS))
        self.language_rows = self.prefix - 256 * self.cameras
        self.stats = Stats.load(root / "normalization_stats.npz")
        self.table = np.memmap(root / "embedding.f16", dtype=np.float16, mode="r").reshape(-1, WIDTH)
        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer = tokenizer

    def tokens(self, text: str) -> tuple[np.ndarray, int]:
        """Token ids padded to 200 (truncated like the processor) and the valid count."""
        ids = self.tokenizer.encode(text).ids[:TOKENS]
        padded = np.zeros(TOKENS, np.int64)
        padded[:len(ids)] = ids
        return padded, len(ids)

    def request(self, images: list[np.ndarray], state: np.ndarray, task: str,
                noise: np.ndarray | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
        if len(images) != self.cameras:
            raise ValueError(f"this bundle needs all {self.cameras} cameras, got {len(images)}")
        state = np.asarray(state, np.float32).reshape(-1)[: self.state_dim]
        ids, valid = self.tokens(prompt(task, self.stats.normalize_state(state)))
        language = np.zeros((self.language_rows, WIDTH), np.float32)
        language[:valid] = self.table[ids[:valid]].astype(np.float32)
        bias = np.zeros(self.prefix, np.float32)
        bias[256 * self.cameras + valid:] = MASK
        if noise is None:
            noise = np.random.default_rng(seed).standard_normal((CHUNK, LANE)).astype(np.float32)
        if self.backend == "llima":  # LLiMa's vision ELF takes the [-1, 1] image as HWC
            request = {f"image{k}": np.ascontiguousarray(resize_with_pad(img).transpose(1, 2, 0))
                       for k, img in enumerate(images)}
        else:
            request = {f"image{k}": patchify(resize_with_pad(img)) for k, img in enumerate(images)}
        request.update(language=language, prefix_bias=bias,
                       suffix_rope=rope_rows(256 * self.cameras + valid), noise=noise.astype(np.float32))
        return request

    def actions(self, normalized: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
        """Server output [50, action_dim] -> robot actions."""
        out = self.stats.unnormalize_actions(np.asarray(normalized, np.float32).reshape(CHUNK, -1))
        if self.config.get("relative_actions") and state is not None:
            mask = np.asarray(self.config.get("relative_mask") or [True] * self.action_dim, bool)
            out[:, mask] += np.asarray(state, np.float32)[: self.action_dim][mask]
        return out
