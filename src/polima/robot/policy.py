"""Bundle-driven policy input preparation and transport; no robot control.

- CompiledPolicyAdapter: a GR00T or Pi0.5 bundle's exact request tensors (camera
  frames, state, noise; Pi0.5 also its tokenized prompt) for the SO-101 loop in
  polima.robot.so101.
- BundlePolicy: an ACT bundle, its whole contract read from the bundle's plan.json.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from polima.bundle.layout import Bundle
from polima.policies.base import PolicySpec, TensorSpec
from polima.policies.registry import get_policy
from polima.util.jsonio import read_json
from polima.wire.client import PolimaSOMClient
from polima.wire.protocol import DTYPE, ProtocolError


@dataclass(frozen=True)
class PolicyPrediction:
    """One replayable policy transaction before robot safety handling."""

    request_tensors: dict[str, np.ndarray]
    raw_actions: np.ndarray
    actions: np.ndarray
    latency_ms: float


def prepare_groot_image(image: np.ndarray, context: dict[str, Any]) -> np.ndarray:
    """Apply the checkpoint's deterministic evaluation transform and patchify."""

    preprocess = dict(context.get("image_preprocess") or {})
    image_shape = tuple(int(value) for value in context.get("image_shape", ()))
    if len(image_shape) != 2 or not preprocess:
        raise ValueError(
            "GR00T bundle lacks portable image preprocessing metadata; "
            "re-export it with the current PoLiMa before robot execution"
        )

    if preprocess.get("format") == "groot-qwen3-image-v1":
        return _prepare_qwen3_image(image, context)
    if preprocess.get("format") != "groot-eagle-image-v1":
        raise ValueError(
            "GR00T bundle has an unsupported image preprocessing contract; "
            "re-export it with the current PoLiMa before robot execution"
        )
    if preprocess.get("color_space") != "RGB":
        raise ValueError("GR00T image preprocessing contract must use RGB")
    groot_eval = dict(preprocess.get("gr00t_eval") or {})
    eagle = dict(preprocess.get("eagle") or {})
    if not groot_eval or not eagle:
        raise ValueError("GR00T image preprocessing contract is incomplete")

    array = _rgb_hwc(image)
    if groot_eval.get("use_albumentations"):
        transformed = _groot_albumentations_eval(array, groot_eval)
    else:
        transformed = _groot_torchvision_eval(array, groot_eval)

    if not eagle.get("resize_to_compiled_shape"):
        raise ValueError("GR00T Eagle resize contract is missing")
    if eagle.get("resize_resample") != "bicubic":
        raise ValueError("GR00T Eagle resize must use Pillow bicubic interpolation")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("GR00T robot inference needs Pillow") from error
    transformed = np.asarray(
        Image.fromarray(transformed).resize(
            (image_shape[1], image_shape[0]), resample=Image.Resampling.BICUBIC
        )
    )
    if transformed.shape[:2] != image_shape:
        raise ValueError(
            f"GR00T preprocessing produced {transformed.shape[:2]}, "
            f"but this compiled bundle requires {image_shape}"
        )
    rescale_factor = eagle.get("rescale_factor")
    if rescale_factor is None or float(rescale_factor) <= 0:
        raise ValueError("GR00T bundle has no valid image rescale factor")
    mean_values = eagle.get("image_mean")
    std_values = eagle.get("image_std")
    if mean_values is None or std_values is None:
        raise ValueError("GR00T bundle has no image normalization metadata")
    mean = np.asarray(mean_values, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std_values, dtype=np.float32).reshape(1, 1, 3)
    if np.any(std <= 0):
        raise ValueError("GR00T bundle has invalid image normalization metadata")

    from polima.policies.groot.runtime import PATCH, PATCH_CHANNELS

    height, width = transformed.shape[:2]
    if height % PATCH or width % PATCH:
        raise ValueError(f"GR00T image shape {image_shape} is not divisible by patch {PATCH}")
    rows, columns = height // PATCH, width // PATCH

    values = transformed.astype(np.float32)
    values *= np.float32(rescale_factor)
    # The same float32 subtract and divide per element as broadcasting the
    # (1, 1, 3) statistics, but against full-size copies of them: NumPy's
    # broadcast over a last axis of 3 is slow on the A65 (devkit, 336x252:
    # 8.4 -> 1.2 ms per camera, bit-identical).
    mean_full, std_full = _tiled_channel_stats(values.shape, mean.tobytes(), std.tobytes())
    values -= mean_full
    values /= std_full
    folded = values.reshape(rows, PATCH, columns, PATCH, 3)
    return folded.transpose(0, 2, 1, 3, 4).reshape(rows * columns, PATCH_CHANNELS)


@lru_cache(maxsize=8)
def _tiled_channel_stats(
    shape: tuple[int, ...], mean: bytes, std: bytes
) -> tuple[np.ndarray, np.ndarray]:
    def tile(raw: bytes) -> np.ndarray:
        values = np.frombuffer(raw, dtype=np.float32).reshape(1, 1, 3)
        full = np.ascontiguousarray(np.broadcast_to(values, shape))
        full.flags.writeable = False
        return full

    return tile(mean), tile(std)


def _groot_torchvision_eval(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    try:
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform_f
    except ImportError as error:
        raise RuntimeError("GR00T robot inference needs torch and torchvision") from error

    target = config.get("image_target_size")
    crop = config.get("image_crop_size")
    if not target or not crop:
        raise ValueError("GR00T torchvision preprocessing requires target and crop sizes")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    if config.get("letter_box_transform"):
        height, width = tensor.shape[-2:]
        side = max(height, width)
        left = (side - width) // 2
        top = (side - height) // 2
        tensor = transform_f.pad(
            tensor,
            [left, top, side - width - left, side - height - top],
            fill=0,
        )
    tensor = transform_f.resize(
        tensor,
        [int(target[0]), int(target[1])],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = transform_f.center_crop(tensor, [int(crop[0]), int(crop[1])])
    tensor = transform_f.resize(
        tensor,
        [int(target[0]), int(target[1])],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return tensor.permute(1, 2, 0).contiguous().numpy()


def _groot_albumentations_eval(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("GR00T robot inference needs opencv") from error

    result = image
    if config.get("letter_box_transform"):
        height, width = result.shape[:2]
        side = max(height, width)
        top = (side - height) // 2
        left = (side - width) // 2
        result = cv2.copyMakeBorder(
            result,
            top,
            side - height - top,
            left,
            side - width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )

    shortest = config.get("shortest_image_edge")
    target = config.get("image_target_size")
    if shortest is None and target:
        shortest = target[0]
    fraction = config.get("crop_fraction")
    crop = config.get("image_crop_size")
    if fraction is None and target and crop:
        fraction = float(crop[0]) / float(target[0])
    if shortest is None or fraction is None or not 0 < float(fraction) <= 1:
        raise ValueError("GR00T albumentations preprocessing metadata is incomplete")

    def resize_shortest(value: np.ndarray) -> np.ndarray:
        height, width = value.shape[:2]
        scale = float(shortest) / min(height, width)
        shape = (round(width * scale), round(height * scale))
        return cv2.resize(value, shape, interpolation=cv2.INTER_AREA)

    result = resize_shortest(result)
    height, width = result.shape[:2]
    crop_height = max(1, int(height * float(fraction)))
    crop_width = max(1, int(width * float(fraction)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    result = result[top : top + crop_height, left : left + crop_width]
    return resize_shortest(result)


def _rgb_hwc(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = array.transpose(1, 2, 0)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected an RGB HWC camera frame, got {array.shape}")
    return np.ascontiguousarray(array)


class CompiledPolicyAdapter:
    """Convert robot observations to one bundle's exact TCP tensors."""

    def __init__(
        self,
        spec: PolicySpec,
        bundle: Bundle,
        address: str,
        *,
        task_id: int = 0,
        timeout: float = 30.0,
        seed: int | None = None,
        noise: str = "random",
        client_factory=PolimaSOMClient,
    ) -> None:
        # "zero" starts every flow-matching request from the same all-zero
        # noise, so consecutive plans stop disagreeing merely because of a
        # different random draw (ACT likewise uses a zero latent at inference).
        if noise not in ("random", "zero"):
            raise ValueError(f"noise must be 'random' or 'zero', not {noise!r}")
        stats = None
        self.pi05 = None
        if spec.name == "pi05":
            # Pi0.5's client normalizes (quantiles), tokenizes and embeds the prompt itself.
            from polima.policies.pi05.client import Pi05Client

            self.pi05 = Pi05Client(bundle.root / "robot_client" / "pi05")
        elif spec.wire.normalization_side == "client":
            candidates = (
                bundle.root / spec.wire.stats_file,
                bundle.fixtures_dir / spec.wire.stats_file,
            )
            stats = next((path for path in candidates if path.is_file()), None)
            if stats is None:
                raise FileNotFoundError(
                    f"{spec.name} bundle needs {spec.wire.stats_file} for inference"
                )
        self.spec = spec
        self.bundle = bundle
        self.task_id = int(task_id)
        self.client = client_factory(spec, address, stats=stats, timeout=timeout)
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        #: Pi0.5 instruction (it is part of the prompt, not a compiled catalog row).
        self.task = ""

    def _camera_pool(self, cameras: int):
        pool = getattr(self, "_camera_executor", None)
        if pool is None or pool._max_workers < cameras:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=max(1, cameras), thread_name_prefix="camera")
            self._camera_executor = pool
        return pool

    def _observation_inputs(
        self, observation: dict[str, Any]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        required = (*self.spec.robot.joint_names, *self.spec.robot.camera_keys)
        missing = [name for name in required if name not in observation]
        if missing:
            raise KeyError(f"Robot observation is missing keys: {missing}")
        state = np.asarray(
            [observation[name] for name in self.spec.robot.joint_names], dtype=np.float32
        )
        images = [_rgb_hwc(np.asarray(observation[name])) for name in self.spec.robot.camera_keys]
        return state, images

    def _prepare_request(
        self, state: np.ndarray, images: list[np.ndarray]
    ) -> dict[str, np.ndarray]:
        if self.spec.name == "act":
            return self.client.prepare_inputs(images, state)
        if self.pi05 is not None:
            noise = (np.zeros((50, 32), DTYPE) if self.noise == "zero"
                     else self.rng.standard_normal((50, 32)).astype(DTYPE))
            task = self.task or self.pi05.config.get("task", "")
            return self.pi05.request(images, state, task, noise=noise)

        tensors: dict[str, np.ndarray] = {"state": state}
        image_specs = [
            tensor
            for tensor in self.spec.wire.request_tensors
            if tensor.name.startswith(("image", "patches"))
        ]
        if len(image_specs) != len(images):
            raise ProtocolError(
                f"bundle expects {len(image_specs)} cameras, robot supplied {len(images)}"
            )
        if self.spec.name == "groot":
            context = dict(self.spec.runtime_context)
            # One thread per camera: OpenCV, Pillow, torch and the large numpy
            # operations release the GIL, so two 640x480 frames take ~the time
            # of one (measured on the devkit with the earlier per-pixel
            # normalisation: 60 -> 33 ms N1.6, 35 -> 21 ms N1.7). Same values.
            prepared = self._camera_pool(len(images)).map(
                lambda image: prepare_groot_image(image, context), images
            )
            for tensor, values in zip(image_specs, prepared, strict=True):
                tensors[tensor.name] = np.asarray(values, dtype=DTYPE)
        else:
            raise ValueError(f"Compiled client does not support policy {self.spec.name!r}")

        noise_spec = next(
            (tensor for tensor in self.spec.wire.request_tensors if tensor.name == "noise"),
            None,
        )
        if noise_spec is not None:
            if self.noise == "zero":
                tensors["noise"] = np.zeros(noise_spec.shape, dtype=DTYPE)
            else:
                tensors["noise"] = self.rng.standard_normal(noise_spec.shape).astype(DTYPE)
        return tensors

    def predict_detailed(self, observation: dict[str, Any]) -> PolicyPrediction:
        """Return the exact request, raw response, and robot-unit actions."""

        state, images = self._observation_inputs(observation)
        tensors = self._prepare_request(state, images)
        raw_actions, latency_ms = self.client.predict_raw(tensors, task_id=self.task_id)
        actions = (self.pi05.actions(raw_actions, state) if self.pi05 is not None
                   else self.client.decode_actions(raw_actions))
        return PolicyPrediction(
            request_tensors=tensors,
            raw_actions=raw_actions,
            actions=actions,
            latency_ms=latency_ms,
        )

    def predict(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        prediction = self.predict_detailed(observation)
        return prediction.actions, prediction.latency_ms


def _prepare_qwen3_image(image: np.ndarray, context: dict[str, Any]) -> np.ndarray:
    """Reuse GR00T eval transforms, then the saved Qwen3 fast image contract."""
    import torch
    from torchvision.transforms.functional import InterpolationMode, resize

    config = context["image_preprocess"]
    evaluation = config["gr00t_eval"]
    array = _rgb_hwc(image)
    transformed = (_groot_albumentations_eval(array, evaluation)
                   if evaluation.get("use_albumentations") else _groot_torchvision_eval(array, evaluation))
    height, width = map(int, context["image_shape"])
    qwen = config["qwen"]
    patch = int(qwen["patch_size"])
    temporal = int(qwen["temporal_patch_size"])
    merge = int(qwen["merge_size"])
    if height % (patch * merge) or width % (patch * merge):
        raise ValueError("Qwen3 image shape must align to patch and merge sizes")
    value = torch.from_numpy(np.ascontiguousarray(transformed.transpose(2, 0, 1)))
    value = resize(value, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
    # Same arithmetic order as the fast image processor's fused rescale/normalize.
    mean = torch.tensor(qwen["image_mean"])[:, None, None] * 255.0
    std = torch.tensor(qwen["image_std"])[:, None, None] * 255.0
    value = (value - mean) / std
    gh, gw = height // patch, width // patch
    value = value[None].expand(temporal, -1, -1, -1)
    value = value.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    return value.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * temporal * patch * patch).numpy()


# -- ACT bundles ----------------------------------------------------------------

class BundlePolicy:
    def __init__(self, root: Path, address: str, task: str | int = "",
                 rng: np.random.Generator | None = None) -> None:
        manifest, plan = read_json(root / "bundle.json"), read_json(root / "plan.json")
        bundle = Bundle.from_dict(manifest, root)
        self.spec = with_plan(get_policy(bundle.policy), plan)
        robot = plan.get("robot") or {}
        self.camera_keys = [role for role, _ in robot.get("camera_roles", ())] or list(
            self.spec.robot.camera_keys)
        self.joint_names = list(robot.get("joint_names") or self.spec.robot.joint_names)
        self.context = plan.get("runtime_context") or {}
        self.tasks = list(plan.get("tasks") or manifest.get("tasks") or [])
        self.task_id = task_index(task, self.tasks, int(plan.get("default_task") or 0))
        self.rng = rng or np.random.default_rng()
        stats = root / "fixtures" / self.spec.wire.stats_file
        self.client = PolimaSOMClient(self.spec, address,
                                      stats=stats if stats.is_file() else None)

    @property
    def task(self) -> str:
        return self.tasks[self.task_id] if self.tasks else ""

    def predict(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Robot-unit actions shaped like the bundle's response, and model time in ms."""
        state = np.asarray([observation[name] for name in self.joint_names], dtype=np.float32)
        images = [observation[key] for key in self.camera_keys]
        if self.spec.name != "act":  # GR00T uses CompiledPolicyAdapter
            raise ValueError(f"BundlePolicy runs ACT bundles, not {self.spec.display_name}")
        return self.client.predict(images, state)


def with_plan(spec: PolicySpec, plan: dict) -> PolicySpec:
    """The registry spec with the wire geometry the bundle was compiled with."""
    wire = plan.get("wire") or {}
    if not wire:
        return spec
    return replace(spec, wire=replace(
        spec.wire,
        magic=int(wire.get("magic", spec.wire.magic)),
        version=int(wire.get("version", spec.wire.version)),
        default_port=int(wire.get("default_port", spec.wire.default_port)),
        request_tensors=tuple(TensorSpec(item["name"], tuple(item["shape"]))
                              for item in wire.get("request_tensors", ())) or spec.wire.request_tensors,
        response_shape=tuple(wire.get("response_shape", spec.wire.response_shape)),
    ))


def task_index(task: str | int, tasks: list[str], default: int = 0) -> int:
    """A catalog index from Studio's instruction text, an index, or nothing."""
    if task in ("", None):
        return default
    if str(task).isdigit() and int(task) < max(len(tasks), 1):
        return int(task)
    if task in tasks:
        return tasks.index(task)
    raise ValueError(f"unknown task {task!r}; this bundle knows {tasks}")
