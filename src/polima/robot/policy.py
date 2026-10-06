"""Shared bundle-driven policy preprocessing and transport; no robot control.

Two front ends share this module:

- CompiledPolicyAdapter: GR00T, SmolVLA and ACT through the established ACT
  control loop (polima.robot.so101) and the MyCobot client. This is the path
  verified on the SO-101 with GR00T N1.6/N1.7 and SmolVLA.
- BundlePolicy: PoLiMa's own SO-101 loop (polima.robot.so101_loop), which
  takes its whole contract from the bundle's plan.json.
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


def prepare_smolvla_image(image: np.ndarray) -> np.ndarray:
    """Match the proven SO-101 client's left/top-padded SigLIP transform."""

    try:
        import torch
        import torch.nn.functional as torch_f
    except ImportError as error:
        raise RuntimeError(
            "SmolVLA robot inference needs torch in the robot environment"
        ) from error

    array = _rgb_hwc(image)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(dtype=torch.float32)
    if array.dtype == np.uint8 or float(tensor.max()) > 1.0:
        tensor.div_(255.0)
    current_height, current_width = tensor.shape[2:]
    ratio = max(current_width / 512, current_height / 512)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    tensor = torch_f.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    tensor = torch_f.pad(
        tensor,
        (512 - resized_width, 0, 512 - resized_height, 0),
        value=0,
    )
    return tensor.mul(2.0).sub(1.0)[0].permute(1, 2, 0).contiguous().numpy()


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
        if spec.wire.normalization_side == "client":
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
        if self.spec.name == "smolvla":
            for tensor, image in zip(image_specs, images, strict=True):
                tensors[tensor.name] = np.asarray(prepare_smolvla_image(image), dtype=DTYPE)
        elif self.spec.name == "groot":
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
        actions = self.client.decode_actions(raw_actions)
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


# -- PoLiMa's own SO-101 loop (polima.robot.so101_loop) ---------------------

PATCH = 14  # GR00T vision patch side, in pixels


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
        if self.spec.name == "act":
            return self.client.predict(images, state)
        if self.spec.name != "groot":
            raise ValueError(f"the robot client does not run {self.spec.display_name} yet")
        tensors: dict[str, np.ndarray] = {"state": state}
        wire = self.spec.wire.request_tensors
        for tensor, image in zip((t for t in wire if t.name.startswith("patches")), images):
            tensors[tensor.name] = groot_patches(image, self.context)
        noise = next((t for t in wire if t.name == "noise"), None)
        if noise is not None:  # flow matching starts each plan from fresh noise
            tensors["noise"] = self.rng.standard_normal(noise.shape).astype(DTYPE)
        return self.client.predict_raw(tensors, task_id=self.task_id)


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


def groot_patches(image: np.ndarray, context: dict[str, Any]) -> np.ndarray:
    """GR00T N1.6's evaluation transform for one camera, cut into 14x14 patches.

    Follows the checkpoint's own eval pipeline (crop, resize), then Eagle's
    bicubic resize to the compiled shape and mean/std normalization.
    """
    from PIL import Image

    config = context.get("image_preprocess") or {}
    if config.get("format") != "groot-eagle-image-v1" or config.get("color_space") != "RGB":
        raise ValueError("this GR00T bundle has no Eagle image contract; repack it")
    height, width = (int(value) for value in context["image_shape"])
    evaluation, eagle = config["gr00t_eval"], config["eagle"]
    frame = rgb(image)
    frame = (eval_albumentations if evaluation.get("use_albumentations") else eval_torchvision)(
        frame, evaluation)
    frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC))
    values = frame.astype(np.float32) * np.float32(eagle["rescale_factor"])
    values = (values - np.float32(eagle["image_mean"])) / np.float32(eagle["image_std"])
    rows, columns = height // PATCH, width // PATCH
    patches = values.reshape(rows, PATCH, columns, PATCH, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(patches.reshape(rows * columns, PATCH * PATCH * 3), dtype=DTYPE)


def eval_albumentations(frame: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    import cv2

    if config.get("letter_box_transform"):
        frame = letter_box(frame)
    target, crop = config.get("image_target_size"), config.get("image_crop_size")
    shortest = config.get("shortest_image_edge") or (target and target[0])
    fraction = config.get("crop_fraction") or (target and crop and crop[0] / target[0])
    if not shortest or not fraction:
        raise ValueError("the bundle's GR00T crop settings are incomplete")

    def resize(value: np.ndarray) -> np.ndarray:
        scale = float(shortest) / min(value.shape[:2])
        size = (round(value.shape[1] * scale), round(value.shape[0] * scale))
        return cv2.resize(value, size, interpolation=cv2.INTER_AREA)

    frame = resize(frame)
    height, width = frame.shape[:2]
    crop_h, crop_w = max(1, int(height * fraction)), max(1, int(width * fraction))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return resize(frame[top:top + crop_h, left:left + crop_w])


def eval_torchvision(frame: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms

    if config.get("letter_box_transform"):
        frame = letter_box(frame)
    target, crop = config["image_target_size"], config["image_crop_size"]
    value = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
    value = transforms.resize(value, list(target), InterpolationMode.BILINEAR, antialias=True)
    value = transforms.center_crop(value, list(crop))
    value = transforms.resize(value, list(target), InterpolationMode.BILINEAR, antialias=True)
    return value.permute(1, 2, 0).contiguous().numpy()


def letter_box(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    boxed = np.zeros((side, side, 3), dtype=frame.dtype)
    boxed[top:top + height, left:left + width] = frame
    return boxed


def rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = array.transpose(1, 2, 0)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected an RGB camera frame, got {array.shape}")
    return np.ascontiguousarray(array)
