"""prepare_groot_image must stay bit-identical to the per-pixel implementation.

The lookup-table normalisation evaluates the same float32 arithmetic once per
(channel, byte) instead of once per pixel. The reference functions below compute
it per pixel; every output is compared byte for byte on random, constant and
gradient frames, for both SO-101 GR00T contracts (N1.6 Eagle, N1.7 Qwen3).
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from polima.robot import policy

EVAL = {
    "crop_fraction": 0.95, "image_crop_size": None, "image_target_size": None,
    "letter_box_transform": False, "shortest_image_edge": 256, "use_albumentations": True,
}
EAGLE_CONTEXT = {
    "image_shape": [252, 336],
    "image_preprocess": {
        "color_space": "RGB", "format": "groot-eagle-image-v1", "gr00t_eval": EVAL,
        "eagle": {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                  "rescale_factor": 0.00392156862745098, "resize_resample": "bicubic",
                  "resize_to_compiled_shape": True},
    },
}
QWEN_CONTEXT = {
    "image_shape": [256, 352],
    "image_preprocess": {
        "color_space": "RGB", "format": "groot-qwen3-image-v1",
        "gr00t_eval": dict(EVAL, image_crop_size=[230, 230], image_target_size=[256, 256]),
        "qwen": {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                 "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2},
    },
}
# A contract with unequal channel statistics, so a channel mix-up cannot pass.
SKEWED = {"image_mean": [0.481, 0.458, 0.408], "image_std": [0.269, 0.261, 0.276]}


def reference_eagle(image: np.ndarray, context: dict) -> np.ndarray:
    from PIL import Image

    from polima.policies.groot.runtime import PATCH, PATCH_CHANNELS

    preprocess = context["image_preprocess"]
    eagle = preprocess["eagle"]
    image_shape = tuple(context["image_shape"])
    transformed = policy._groot_albumentations_eval(policy._rgb_hwc(image), preprocess["gr00t_eval"])
    transformed = np.asarray(
        Image.fromarray(transformed).resize(
            (image_shape[1], image_shape[0]), resample=Image.Resampling.BICUBIC
        )
    )
    values = transformed.astype(np.float32)
    values *= np.float32(eagle["rescale_factor"])
    mean = np.asarray(eagle["image_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(eagle["image_std"], dtype=np.float32).reshape(1, 1, 3)
    values = (values - mean) / std
    height, width = values.shape[:2]
    rows, columns = height // PATCH, width // PATCH
    folded = values.reshape(rows, PATCH, columns, PATCH, 3)
    return folded.transpose(0, 2, 1, 3, 4).reshape(rows * columns, PATCH_CHANNELS)


def reference_qwen(image: np.ndarray, context: dict) -> np.ndarray:
    from torchvision.transforms.functional import InterpolationMode, resize

    config = context["image_preprocess"]
    transformed = policy._groot_albumentations_eval(policy._rgb_hwc(image), config["gr00t_eval"])
    height, width = map(int, context["image_shape"])
    qwen = config["qwen"]
    patch, temporal, merge = int(qwen["patch_size"]), int(qwen["temporal_patch_size"]), int(qwen["merge_size"])
    value = torch.from_numpy(np.ascontiguousarray(transformed.transpose(2, 0, 1)))
    value = resize(value, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
    mean = torch.tensor(qwen["image_mean"])[:, None, None] * 255.0
    std = torch.tensor(qwen["image_std"])[:, None, None] * 255.0
    value = (value - mean) / std
    gh, gw = height // patch, width // patch
    value = value[None].expand(temporal, -1, -1, -1)
    value = value.reshape(1, temporal, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    return value.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * temporal * patch * patch).numpy()


def frames() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    gradient = np.broadcast_to(np.arange(640, dtype=np.uint16)[None, :, None] * 255 // 639,
                               (480, 640, 3)).astype(np.uint8)
    return [
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), np.uint8),
        np.full((480, 640, 3), 255, np.uint8),
        np.ascontiguousarray(gradient),
        np.ascontiguousarray(gradient.transpose(2, 0, 1)),  # CHW input is accepted too
    ]


def with_stats(context: dict, key: str) -> dict:
    copy = {**context, "image_preprocess": dict(context["image_preprocess"])}
    copy["image_preprocess"][key] = {**context["image_preprocess"][key], **SKEWED}
    return copy


@pytest.mark.parametrize(
    ("context", "reference"),
    [
        (EAGLE_CONTEXT, reference_eagle),
        (with_stats(EAGLE_CONTEXT, "eagle"), reference_eagle),
        (QWEN_CONTEXT, reference_qwen),
        (with_stats(QWEN_CONTEXT, "qwen"), reference_qwen),
    ],
)
def test_prepare_groot_image_bit_identical(context: dict, reference) -> None:
    for frame in frames():
        expected = reference(frame, context)
        actual = policy.prepare_groot_image(frame, context)
        assert actual.dtype == expected.dtype == np.float32
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
