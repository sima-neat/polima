from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class StudioState(str, Enum):
    IDLE = "idle"
    PREVIEWING = "previewing"
    ARMING = "arming"
    RUNNING = "running"
    STOPPING = "stopping"
    CALIBRATING = "calibrating"
    BENCHMARKING = "benchmarking"
    FAULT = "fault"


@dataclass(frozen=True)
class RunConfig:
    bundle: str
    task: str
    robot_port: str
    overhead_camera: str
    wrist_camera: str
    preview: bool = True
    repeat: bool = True
    #: None selects the policy default; GR00T disables the gripper heuristic.
    autocomplete: bool | None = None
    speed: str = "fast"
    fps: int | None = None
    actions_per_chunk: int | None = None
    max_relative_target: float | None = None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> RunConfig:
        required = ("bundle", "task", "robot_port", "overhead_camera", "wrist_camera")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError("missing run configuration: " + ", ".join(missing))
        speed = value.get("speed", "slow" if value.get("fps") == 15 else "fast")
        if speed not in ("fast", "slow"):
            raise ValueError("speed must be fast or slow")
        fps = value.get("fps")
        actions = value.get("actions_per_chunk")
        target = value.get("max_relative_target")
        if fps is not None and not 1 <= int(fps) <= 60:
            raise ValueError("fps must be between 1 and 60")
        if actions is not None and not 1 <= int(actions) <= 1000:
            raise ValueError("actions_per_chunk must be between 1 and 1000")
        if target is not None and not 0 < float(target) <= 90:
            raise ValueError("max_relative_target must be between 0 and 90")
        return cls(
            bundle=str(value["bundle"]),
            task=str(value["task"]).strip(),
            robot_port=str(value["robot_port"]),
            overhead_camera=str(value["overhead_camera"]),
            wrist_camera=str(value["wrist_camera"]),
            preview=bool(value.get("preview", True)),
            repeat=bool(value.get("repeat", True)),
            autocomplete=bool(value["autocomplete"]) if value.get("autocomplete") is not None else None,
            speed=speed,
            fps=int(fps) if fps is not None else None,
            actions_per_chunk=int(actions) if actions is not None else None,
            max_relative_target=float(target) if target is not None else None,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
