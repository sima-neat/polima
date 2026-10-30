"""Spread each plan-switch jump over several moves (POLIMA_SMOOTH_MOVES=K).

When a new action plan takes over the queue, its target for the next timestep is usually
several degrees away from where the previous plan was heading (the plans are predicted
from observations ~0.5 s apart). Executed as-is, that difference is one step, often a
reversal. Here the switch keeps the last sent target plus the new plan's own movement for
that timestep, and the remaining difference to the new plan is removed in K equal parts
over the next K moves. Afterwards the new plan is followed exactly.

POLIMA_SMOOTH_SHAPE=smoothstep removes that difference with an S-shaped ease (small parts
first and last) instead of equal parts; the default is linear.

Use with POLIMA_AGGREGATE=latest_only (the default while this is on): an overlap blend
would split the switch and land its remainder after this smoothing. K should be shorter
than the moves between plan arrivals (about 7 at 15 Hz with ~0.47 s inference).

Only the five arm joints are changed; the gripper follows the plan unchanged. Queue timing,
timestep bookkeeping (latest_action), rest snapping and the max_relative_target clamp are
LeRobot's/the bundle script's and run on the smoothed target. No motor reads or writes are
added. K=0 or unset leaves the client unchanged.
"""
from __future__ import annotations

import os
import time

import numpy as np

ENV = "POLIMA_SMOOTH_MOVES"
CHUNK_ENV = "POLIMA_CHUNK_FILTER"
ARM_JOINTS = 5  # SO101 order: five arm joints, then the gripper
# A longer pause starts fresh (startup, return to rest, auto-repeat). Longer than one
# synchronous inference wait (~0.5 s), so threshold-0 runs are smoothed too.
RESET_AFTER_S = 1.0
_TAG = "_polima_plan"


SHAPE_ENV = "POLIMA_SMOOTH_SHAPE"


def shape_from_env(environ=os.environ) -> str:
    shape = environ.get(SHAPE_ENV, "linear").strip() or "linear"
    if shape not in SHAPES:
        raise ValueError(f"{SHAPE_ENV} must be one of {sorted(SHAPES)}")
    return shape


def moves_from_env(environ=os.environ) -> int:
    value = environ.get(ENV, "0").strip() or "0"
    try:
        moves = int(value)
    except ValueError:
        moves = -1
    if not 0 <= moves <= 32:
        raise ValueError(f"{ENV} must be an integer number of moves between 0 (off) and 32")
    return moves


def chunk_filter_from_env(environ=os.environ) -> str:
    value = environ.get(CHUNK_ENV, "off").strip().lower() or "off"
    if value not in ("off", "fir3"):
        raise ValueError(f"{CHUNK_ENV} must be off or fir3")
    return value


def filter_chunk(actions: np.ndarray, kind: str) -> np.ndarray:
    """Smooth arm targets inside one predicted plan; preserve endpoints and gripper."""
    values = np.asarray(actions)
    if kind == "off":
        return values
    if kind != "fir3":
        raise ValueError(f"unsupported action-chunk filter {kind!r}")
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < ARM_JOINTS + 1:
        raise ValueError("action chunk must contain at least three six-joint targets")
    result = values.copy()
    result[1:-1, :ARM_JOINTS] = (
        values[:-2, :ARM_JOINTS]
        + 4 * values[1:-1, :ARM_JOINTS]
        + values[2:, :ARM_JOINTS]
    ) / 6
    return result


def _values(action) -> np.ndarray:
    return np.asarray(action.tolist(), dtype=np.float64)


class Plan:
    """Raw targets of one arriving plan, by timestep."""

    __slots__ = ("steps",)

    def __init__(self, steps):
        self.steps = steps


SHAPES = {
    # Fraction of the switch jump still to remove after part x = (j + 1) / K.
    "linear": lambda x: 1.0 - x,
    "smoothstep": lambda x: 1.0 - x * x * (3.0 - 2.0 * x),
}


class PlanSwitchSmoother:
    """Per-command target adjustment; called only from the control-loop thread."""

    def __init__(self, moves: int, joints: int = ARM_JOINTS, reset_after_s: float = RESET_AFTER_S,
                 clock=time.monotonic, shape: str = "linear"):
        if moves < 1:
            raise ValueError("moves must be positive")
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {sorted(SHAPES)}")
        self.moves, self.joints, self.reset_after_s, self.clock = moves, joints, reset_after_s, clock
        self.remaining_fraction = SHAPES[shape]
        self.offset = None
        self.reset()

    def reset(self) -> None:
        self.last = self.last_plan = self.last_timestep = self.last_time = None
        self.remaining = 0

    def __call__(self, target: np.ndarray, plan: Plan | None, timestep: int | None) -> np.ndarray:
        now = self.clock()
        fresh = (plan is None or self.last is None
                 or now - self.last_time > self.reset_after_s)
        if fresh:
            self.remaining = 0
        elif plan is not self.last_plan:
            current, previous = plan.steps.get(timestep), plan.steps.get(self.last_timestep)
            own_step = current - previous if current is not None and previous is not None else 0.0
            # Continue from the last sent target with the new plan's own movement.
            self.offset = (self.last + own_step) - target
            self.offset[self.joints:] = 0.0
            self.remaining = self.moves
        result = target
        if self.remaining > 0:
            self.remaining -= 1
            part = (self.moves - self.remaining) / self.moves
            result = target + self.offset * self.remaining_fraction(part)
        self.last, self.last_plan, self.last_timestep, self.last_time = result, plan, timestep, now
        return result


def plan_tag(action_tensor):
    """(Plan, timestep) of a queued action tensor, or (None, None) if untagged."""
    return getattr(action_tensor, _TAG, None) or (None, None)


def install_tags(client_cls) -> None:
    """Tag every queued action tensor with its plan and timestep at merge time (once).

    Tags are set before LeRobot swaps in the new queue, so the control loop never pops an
    untagged action from it. Shared by plan-switch smoothing and sub-step sending.
    """
    if vars(client_cls).get("_polima_tags_installed"):
        return
    original_merge = client_cls._aggregate_action_queues

    def tag(tensor, value) -> None:
        try:
            setattr(tensor, _TAG, value)
        except AttributeError:  # not a tensor that accepts attributes: left untagged
            pass

    def merge(self, incoming_actions, aggregate_fn=None):
        plan = Plan({a.get_timestep(): _values(a.get_action()) for a in incoming_actions})
        for action in incoming_actions:
            tag(action.get_action(), (plan, action.get_timestep()))

        def blend(old, new):
            # Same default as LeRobot's merge when no aggregate function is given.
            result = aggregate_fn(old, new) if aggregate_fn is not None else new
            if result is not new:
                tag(result, getattr(new, _TAG, None))
            return result

        return original_merge(self, incoming_actions, blend)

    client_cls._aggregate_action_queues = merge
    client_cls._polima_tags_installed = True


def install(client_cls, moves: int, clock=time.monotonic, shape: str = "linear") -> PlanSwitchSmoother:
    """Tag queued actions with their plan at merge time; smooth when each is converted.

    Must run before the bundle's robot-client script executes: that script captures
    RobotClient._aggregate_action_queues and _action_tensor_to_action_dict at import and
    wraps them, so rest snapping sees the smoothed target and LeRobot's clamp follows.
    """
    smoother = PlanSwitchSmoother(moves, clock=clock, shape=shape)
    install_tags(client_cls)
    original_convert = client_cls._action_tensor_to_action_dict

    def convert(self, action_tensor):
        if getattr(self, "_rest_snap_latched", False):
            # The bundle script replaces this target with the rest pose and, when the
            # hold ends, clears the queue for a new cycle: continue from nothing stale.
            smoother.reset()
            return original_convert(self, action_tensor)
        plan, timestep = plan_tag(action_tensor)
        values = _values(action_tensor)
        smoothed = smoother(values, plan, timestep)
        if smoothed is values:
            return original_convert(self, action_tensor)
        return original_convert(self, action_tensor.new_tensor(smoothed))

    client_cls._action_tensor_to_action_dict = convert
    return smoother
