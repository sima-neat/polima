"""Send each policy target in N evenly spaced steps within its tick (POLIMA_SUBSTEPS=N).

At 15 Hz every target is held for 67 ms, so the servos move, stop and move again. With
N > 1, a tick still pops exactly one action from LeRobot's queue, but the arm receives
N - 1 linearly interpolated targets from the previously sent goal toward the new target,
spaced 1 / (fps * N) apart from the start of the tick, and then the tick's own target.
Every send goes through the robot's own send_action, so the max_relative_target clamp
reads the present position and caps each one (at 15 Hz x 2 the bus traffic equals a plain
30 Hz run). The tick's work ends (N - 1) / N of a period after it starts plus the sends,
so LeRobot's control loop keeps its 1 / fps period.

Bookkeeping stays per tick: one queue pop, one latest_action update, one value returned
to the bundle script's completion/rest-snap logic (the final send's goal), one
action_command in the action trace. Because the tick's own target is now sent later in the
tick, latest_action is set to the popped timestep before the first interpolated send
(LeRobot sets the same value after its send); otherwise a plan merging in during the wait
would keep the timestep being executed and replay it. Interpolation is skipped (the target
is sent at once, as before) for the first command, after any non-policy send (return to
rest, startup), when the previous send is older than STALE_S, or when the popped action's
timestep is unknown.
"""
from __future__ import annotations

import os
import threading
import time

from polima.robot.smoothing import install_tags, plan_tag

ENV = "POLIMA_SUBSTEPS"
STALE_S = 0.2


def substeps_from_env(environ=os.environ) -> int:
    value = environ.get(ENV, "1").strip() or "1"
    try:
        count = int(value)
    except ValueError:
        count = 0
    if not 1 <= count <= 4:
        raise ValueError(f"{ENV} must be an integer between 1 (off) and 4")
    return count


class SubstepState:
    """Per-process state; touched only by the control-loop thread."""

    def __init__(self, count: int):
        self.count = count
        self.previous = self.previous_time = None
        self.record = None  # optional metrics callback(kind, **values)
        self.local = threading.local()


def install(client_cls, count: int) -> SubstepState:
    """Install before the bundle's robot-client script runs (it captures these methods).

    - _action_tensor_to_action_dict: note the popped action's timestep (plan tag).
    - control_loop_action: mark the policy tick and its start time.
    - __init__: wrap this robot's send_action (instance attribute) to split the tick's send.
    """
    state = SubstepState(count)
    install_tags(client_cls)
    original_convert = client_cls._action_tensor_to_action_dict
    original_action = client_cls.control_loop_action
    original_init = client_cls.__init__

    def convert(self, action_tensor):
        state.local.timestep = plan_tag(action_tensor)[1]
        return original_convert(self, action_tensor)

    def control_loop_action(self, *args, **kwargs):
        state.local.tick_start = time.perf_counter()
        state.local.timestep = None
        try:
            return original_action(self, *args, **kwargs)
        finally:
            state.local.tick_start = None

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        send = self.robot.send_action

        def send_action(action, *a, **k):
            tick = getattr(state.local, "tick_start", None)
            if tick is None:  # startup, return to rest, settle: not a policy command
                state.previous = None
                return send(action, *a, **k)
            timestep, state.local.timestep = getattr(state.local, "timestep", None), None
            previous = state.previous
            if (count > 1 and previous is not None and timestep is not None
                    and set(previous) == set(action)
                    and time.perf_counter() - state.previous_time <= STALE_S):
                with self.latest_action_lock:
                    self.latest_action = timestep  # the pop is committed (see module doc)
                period = float(self.config.environment_dt) / count
                final_due = tick + (count - 1) * period
                for index in range(1, count):
                    due = tick + (index - 1) * period
                    now = time.perf_counter()
                    if now >= final_due:
                        break  # running late: go straight to the tick's own target
                    if due > now:
                        time.sleep(due - now)
                    fraction = index / count
                    send({key: previous[key] + (action[key] - previous[key]) * fraction
                          for key in action})
                    if state.record is not None:
                        state.record("substep_send", timestep=timestep, index=index,
                                     tick_offset_ms=(time.perf_counter() - tick) * 1000)
                remaining = final_due - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            result = send(action, *a, **k)
            state.previous, state.previous_time = dict(result), time.perf_counter()
            return result

        self.robot.send_action = send_action

    client_cls._action_tensor_to_action_dict = convert
    client_cls.control_loop_action = control_loop_action
    client_cls.__init__ = init
    return state
