"""Stiffer arm servos for fine motion (POLIMA_ARM_P=N).

LeRobot's SO101 follower lowers every servo's position gain (P_Coefficient) from the
Feetech default 32 to 16 on each connect, "to avoid shakiness". Robot traces show the cost
during the slow approach after the gripper opens: targets move 0.3-0.5 degrees per command
and 31-41% of commanded moves above 0.5 degrees leave the motor where it was, so the arm
sticks and then lurches. A higher gain lets the servo follow those small steps.

With N set, LeRobot's own configure runs unchanged (it still writes 16 to every motor)
and then the five arm joints get P_Coefficient N, written the same way under the same
torque-disabled block LeRobot uses. The gripper keeps LeRobot's values. Unset leaves the
client unchanged, and since LeRobot writes 16 on every connect, the next run without the
setting is back to LeRobot's value.
"""
from __future__ import annotations

import os

ENV = "POLIMA_ARM_P"
ARM_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LEROBOT_P, FEETECH_DEFAULT_P = 16, 32


def gain_from_env(environ=os.environ) -> int | None:
    value = environ.get(ENV, "").strip()
    if not value:
        return None
    try:
        gain = int(value)
    except ValueError:
        gain = 0
    if not LEROBOT_P <= gain <= FEETECH_DEFAULT_P:
        raise ValueError(f"{ENV} must be an integer from {LEROBOT_P} (LeRobot's value) "
                         f"to {FEETECH_DEFAULT_P} (the Feetech default)")
    return gain


def install(follower_cls, gain: int):
    """Wrap the follower's configure; returns a function that restores it."""
    original = follower_cls.configure

    def configure(self):
        original(self)
        with self.bus.torque_disabled():
            for motor in ARM_MOTORS:
                self.bus.write("P_Coefficient", motor, gain)
        print(f"PoLiMa: arm P_Coefficient {gain} (LeRobot {LEROBOT_P}; gripper unchanged)",
              flush=True)

    follower_cls.configure = configure
    return lambda: setattr(follower_cls, "configure", original)
