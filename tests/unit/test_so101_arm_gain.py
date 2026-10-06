from contextlib import contextmanager

import pytest

from polima.robot import arm_gain

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class Bus:
    def __init__(self):
        self.log, self.torque = [], True
        self.motors = {name: None for name in MOTORS}

    @contextmanager
    def torque_disabled(self):
        self.torque = False
        self.log.append("torque off")
        try:
            yield
        finally:
            self.torque = True
            self.log.append("torque on")

    def write(self, register, motor, value):
        assert not self.torque, "EEPROM writes need torque disabled"
        self.log.append((register, motor, value))


class Follower:
    """The configure of the board's LeRobot 0.4.4 SOFollower, on a fake bus."""

    def __init__(self):
        self.bus = Bus()

    def configure(self):
        with self.bus.torque_disabled():
            for motor in self.bus.motors:
                self.bus.write("P_Coefficient", motor, 16)


def test_arm_gain_runs_lerobot_configure_then_sets_only_arm_joints():
    restore = arm_gain.install(Follower, 24)
    try:
        robot = Follower()
        robot.configure()
    finally:
        restore()
    log = robot.bus.log
    lerobot, ours = log[:len(MOTORS) + 2], log[len(MOTORS) + 2:]
    assert lerobot == ["torque off", *[("P_Coefficient", m, 16) for m in MOTORS], "torque on"]
    assert ours == ["torque off", *[("P_Coefficient", m, 24) for m in MOTORS[:5]], "torque on"]
    assert robot.bus.torque  # left enabled, as LeRobot leaves it
    # Restored: a later connect is LeRobot's alone.
    plain = Follower()
    plain.configure()
    assert ("P_Coefficient", "shoulder_pan", 24) not in plain.bus.log


def test_arm_gain_setting_validation():
    assert arm_gain.gain_from_env({}) is None
    assert arm_gain.gain_from_env({"POLIMA_ARM_P": " "}) is None
    assert arm_gain.gain_from_env({"POLIMA_ARM_P": "16"}) == 16
    assert arm_gain.gain_from_env({"POLIMA_ARM_P": "32"}) == 32
    for value in ("15", "33", "0", "x", "24.5"):
        with pytest.raises(ValueError, match="POLIMA_ARM_P"):
            arm_gain.gain_from_env({"POLIMA_ARM_P": value})
