# LeRobot teleoperation on SiMa.ai Modalix

This directory contains a LeRobot 0.4.4 install script and hardware checks for an
SO-100/SO-101 leader–follower pair.

## Which arm is which?

- **Leader = input/controller.** Put it beside the operator and move it by hand.
- **Follower = output/robot.** Put it in the workspace; its powered motors reproduce
  the leader's pose.

The arms and USB boards may be physically identical. A device name does not reveal its
role: you assign each role by deciding which arm the operator will move. Do not use raw
`/dev/ttyACM0` and `/dev/ttyACM1` names if stable links are available, because those
names can swap after reconnecting or rebooting.

## Test and run

Install LeRobot first:

```bash
./install_lerobot_modalix.sh
```

Identify stable serial paths with `ls -l /dev/serial/by-id/`. If the boards share a
USB serial number, use `ls -l /dev/serial/by-path/` and always reconnect each arm to
the same USB socket. To identify an unknown cable, run `/media/nvme/lerobot/bin/lerobot-find-port`
and unplug only the arm you intend to name when prompted.

Run the non-moving pair test:

```bash
/media/nvme/lerobot/bin/python test_teleop_pair.py \
  --leader-port /dev/serial/by-path/LEADER_LINK \
  --follower-port /dev/serial/by-path/FOLLOWER_LINK
```

Then start teleoperation with the same paths:

```bash
./run_teleop_modalix.sh \
  /dev/serial/by-path/LEADER_LINK \
  /dev/serial/by-path/FOLLOWER_LINK
```

If you do not know the ports, omit both arguments:

```bash
./run_teleop_modalix.sh
```

The launcher will ask you to disconnect and reconnect the leader first, then the
follower. It detects which serial device disappeared and selects a stable `by-id` or
`by-path` link when one exists. Disconnect only the requested arm at each prompt.

The launcher runs the read-only pair test first and prints the role map before LeRobot
can enable follower torque. It then starts `lerobot-teleoperate`. If calibration is
missing, LeRobot's interactive calibration runs before teleoperation and calibrates
both roles as needed. Calibration is stored under the fixed IDs `modalix_leader` and
`modalix_follower`, so never reverse the two port arguments after calibration; delete
and repeat the affected calibration if you intentionally reassign the physical arms.

For a deeper electrical diagnosis of one arm, use:

```bash
/media/nvme/lerobot/bin/python check_arm.py /dev/serial/by-path/ARM_LINK
```
