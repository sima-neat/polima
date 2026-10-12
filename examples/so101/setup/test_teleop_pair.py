#!/usr/bin/env python3
"""Non-moving preflight test for a LeRobot SO-100/SO-101 teleoperation pair.

The LEADER is the input/controller arm: the operator moves it by hand.
The FOLLOWER is the output/robot arm: its motors copy the leader during teleop.

This script only pings and reads motors. It never enables torque or commands motion.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import scservo_sdk as scs
import serial

BAUDRATE = 1_000_000
EXPECTED_IDS = tuple(range(1, 7))
PRESENT_POSITION = 56


@dataclass
class ArmResult:
    role: str
    port: str
    resolved_port: str
    ids: list[int]
    positions: dict[int, int]
    errors: list[str]


def inspect_arm(role: str, port: str) -> ArmResult:
    resolved = os.path.realpath(port)
    result = ArmResult(role, port, resolved, [], {}, [])

    if not os.path.exists(port):
        result.errors.append(f"port does not exist: {port}")
        return result

    try:
        with serial.Serial(port, BAUDRATE, timeout=0.2):
            pass
    except Exception as exc:  # noqa: BLE001 - report whatever stops the port opening
        result.errors.append(f"cannot open port: {type(exc).__name__}: {exc}")
        return result

    handler = scs.PortHandler(port)
    if not handler.openPort():
        result.errors.append("servo SDK could not open the port")
        return result

    try:
        if not handler.setBaudRate(BAUDRATE):
            result.errors.append(f"could not set baudrate to {BAUDRATE}")
            return result

        packets = scs.PacketHandler(0)
        for motor_id in EXPECTED_IDS:
            _, comm, _ = packets.ping(handler, motor_id)
            if comm == scs.COMM_SUCCESS:
                result.ids.append(motor_id)
                value, comm, error = packets.read2ByteTxRx(
                    handler, motor_id, PRESENT_POSITION
                )
                if comm == scs.COMM_SUCCESS and error == 0:
                    result.positions[motor_id] = value
                else:
                    result.errors.append(f"motor {motor_id}: position read failed")
    finally:
        handler.closePort()

    missing = sorted(set(EXPECTED_IDS) - set(result.ids))
    if missing:
        result.errors.append(f"missing motor IDs: {missing}")
    return result


def print_result(result: ArmResult) -> None:
    explanation = (
        "INPUT arm; move this one by hand"
        if result.role == "LEADER"
        else "OUTPUT arm; this one will move under motor power"
    )
    print(f"\n{result.role}: {explanation}")
    print(f"  configured: {result.port}")
    print(f"  resolves to: {result.resolved_port}")
    print(f"  motors found: {result.ids or 'none'}")
    if result.positions:
        positions = ", ".join(f"{key}={value}" for key, value in result.positions.items())
        print(f"  positions: {positions}")
    for error in result.errors:
        print(f"  FAIL: {error}")
    if not result.errors:
        print("  PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True, help="port for the arm moved by hand")
    parser.add_argument("--follower-port", required=True, help="port for the powered robot arm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    leader_resolved = os.path.realpath(args.leader_port)
    follower_resolved = os.path.realpath(args.follower_port)
    if leader_resolved == follower_resolved:
        print("FAIL: leader and follower resolve to the same serial device", file=sys.stderr)
        return 2

    print("ROLE MAP (this script does not move either arm)")
    print(f"  LEADER   [you move it] : {args.leader_port}")
    print(f"  FOLLOWER [robot moves]  : {args.follower_port}")

    leader = inspect_arm("LEADER", args.leader_port)
    follower = inspect_arm("FOLLOWER", args.follower_port)
    print_result(leader)
    print_result(follower)

    failed = leader.errors or follower.errors
    print("\nPAIR CHECK " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
