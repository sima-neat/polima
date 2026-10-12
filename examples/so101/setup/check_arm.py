#!/usr/bin/env python3
"""Walk the arm connection chain and stop at the first thing that is actually broken.

Run with the venv interpreter so lerobot's own deps are importable:

    /media/nvme/lerobot/bin/python check_arm.py [/dev/ttyACM0]

The chain has four independent links, and they fail for completely different
reasons. Testing them in order matters: a silent servo bus looks identical whether
the cause is an unpowered arm, a missing dialout group, or a wrong baudrate, and
only the order tells them apart.

  1. USB   -- the driver board enumerates (USB bus power alone is enough for this)
  2. PERMS -- this user may open the tty (dialout group)
  3. BUS   -- a servo actually answers a ping (needs the ARM's own power supply)
  4. COMMS -- reads/writes round-trip, so the link is usable and not just alive
"""
import sys

FAIL = []


def hdr(n, name):
    print(f"\n[{n}] {name}")


def ok(msg):
    print(f"    PASS  {msg}")


def bad(msg, *hints):
    print(f"    FAIL  {msg}")
    for h in hints:
        print(f"          -> {h}")
    FAIL.append(msg)


PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"

# ---------------------------------------------------------------- 1. USB ----
hdr(1, "USB enumeration -- is the driver board visible at all?")
from serial.tools import list_ports

usb = [p for p in list_ports.comports() if p.vid]
for p in usb:
    print(f"    {p.device}  {p.description}  {p.vid:04x}:{p.pid:04x}  SER={p.serial_number}")
if not usb:
    bad(
        "no USB serial adapter enumerated",
        "Check the USB cable between the SOM and the driver board.",
        "`lsusb` should list a 1a86: device for a CH34x board.",
    )
    print("\nStopping: nothing downstream can pass.")
    sys.exit(1)
ok(f"{len(usb)} USB serial adapter(s)")
if len(usb) < 2:
    print("    NOTE  only one board present; teleoperation needs two (leader + follower)")

# -------------------------------------------------------------- 2. PERMS ----
hdr(2, f"Permissions -- can this user open {PORT}?")
import grp
import os

import serial

if not os.path.exists(PORT):
    bad(f"{PORT} does not exist", "Use one of the devices listed above.")
    sys.exit(1)
try:
    s = serial.Serial(PORT, 1000000, timeout=0.2)
    s.close()
    ok(f"{PORT} opens")
except PermissionError:
    groups = [g.gr_name for g in grp.getgrall() if os.getlogin() in g.gr_mem]
    bad(
        f"permission denied on {PORT} (groups: {groups})",
        "sudo usermod -aG dialout $USER   # then log out and back in",
    )
    sys.exit(1)
except Exception as e:  # noqa: BLE001 - report whatever stops the port opening
    bad(f"{type(e).__name__}: {e}", "Something else may hold the port (ModemManager?).")
    sys.exit(1)

# ---------------------------------------------------------------- 3. BUS ----
hdr(3, "Servo bus -- does anything answer a ping?")
import scservo_sdk as scs

BAUDS = (1000000, 500000, 250000, 128000, 115200, 57600)
hits = {}
for baud in BAUDS:
    ph = scs.PortHandler(PORT)
    if not ph.openPort():
        continue
    ph.setBaudRate(baud)
    pk = scs.PacketHandler(0)
    found = [sid for sid in range(1, 21) if pk.ping(ph, sid)[1] == scs.COMM_SUCCESS]
    ph.closePort()
    if found:
        hits[baud] = found
        print(f"    baud {baud:>7}: ids {found}")

if not hits:
    bad(
        "no servo replied at any baudrate -- the bus is silent",
        "MOST LIKELY: the arm's external power supply is not connected. USB powers",
        "the driver board but NOT the servos, so step 1 passes with a dead bus.",
        "Then check the 3-pin cable from the board into the first servo,",
        "and that the daisy-chain is unbroken to the last servo.",
    )
    print("\nStopping: cannot test comms without a servo.")
    sys.exit(1)

baud = max(hits, key=lambda b: len(hits[b]))
ids = hits[baud]
ok(f"{len(ids)} servo(s) at {baud} baud: {ids}")
if baud != 1000000:
    print("    NOTE  lerobot expects 1000000 for SO-100/SO-101; re-flash IDs/baud with")
    print("          lerobot-setup-motors if this arm is not at 1 Mbaud")
if len(ids) != 6:
    print(f"    NOTE  SO-101 has 6 servos (ids 1-6); found {len(ids)}. A gap usually means")
    print("          a break in the chain right after the last id that answered.")

# -------------------------------------------------------------- 4. COMMS ----
hdr(4, "Comms -- does a real read round-trip?")
ph = scs.PortHandler(PORT)
ph.openPort()
ph.setBaudRate(baud)
pk = scs.PacketHandler(0)
STS_PRESENT_POSITION = 56
for sid in ids:
    val, comm, err = pk.read2ByteTxRx(ph, sid, STS_PRESENT_POSITION)
    if comm == scs.COMM_SUCCESS:
        ok(f"id {sid}: present_position = {val}")
    else:
        bad(f"id {sid}: ping works but read failed ({pk.getTxRxResult(comm)})",
            "Usually marginal wiring or a baudrate mismatch under load.")
ph.closePort()

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED"))
sys.exit(1 if FAIL else 0)
