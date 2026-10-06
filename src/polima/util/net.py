"""Network address helpers."""

from __future__ import annotations

import socket


def lan_address(host: str = "0.0.0.0") -> str:
    """The address other machines use to reach a server bound to `host`."""
    if host not in {"0.0.0.0", "::"}:
        return host
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))  # selects the LAN interface; nothing is sent
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()
