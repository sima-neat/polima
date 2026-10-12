"""Network address helpers."""

from __future__ import annotations

import ipaddress
import os
import socket


def lan_address(host: str = "0.0.0.0") -> str:
    """The address other machines use to reach a server bound to `host`."""
    if host not in {"0.0.0.0", "::"}:
        return host
    connection = os.environ.get("SSH_CONNECTION", "").split()
    if len(connection) == 4:
        try:
            address = ipaddress.ip_address(connection[2])
            if not (address.is_loopback or address.is_unspecified or address.is_link_local):
                return str(address)
        except ValueError:
            pass
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))  # selects the LAN interface; nothing is sent
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()
