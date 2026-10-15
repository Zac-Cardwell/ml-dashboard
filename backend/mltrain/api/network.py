"""LAN-only access: refuse clients whose address is not on a private network.

The dashboard binds to all interfaces so phones/laptops at home can reach it, but it should never be
reachable from the internet. Your router already blocks that unless a port is forwarded; this check is a
second line of defence that rejects any request coming from a public IP address.
"""
from __future__ import annotations

import ipaddress

DEFAULT_NETWORKS = [
    "127.0.0.0/8", "::1/128",                              # this machine
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",       # private LAN ranges
    "169.254.0.0/16", "fe80::/10", "fc00::/7",             # link-local / unique-local
]


class NetworkGuard:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("lan_only", True))

    def networks(self):
        extra = self.cfg.get("allowed_networks") or []      # e.g. "100.64.0.0/10" for Tailscale
        return [ipaddress.ip_network(n, strict=False) for n in DEFAULT_NETWORKS + list(extra)]

    def allowed(self, host: str | None) -> bool:
        if not self.enabled:
            return True
        if not host:
            return False
        if host in ("testclient", "localhost"):
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return any(ip in n for n in self.networks())
