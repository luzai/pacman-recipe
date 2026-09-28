"""Local startup patches for AReaL Pacman workers."""

from __future__ import annotations

import os
import re
import socket


def _patch_hostname_resolution() -> None:
    forced_host_ip = os.environ.get("AREAL_FORCE_HOST_IP")
    if not forced_host_ip:
        return

    hostname = socket.gethostname()
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if host == hostname and family in (0, socket.AF_UNSPEC, socket.AF_INET):
            return [
                (
                    socket.AF_INET,
                    type or socket.SOCK_DGRAM,
                    proto,
                    "",
                    (forced_host_ip, port or 0),
                )
            ]
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo


def _tms_binary_from_ld_preload(ld_preload: str) -> str:
    """Return the TMS preload binary from a multi-library LD_PRELOAD value."""
    candidates = [
        entry
        for entry in re.split(r"[:\s]+", ld_preload)
        if "torch_memory_saver_hook_mode_preload" in entry
    ]
    if not candidates:
        raise RuntimeError(
            "LD_PRELOAD does not contain the torch-memory-saver preload library"
        )
    return candidates[0]


def _patch_tms_preload_resolution() -> None:
    """Let TMS coexist with AReaL's nested coreutils ``stdbuf`` wrappers."""
    if os.environ.get("TMS_INIT_ENABLE") != "1":
        return
    try:
        from torch_memory_saver.hooks.mode_preload import HookUtilModePreload
    except ImportError:
        return

    def get_path_binary(self) -> str:
        del self
        return _tms_binary_from_ld_preload(os.environ.get("LD_PRELOAD", ""))

    HookUtilModePreload.get_path_binary = get_path_binary


def _disable_cudnn_sdpa() -> None:
    """Avoid cuDNN frontend plan failures in AReaL reference log-probs."""
    enabled = (
        os.environ.get("PACMAN_DISABLE_CUDNN_SDPA", os.environ.get("MAAPACMAN_DISABLE_CUDNN_SDPA")) == "1"
        or os.environ.get("AREAL_DISABLE_CUDNN_SDPA") == "1"
    )
    if not enabled:
        return
    try:
        import torch
    except ImportError:
        return

    torch.backends.cuda.enable_cudnn_sdp(False)


_patch_hostname_resolution()
_patch_tms_preload_resolution()
_disable_cudnn_sdpa()
