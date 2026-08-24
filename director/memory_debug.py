"""RAM / VRAM / timing diagnostics for MiniMax H3 Director (Phase 1: observe only)."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.memory")

MEMORY_STRATEGIES = ("standard", "balanced_20gb", "aggressive_lowmem")


def normalize_memory_strategy(value: str | None) -> str:
    raw = str(value or "standard").strip().lower()
    if raw in MEMORY_STRATEGIES:
        return raw
    return "standard"


def _bytes_to_mib(n: int | float | None) -> float | None:
    if n is None:
        return None
    try:
        return float(n) / (1024.0 * 1024.0)
    except (TypeError, ValueError):
        return None


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def _system_available_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    return None


def _cuda_stats() -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "cuda_allocated_mib": None,
        "cuda_reserved_mib": None,
        "cuda_max_allocated_mib": None,
    }
    if not torch.cuda.is_available():
        return out
    try:
        device = torch.cuda.current_device()
        out["cuda_allocated_mib"] = _bytes_to_mib(torch.cuda.memory_allocated(device))
        out["cuda_reserved_mib"] = _bytes_to_mib(torch.cuda.memory_reserved(device))
        try:
            out["cuda_max_allocated_mib"] = _bytes_to_mib(
                torch.cuda.max_memory_allocated(device)
            )
        except Exception:
            pass
    except Exception:
        pass
    return out


def snapshot_memory() -> dict[str, Any]:
    rss = _process_rss_bytes()
    avail = _system_available_bytes()
    snap: dict[str, Any] = {
        "process_rss_mib": _bytes_to_mib(rss),
        "system_available_mib": _bytes_to_mib(avail),
    }
    snap.update(_cuda_stats())
    return snap


def _tensor_brief(value: Any) -> str:
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    dev = str(value.device)
    dtype = str(value.dtype).replace("torch.", "")
    elem = int(value.numel()) * int(value.element_size())
    return f"Tensor({tuple(value.shape)} {dtype} {dev} ~{_bytes_to_mib(elem):.1f}MiB)"


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        return int(a.data_ptr()) == int(b.data_ptr()) and int(a.storage_offset()) == int(
            b.storage_offset()
        )
    except Exception:
        return a is b


def analyze_tensor_copy(
    *,
    label: str,
    op: str,
    src: Any,
    result: Any,
) -> dict[str, Any]:
    """Classify whether an op likely allocated a new buffer (diagnostic only)."""
    info: dict[str, Any] = {
        "label": label,
        "op": op,
        "allocates_new_buffer": False,
        "estimated_new_mib": 0.0,
        "same_object": result is src,
        "same_storage": False,
        "src": _tensor_brief(src),
        "result": _tensor_brief(result),
        "note": "",
    }
    if not isinstance(result, torch.Tensor):
        info["note"] = "result is not a tensor"
        return info
    if isinstance(src, torch.Tensor):
        info["same_storage"] = _same_storage(src, result)
        if info["same_object"] or info["same_storage"]:
            info["note"] = "view or no-op; no new pixel buffer expected"
            return info
    info["allocates_new_buffer"] = True
    info["estimated_new_mib"] = float(_bytes_to_mib(result.numel() * result.element_size()) or 0.0)
    if isinstance(src, torch.Tensor) and src.device != result.device:
        info["note"] = "device transfer likely copied bytes"
    elif isinstance(src, torch.Tensor) and src.dtype != result.dtype:
        info["note"] = "dtype conversion likely copied bytes"
    else:
        info["note"] = "distinct storage; treat as new allocation"
    return info


class DirectorMemoryDebug:
    """Checkpoint logger for one Director execute run. No-op when disabled."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        memory_strategy: str = "standard",
        node_id: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.memory_strategy = normalize_memory_strategy(memory_strategy)
        self.node_id = node_id
        self._run_t0 = time.perf_counter()
        self._segment_t0: float | None = None
        self._last_cp_t = self._run_t0
        self._last_cp_label = "run_start"
        self._segment_index = 0
        self._segment_total = 0
        self._lines: list[str] = []
        self._copy_lines: list[str] = []

    def _prefix(self) -> str:
        nid = f" node={self.node_id}" if self.node_id else ""
        return f"[Memory]{nid}"

    def _emit(self, label: str, snap: dict[str, Any], *, elapsed_s: float | None) -> None:
        parts = [
            f"{self._prefix()} {label}",
            f"RSS={snap.get('process_rss_mib'):.1f}MiB"
            if snap.get("process_rss_mib") is not None
            else f"{self._prefix()} {label} RSS=n/a",
        ]
        avail = snap.get("system_available_mib")
        if avail is not None:
            parts.append(f"avail={avail:.1f}MiB")
        if snap.get("cuda_allocated_mib") is not None:
            parts.append(f"CUDA alloc={snap['cuda_allocated_mib']:.1f}MiB")
        if snap.get("cuda_reserved_mib") is not None:
            parts.append(f"reserved={snap['cuda_reserved_mib']:.1f}MiB")
        if snap.get("cuda_max_allocated_mib") is not None:
            parts.append(f"max={snap['cuda_max_allocated_mib']:.1f}MiB")
        if elapsed_s is not None:
            parts.append(f"Δt={elapsed_s:.2f}s")
        line = " | ".join(parts)
        self._lines.append(line)
        log.info(line)

    def checkpoint(self, label: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        elapsed = now - self._last_cp_t
        self._emit(label, snapshot_memory(), elapsed_s=elapsed)
        self._last_cp_t = now
        self._last_cp_label = label

    def begin_run(self, *, plan_note: str = "") -> None:
        if not self.enabled:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self.checkpoint("Run Start")
        if plan_note:
            log.info("%s strategy=%s | %s", self._prefix(), self.memory_strategy, plan_note)

    def begin_segment(self, seg_index: int, seg_total: int) -> None:
        if not self.enabled:
            return
        self._segment_index = int(seg_index)
        self._segment_total = int(seg_total)
        self._segment_t0 = time.perf_counter()
        self._last_cp_t = self._segment_t0
        self.checkpoint(
            f"Segment {seg_index + 1}/{seg_total} Start",
        )

    def finish_segment(self) -> None:
        if not self.enabled:
            return
        self.checkpoint(
            f"Segment {self._segment_index + 1}/{self._segment_total} End",
        )
        if self._segment_t0 is not None:
            total = time.perf_counter() - self._segment_t0
            log.info(
                "%s Segment %d/%d total elapsed=%.2fs",
                self._prefix(),
                self._segment_index + 1,
                self._segment_total,
                total,
            )

    def finish_run(self) -> None:
        if not self.enabled:
            return
        total = time.perf_counter() - self._run_t0
        self.checkpoint("Run End")
        log.info("%s Run total elapsed=%.2fs", self._prefix(), total)

    def track_copy(
        self,
        *,
        label: str,
        op: str,
        src: Any,
        result: Any,
    ) -> None:
        if not self.enabled:
            return
        info = analyze_tensor_copy(label=label, op=op, src=src, result=result)
        flag = "ALLOC" if info["allocates_new_buffer"] else "reuse"
        line = (
            f"[MemoryCopy] {label} | {op} | {flag} "
            f"~{info['estimated_new_mib']:.1f}MiB | {info['note']} | "
            f"src={info['src']} -> dst={info['result']}"
        )
        self._copy_lines.append(line)
        log.info(line)

    def report_section(self) -> str:
        if not self.enabled or not self._lines:
            return ""
        copies = "\n".join(self._copy_lines) if self._copy_lines else "(no instrumented copy sites this run)"
        body = "\n".join(self._lines)
        return (
            "\n\n=== Memory debug (Phase 1, observe-only) ===\n"
            f"memory_strategy={self.memory_strategy} (no behavior change in Phase 1)\n\n"
            "--- Checkpoints ---\n"
            f"{body}\n\n"
            "--- Instrumented tensor copies ---\n"
            f"{copies}\n"
        )
