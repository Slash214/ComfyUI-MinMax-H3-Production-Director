"""RAM / VRAM / timing diagnostics for MiniMax H3 Director."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any, Iterator

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.memory")

MEMORY_STRATEGIES = ("standard", "balanced_20gb", "aggressive_lowmem")

_NVML = {"ready": False, "failed": False, "device": None}

# Windows PDH handle for the "GPU Process Memory" counter set. Opened lazily and
# reused; used to read WDDM *shared* GPU memory, which NVML does not report.
_PDH: dict[str, Any] = {"ready": False, "failed": False}


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


def system_available_gib() -> float | None:
    avail = _system_available_bytes()
    if avail is None:
        return None
    return float(avail) / (1024.0 ** 3)


def _init_nvml() -> bool:
    if _NVML["ready"]:
        return True
    if _NVML["failed"]:
        return False
    try:
        import pynvml

        pynvml.nvmlInit()
        index = 0
        if torch.cuda.is_available():
            try:
                index = int(torch.cuda.current_device())
            except Exception:
                index = 0
        _NVML["device"] = pynvml.nvmlDeviceGetHandleByIndex(index)
        _NVML["backend"] = "pynvml"
        _NVML["ready"] = True
        return True
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            nvml = ctypes.WinDLL("nvml.dll")
            nvml.nvmlInit_v2.restype = ctypes.c_int
            if int(nvml.nvmlInit_v2()) != 0:
                raise OSError("nvmlInit_v2 failed")
            index = 0
            if torch.cuda.is_available():
                try:
                    index = int(torch.cuda.current_device())
                except Exception:
                    index = 0
            handle = ctypes.c_void_p()
            if int(nvml.nvmlDeviceGetHandleByIndex(index, ctypes.byref(handle))) != 0:
                raise OSError("nvmlDeviceGetHandleByIndex failed")
            _NVML["lib"] = nvml
            _NVML["device"] = handle
            _NVML["backend"] = "ctypes"
            _NVML["ready"] = True
            return True
        except Exception:
            pass
    _NVML["failed"] = True
    return False


def _nvml_stats() -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "nvml_used_mib": None,
        "nvml_free_mib": None,
        "nvml_total_mib": None,
    }
    if not _init_nvml():
        return out
    try:
        backend = _NVML.get("backend")
        if backend == "pynvml":
            import pynvml

            info = pynvml.nvmlDeviceGetMemoryInfo(_NVML["device"])
            out["nvml_used_mib"] = _bytes_to_mib(info.used)
            out["nvml_free_mib"] = _bytes_to_mib(info.free)
            out["nvml_total_mib"] = _bytes_to_mib(info.total)
        elif backend == "ctypes":
            import ctypes

            class nvmlMemory_t(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            mem = nvmlMemory_t()
            nvml = _NVML["lib"]
            nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            if int(nvml.nvmlDeviceGetMemoryInfo(_NVML["device"], ctypes.byref(mem))) == 0:
                out["nvml_used_mib"] = _bytes_to_mib(mem.used)
                out["nvml_free_mib"] = _bytes_to_mib(mem.free)
                out["nvml_total_mib"] = _bytes_to_mib(mem.total)
    except Exception as exc:
        log.debug("NVML memory read failed: %s", exc)
    return out


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


def _init_gpu_shared_pdh() -> bool:
    """Open a PDH query for this process's WDDM shared GPU memory (Windows only).

    NVML only reports *dedicated* VRAM. On Windows, once dedicated VRAM fills,
    the WDDM driver silently pages allocations into system RAM over PCIe — the
    generation keeps running but the whole desktop stutters. That spill is
    invisible to NVML and to the torch allocator; this counter is the only
    direct measurement of it.
    """
    if _PDH["ready"]:
        return True
    if _PDH["failed"] or sys.platform != "win32":
        _PDH["failed"] = True
        return False
    try:
        import ctypes
        from ctypes import wintypes

        pdh = ctypes.WinDLL("pdh.dll")

        query = ctypes.c_void_p()
        if int(pdh.PdhOpenQueryW(None, 0, ctypes.byref(query))) != 0:
            raise OSError("PdhOpenQueryW failed")

        # Expand "\GPU Process Memory(*)\Shared Usage" and keep our own pid's
        # instances (one per physical GPU / LUID).
        wildcard = "\\GPU Process Memory(*)\\Shared Usage"
        size = wintypes.DWORD(0)
        pdh.PdhExpandWildCardPathW(None, wildcard, None, ctypes.byref(size), 0)
        if int(size.value) <= 0:
            raise OSError("PdhExpandWildCardPathW returned no paths")
        buf = ctypes.create_unicode_buffer(int(size.value))
        if int(pdh.PdhExpandWildCardPathW(None, wildcard, buf, ctypes.byref(size), 0)) != 0:
            raise OSError("PdhExpandWildCardPathW failed")

        # MULTI_SZ: NUL-separated, double-NUL terminated. Slicing a c_wchar
        # array yields a str that keeps the embedded NULs, so split on them.
        raw = buf[: int(size.value)]
        paths = [p for p in raw.split("\x00") if p]
        needle = f"pid_{os.getpid()}_"
        mine = [p for p in paths if needle in p]
        if not mine:
            raise OSError("no GPU Process Memory instance for this pid")

        counters = []
        for path in mine:
            handle = ctypes.c_void_p()
            if int(pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(handle))) == 0:
                counters.append(handle)
        if not counters:
            raise OSError("PdhAddEnglishCounterW added no counters")

        pdh.PdhCollectQueryData(query)
        _PDH.update({"lib": pdh, "query": query, "counters": counters, "ready": True})
        return True
    except Exception as exc:
        log.debug("GPU shared-memory PDH counter unavailable: %s", exc)
        _PDH["failed"] = True
        return False


def _gpu_shared_mib() -> float | None:
    """Sum this process's WDDM shared GPU memory in MiB, or None."""
    if not _init_gpu_shared_pdh():
        return None
    try:
        import ctypes

        PDH_FMT_LARGE = 0x00000400

        class PDH_FMT_COUNTERVALUE(ctypes.Structure):
            _fields_ = [
                ("CStatus", ctypes.c_ulong),
                ("largeValue", ctypes.c_longlong),
            ]

        pdh = _PDH["lib"]
        pdh.PdhCollectQueryData(_PDH["query"])
        total = 0
        got = False
        for handle in _PDH["counters"]:
            value = PDH_FMT_COUNTERVALUE()
            rc = pdh.PdhGetFormattedCounterValue(
                handle, PDH_FMT_LARGE, None, ctypes.byref(value)
            )
            if int(rc) == 0:
                total += int(value.largeValue)
                got = True
        if not got:
            return None
        return float(total) / (1024.0 * 1024.0)
    except Exception as exc:
        log.debug("GPU shared-memory read failed: %s", exc)
        return None


def _alloc_pressure() -> dict[str, Any]:
    """Torch allocator pressure counters.

    ``num_alloc_retries`` increments whenever the caching allocator failed to
    serve a request, freed cached blocks and tried again. A run that climbs here
    is thrashing, which on Windows is the in-process fingerprint of a WDDM
    spill even when NVML still reports headroom.
    """
    out: dict[str, Any] = {"alloc_retries": None, "cuda_ooms": None}
    try:
        stats = torch.cuda.memory_stats(torch.cuda.current_device())
        out["alloc_retries"] = int(stats.get("num_alloc_retries", 0))
        out["cuda_ooms"] = int(stats.get("num_ooms", 0))
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
    snap.update(_nvml_stats())
    snap["gpu_shared_mib"] = _gpu_shared_mib()
    snap.update(_alloc_pressure())
    return snap


def loaded_models_brief() -> list[str]:
    """Names + VRAM of everything ComfyUI currently holds resident.

    Used to answer a specific question: is the ~15GB Qwen3-VL text encoder still
    resident during the refine pass, squeezing the second sampling on a 20GB
    card? ``balanced_20gb`` deliberately skips ``unload_all_models`` while RAM is
    plentiful, which helps system RAM but may hurt the refine VRAM peak.
    """
    out: list[str] = []
    try:
        import comfy.model_management as mm

        for entry in getattr(mm, "current_loaded_models", []) or []:
            model = getattr(entry, "model", None)
            name = type(model).__name__ if model is not None else type(entry).__name__
            inner = getattr(model, "model", None)
            if inner is not None:
                name = f"{name}<{type(inner).__name__}>"
            size_mib = None
            for attr in ("model_loaded_memory", "loaded_size", "model_size"):
                probe = getattr(entry, attr, None)
                try:
                    value = probe() if callable(probe) else probe
                except Exception:
                    continue
                if isinstance(value, (int, float)) and value > 0:
                    size_mib = float(value) / (1024.0 * 1024.0)
                    break
            device = getattr(entry, "device", None)
            out.append(
                f"{name}"
                + (f" {size_mib:.0f}MiB" if size_mib is not None else "")
                + (f" @{device}" if device is not None else "")
            )
    except Exception as exc:
        log.debug("loaded model dump failed: %s", exc)
    return out


def model_fingerprint(model: Any) -> str:
    """Identify which checkpoint a sampling pass actually used.

    ComfyUI logs "Model MiniMaxH3 prepared for dynamic VRAM loading. NNNNMB
    Staged" without naming the file, and the staged figure varies with free
    VRAM — so it cannot be used to tell checkpoints apart. Parameter count and
    total weight bytes can: on MiniMax H3 the W4A8-mixed and INT8-ConvRot
    checkpoints differ by several GB, which is exactly the confusion this line
    is here to prevent.
    """
    try:
        inner = getattr(model, "model", None)
        target = getattr(inner, "diffusion_model", None) or inner or model

        # state_dict() includes buffers — the int8/int4 weights and their scales.
        # parameters() alone reports the logical bf16 view, which is identical
        # across H3 checkpoints and therefore cannot tell int8-convrot from
        # W4A8-mixed. Storage bytes can.
        seen: set[int] = set()
        stored_bytes = 0
        elems = 0
        dtypes: dict[str, int] = {}
        state = getattr(target, "state_dict", None)
        items = state().items() if callable(state) else []
        for _name, tensor in items:
            if not torch.is_tensor(tensor):
                continue
            try:
                ptr = int(tensor.untyped_storage().data_ptr())
                nbytes = int(tensor.untyped_storage().nbytes())
            except Exception:
                ptr, nbytes = id(tensor), tensor.numel() * tensor.element_size()
            if ptr not in seen:
                seen.add(ptr)
                stored_bytes += nbytes
            n = int(tensor.numel())
            elems += n
            key = str(tensor.dtype).replace("torch.", "")
            dtypes[key] = dtypes.get(key, 0) + n
        if stored_bytes == 0:
            return f"{type(target).__name__} (no weights visible)"
        top = sorted(dtypes.items(), key=lambda kv: -kv[1])[:4]
        dtype_text = ", ".join(f"{k}:{v / 1e6:.0f}M" for k, v in top)
        return (
            f"{type(target).__name__} | tensors={len(seen)} | "
            f"elems={elems / 1e9:.2f}B | "
            f"stored={stored_bytes / (1024.0 ** 3):.2f}GiB | {dtype_text}"
        )
    except Exception as exc:
        return f"fingerprint failed ({type(exc).__name__})"


def latent_token_estimate(latent: Any) -> dict[str, Any]:
    """Estimate the packed video token count from a latent dict or tensor.

    H3's DiT attends over a joint packed sequence (text + conditioning + audio +
    video). The video part dominates and is what scales with resolution and
    frame count, so ``T*H*W`` of the video latent is the number that decides
    whether attention is in the cheap or the quadratic-tax regime — and how much
    a feed-forward chunking patch would save.
    """
    info: dict[str, Any] = {
        "shape": None,
        "video_tokens": None,
        "mib_bf16": None,
        "extra": [],
    }
    try:
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        if not torch.is_tensor(samples):
            return info
        shape = tuple(int(x) for x in samples.shape)
        info["shape"] = shape
        # H3 packs audio alongside video (see LTXVSeparateAVLatent). List any
        # other tensors in the dict so the log shows the whole packed sequence,
        # not just the stream under "samples".
        if isinstance(latent, dict):
            for key, value in latent.items():
                if key == "samples" or not torch.is_tensor(value):
                    continue
                info["extra"].append(f"{key}={tuple(int(x) for x in value.shape)}")

        # Token count = every dim except batch and channels. H3's AV latent is
        # not always [B,C,T,H,W] — packed layouts show up as rank 3 and 4 too —
        # so derive it generically instead of pattern-matching one rank.
        if len(shape) >= 3:
            info["channels"] = int(shape[1])
            tokens = 1
            for dim in shape[2:]:
                tokens *= int(dim)
            info["video_tokens"] = tokens
        elif len(shape) == 2:  # [tokens, features] — already packed
            info["channels"] = int(shape[1])
            info["video_tokens"] = int(shape[0])
        if info["video_tokens"]:
            # H3's first FFN projection intermediate is ~56 KiB per token.
            info["mib_bf16"] = round(info["video_tokens"] * 56.0 / 1024.0, 1)
    except Exception:
        pass
    return info


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
        self._timing_lines: list[str] = []
        self._phase_totals: dict[str, float] = {}

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
            parts.append(f"AvailRAM={avail:.1f}MiB")
        if snap.get("cuda_allocated_mib") is not None:
            parts.append(f"TorchAlloc={snap['cuda_allocated_mib']:.1f}MiB")
        if snap.get("cuda_reserved_mib") is not None:
            parts.append(f"TorchReserved={snap['cuda_reserved_mib']:.1f}MiB")
        if snap.get("cuda_max_allocated_mib") is not None:
            parts.append(f"TorchMax={snap['cuda_max_allocated_mib']:.1f}MiB")
        if snap.get("nvml_used_mib") is not None:
            parts.append(f"NVMLUsed={snap['nvml_used_mib']:.1f}MiB")
        if snap.get("nvml_free_mib") is not None:
            parts.append(f"NVMLFree={snap['nvml_free_mib']:.1f}MiB")
        if snap.get("nvml_total_mib") is not None:
            parts.append(f"NVMLTotal={snap['nvml_total_mib']:.1f}MiB")
        # WDDM spill indicators — the reason a refine pass can make the whole
        # desktop stutter while NVML still looks fine.
        if snap.get("gpu_shared_mib") is not None:
            parts.append(f"GPUShared={snap['gpu_shared_mib']:.1f}MiB")
        if snap.get("alloc_retries"):
            parts.append(f"AllocRetries={snap['alloc_retries']}")
        if snap.get("cuda_ooms"):
            parts.append(f"CudaOOMs={snap['cuda_ooms']}")
        if elapsed_s is not None:
            parts.append(f"Δt={elapsed_s:.2f}s")
        line = " | ".join(parts)
        self._lines.append(line)
        log.info(line)

    def probe(
        self,
        label: str,
        *,
        latent: Any = None,
        models: bool = False,
        model: Any = None,
    ) -> None:
        """Checkpoint plus optional latent token count and resident model dump.

        Used at the sampling boundaries so a log can answer, without guesswork:
        how many tokens each pass attends over, what is resident at that moment,
        and whether the allocator is thrashing.
        """
        if not self.enabled:
            return
        self.checkpoint(label)
        if model is not None:
            line = f"[Model] {label} | {model_fingerprint(model)}"
            self._lines.append(line)
            log.info(line)
        if latent is not None:
            info = latent_token_estimate(latent)
            if info.get("video_tokens"):
                extra = info.get("extra") or []
                line = (
                    f"[Tokens] {label} | latent={info['shape']} | "
                    f"video_tokens={info['video_tokens']:,} | "
                    f"ffn_intermediate~{info['mib_bf16']:.0f}MiB "
                    f"(chunk x2 would save ~{info['mib_bf16'] * 0.37:.0f}MiB)"
                    + (f" | other streams: {', '.join(extra)}" if extra else "")
                )
                self._lines.append(line)
                log.info(line)
        if models:
            resident = loaded_models_brief()
            line = f"[Resident] {label} | " + (
                "; ".join(resident) if resident else "(none reported)"
            )
            self._lines.append(line)
            log.info(line)

    def timing(self, label: str, *, elapsed_s: float | None = None) -> None:
        if not self.enabled:
            return
        snap = snapshot_memory()
        parts = [f"[Timing] {label}"]
        if elapsed_s is not None:
            parts.append(f"elapsed={elapsed_s:.2f}s")
        if snap.get("nvml_used_mib") is not None:
            parts.append(f"NVMLUsed={snap['nvml_used_mib']:.1f}MiB")
        if snap.get("process_rss_mib") is not None:
            parts.append(f"RSS={snap['process_rss_mib']:.1f}MiB")
        line = " | ".join(parts)
        self._timing_lines.append(line)
        log.info(line)

    def note_phase_total(self, phase: str, seconds: float) -> None:
        if not self.enabled:
            return
        self._phase_totals[phase] = float(seconds)
        self.timing(
            f"{phase} total={seconds:.2f}s "
            f"(Model Load / Sampler / Cleanup breakdown in prior [Timing] lines)",
        )

    @contextlib.contextmanager
    def watch_model_load(self, prefix: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        import comfy.model_management as mm

        original = mm.load_models_gpu
        loading = {"active": False}

        def wrapped(*args, **kwargs):
            if loading["active"]:
                return original(*args, **kwargs)
            loading["active"] = True
            self.timing(f"{prefix} Model Load Start")
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.timing(f"{prefix} Model Load End", elapsed_s=time.perf_counter() - t0)
                loading["active"] = False

        mm.load_models_gpu = wrapped
        try:
            yield
        finally:
            mm.load_models_gpu = original

    def checkpoint(self, label: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        elapsed = now - self._last_cp_t
        self._emit(label, snapshot_memory(), elapsed_s=elapsed)
        self._last_cp_t = now
        self._last_cp_label = label

    # A timed run is only comparable to another timed run that started from the
    # same place. These thresholds flag a start state that is carrying models,
    # shared-memory mappings or RAM pressure over from a previous run.
    COLD_MAX_GPU_SHARED_MIB = 2048.0
    COLD_MIN_AVAIL_RAM_MIB = 24576.0

    def _classify_start_state(self, snap: dict[str, Any]) -> str:
        reasons: list[str] = []
        shared = snap.get("gpu_shared_mib")
        avail = snap.get("system_available_mib")
        used = snap.get("nvml_used_mib")
        if shared is not None and shared > self.COLD_MAX_GPU_SHARED_MIB:
            reasons.append(f"GPUShared={shared:.0f}MiB already mapped")
        if avail is not None and avail < self.COLD_MIN_AVAIL_RAM_MIB:
            reasons.append(f"only {avail:.0f}MiB RAM free at start")
        if used is not None and used > 4096.0:
            reasons.append(f"NVMLUsed={used:.0f}MiB already resident")
        if not reasons:
            return "[Baseline] COLD start — timings are comparable to other cold runs."
        return (
            "[Baseline] WARM start — NOT comparable to a cold run ("
            + "; ".join(reasons)
            + "). Restart ComfyUI before a timed comparison."
        )

    def begin_run(self, *, plan_note: str = "") -> None:
        if not self.enabled:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self.checkpoint("Run Start")
        try:
            verdict = self._classify_start_state(snapshot_memory())
            self._lines.append(verdict)
            if "WARM" in verdict:
                log.warning("%s", verdict)
            else:
                log.info("%s", verdict)
        except Exception as exc:
            log.debug("start-state classification skipped: %s", exc)
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
        timings = "\n".join(self._timing_lines) if self._timing_lines else "(no timing marks this run)"
        body = "\n".join(self._lines)
        phase = ""
        if self._phase_totals:
            phase = "\n--- Phase totals ---\n" + "\n".join(
                f"{k}: {v:.2f}s" for k, v in self._phase_totals.items()
            )
        return (
            "\n\n=== Memory debug ===\n"
            f"memory_strategy={self.memory_strategy}\n\n"
            "--- Checkpoints ---\n"
            f"{body}\n\n"
            "--- Timing ---\n"
            f"{timings}{phase}\n\n"
            "--- Instrumented tensor copies ---\n"
            f"{copies}\n"
        )
