"""Environment / core compatibility diagnostics for MiniMax H3 Director.

Read-only. Nothing here changes generation behaviour, loads models, or mutates
ComfyUI state — every probe is wrapped so a failure degrades to "unknown"
instead of breaking a run.

Answers the questions a benchmark cannot answer by itself:

* Is ComfyUI's ``comfy_kitchen`` quantised-op CUDA backend actually enabled?
  H3's ``int8 convrot`` checkpoints have a dedicated kernel that ComfyUI gates
  behind ``torch.version.cuda >= 13``. When the gate closes it logs one warning
  and silently falls back to an emulated path — measured elsewhere at ~2.17x
  slower end to end. A benchmark taken in that state is not comparable.
* Which attention backend is really in use, and is sparse attention (Sol-Attn)
  even eligible on this GPU? The Triton kernels require SM89+ (Ada/Blackwell);
  SM86 (RTX 30-series) has no TMA and falls back to dense.
* Has another custom-node pack taken ownership of H3's ``PackedLayout``?
  Director's continuity ("段间引导") patches it and refuses to share with
  unknown wrappers.
* Does the installed ComfyUI H3 core carry known high-VRAM code paths?
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.env")

_UNKNOWN = "unknown"

# Sol-Attn Triton kernels need TMA descriptors (SM90+) or the SM89 pointer
# kernel twins. Ampere consumer cards (SM86) have neither.
_SOL_ATTN_MIN_SM = (8, 9)

# Known high-VRAM patterns to look for in the installed H3 core.
_CORE_PATTERNS = (
    ("v = v.clone()", "full value-tensor clone in the H3 attention path"),
    (".contiguous().clone()", "redundant contiguous+clone"),
    ("torch.cuda.synchronize()", "explicit GPU->CPU sync inside the model"),
)


def _safe(fn, default=_UNKNOWN):
    try:
        value = fn()
        return default if value is None else value
    except Exception:
        return default


# ---------------------------------------------------------------------------
# torch / GPU
# ---------------------------------------------------------------------------


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": _UNKNOWN,
        "torch_cuda": _UNKNOWN,
        "cuda_major": None,
        "gpu_name": _UNKNOWN,
        "sm": None,
        "sm_text": _UNKNOWN,
        "vram_total_mib": None,
        "driver": _UNKNOWN,
    }
    try:
        import torch
    except Exception:
        return info

    info["torch_version"] = str(getattr(torch, "__version__", _UNKNOWN))
    cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
    info["torch_cuda"] = str(cuda_ver) if cuda_ver else "cpu-only"
    if cuda_ver:
        try:
            info["cuda_major"] = int(str(cuda_ver).split(".")[0])
        except (ValueError, IndexError):
            pass

    if not _safe(lambda: torch.cuda.is_available(), False):
        return info

    def _props():
        return torch.cuda.get_device_properties(torch.cuda.current_device())

    props = _safe(_props, None)
    if props is not None:
        info["gpu_name"] = str(getattr(props, "name", _UNKNOWN))
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        if major is not None and minor is not None:
            info["sm"] = (int(major), int(minor))
            info["sm_text"] = f"sm_{major}{minor}"
        total = getattr(props, "total_memory", None)
        if total:
            info["vram_total_mib"] = round(float(total) / (1024.0 * 1024.0), 1)

    # Driver version via NVML (reuse memory_debug's initialised handle if any).
    def _driver():
        import pynvml

        raw = pynvml.nvmlSystemGetDriverVersion()
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    info["driver"] = _safe(_driver)
    return info


# ---------------------------------------------------------------------------
# comfy_kitchen quantised-op backend
# ---------------------------------------------------------------------------


def comfy_kitchen_info() -> dict[str, Any]:
    """Report whether H3's int8-convrot CUDA kernels are live.

    ComfyUI gates this on ``torch.version.cuda >= 13`` in ``comfy/quant_ops.py``
    and disables the backend with a single warning line when the gate closes.
    """
    info: dict[str, Any] = {
        "present": False,
        "cuda_backend": _UNKNOWN,
        "native_ops": [],
        "emulated_ops": [],
        "verdict": _UNKNOWN,
        "detail": "",
    }

    try:
        import comfy_kitchen as ck
    except Exception as exc:
        info["detail"] = f"comfy_kitchen not importable ({type(exc).__name__})"
        # Fall through to the version gate check — still informative.
        _apply_cuda_gate_verdict(info)
        return info

    info["present"] = True
    registry = getattr(ck, "registry", None)

    # The registry API is not stable across ComfyUI versions — probe defensively.
    for attr in ("is_disabled", "disabled"):
        probe = getattr(registry, attr, None)
        if probe is None:
            continue
        try:
            disabled = probe("cuda") if callable(probe) else ("cuda" in probe)
            info["cuda_backend"] = "disabled" if disabled else "enabled"
            break
        except Exception:
            continue

    for attr, key in (("native_ops", "native_ops"), ("emulated_ops", "emulated_ops")):
        value = getattr(registry, attr, None) or getattr(ck, attr, None)
        if value is None:
            continue
        try:
            info[key] = sorted(str(x) for x in (value() if callable(value) else value))
        except Exception:
            continue

    _apply_cuda_gate_verdict(info)
    return info


def _apply_cuda_gate_verdict(info: dict[str, Any]) -> None:
    """Cross-check the registry state against the torch CUDA version gate."""
    cuda_major = torch_info().get("cuda_major")
    native = [str(x).lower() for x in info.get("native_ops") or []]
    has_convrot = any("convrot" in x for x in native)

    if info.get("cuda_backend") == "enabled" or has_convrot:
        info["verdict"] = "OK — H3 quantised CUDA kernels are live"
        return
    if info.get("cuda_backend") == "disabled":
        info["verdict"] = (
            "DEGRADED — comfy_kitchen cuda backend is disabled; H3 int8-convrot "
            "runs on the emulated path"
        )
        return
    if cuda_major is not None and cuda_major < 13:
        info["verdict"] = (
            f"LIKELY DEGRADED — torch reports CUDA {cuda_major}.x; ComfyUI "
            "disables the comfy_kitchen cuda backend below CUDA 13. Check the "
            "ComfyUI startup log for 'You need pytorch with cu130 or higher'."
        )
        return
    info["verdict"] = (
        "UNVERIFIED — could not read the backend state. Search the ComfyUI "
        "startup log for 'comfy_kitchen' and check disabled:True/False."
    )


# ---------------------------------------------------------------------------
# attention backend / Sol-Attn eligibility
# ---------------------------------------------------------------------------


def attention_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "backend": _UNKNOWN,
        "sage_available": False,
        "triton_version": _UNKNOWN,
        "sol_attn_packs": [],
        "sol_attn_eligible": False,
        "sol_attn_note": "",
    }

    def _backend():
        import comfy.ldm.modules.attention as attn

        fn = getattr(attn, "optimized_attention", None)
        return getattr(fn, "__name__", None) or type(fn).__name__

    info["backend"] = _safe(_backend)

    try:
        import sageattention  # noqa: F401

        info["sage_available"] = True
    except Exception:
        pass

    def _triton():
        import triton

        return str(getattr(triton, "__version__", None))

    info["triton_version"] = _safe(_triton)

    # Which Sol-Attn packs are installed (they register under custom_nodes).
    for name in (
        "ComfyUI-SolAttn_triton",
        "ComfyUI-sol-attn",
        "ComfyUI-SolAttn",
        "ComfyUI_sol-attn_Blackwell",
    ):
        for mod in list(sys.modules):
            if name.lower().replace("-", "_") in mod.lower().replace("-", "_"):
                info["sol_attn_packs"].append(name)
                break

    sm = torch_info().get("sm")
    if sm is None:
        info["sol_attn_note"] = "GPU compute capability unknown"
    elif sm >= _SOL_ATTN_MIN_SM:
        info["sol_attn_eligible"] = True
        info["sol_attn_note"] = (
            f"sm_{sm[0]}{sm[1]} meets the SM89+ requirement — Sol-Attn sparse "
            "attention kernels can run"
        )
    else:
        info["sol_attn_note"] = (
            f"sm_{sm[0]}{sm[1]} is below SM89 — Sol-Attn Triton kernels need TMA "
            "(SM90+) or the SM89 pointer twins. Sparse attention will fall back "
            "to dense; do not expect a speedup from installing it."
        )
    return info


# ---------------------------------------------------------------------------
# PackedLayout ownership (continuity compatibility)
# ---------------------------------------------------------------------------


def packed_layout_info() -> dict[str, Any]:
    info: dict[str, Any] = {"owner": _UNKNOWN, "module": "", "qualname": "", "note": ""}
    try:
        import comfy.ldm.minimax.model as mm
    except Exception as exc:
        info["note"] = f"MiniMax H3 core model not importable ({type(exc).__name__})"
        return info

    init = getattr(getattr(mm, "PackedLayout", None), "__init__", None)
    if init is None:
        info["note"] = "PackedLayout not found in the installed core"
        return info

    info["module"] = str(getattr(init, "__module__", ""))
    info["qualname"] = str(getattr(init, "__qualname__", ""))

    # Marker names must stay in sync with director/h3_context_patches.py.
    if getattr(init, "_h3_director_continuity_layout_patch", False):
        info["owner"] = "director"
        info["note"] = "Director continuity owns PackedLayout (expected)"
    elif getattr(init, "_h3_motion_context_layout_patch", False):
        info["owner"] = "h3_motion_context"
        info["note"] = (
            "ComfyUI-H3-Motion-Context owns PackedLayout — Director continuity "
            "will refuse to start. Disable one of the two packs."
        )
    elif "solattn" in info["module"].lower().replace("-", "").replace("_", ""):
        info["owner"] = "solattn"
        info["note"] = (
            "SolAttn observes PackedLayout. Director continuity can compose with "
            "it only if the upstream compatibility fix is present."
        )
    elif info["module"].startswith("comfy."):
        info["owner"] = "stock"
        info["note"] = "Unpatched stock core"
    else:
        info["owner"] = "foreign"
        info["note"] = (
            f"An unrecognised pack patched PackedLayout ({info['module']}). "
            "Director continuity will raise on start."
        )
    return info


# ---------------------------------------------------------------------------
# core regression scan
# ---------------------------------------------------------------------------


def core_scan_info() -> dict[str, Any]:
    """Grep the installed H3 core for known high-VRAM patterns. Read-only."""
    info: dict[str, Any] = {"path": _UNKNOWN, "findings": [], "note": ""}
    try:
        import comfy.ldm.minimax.model as h3_model

        path = getattr(h3_model, "__file__", None)
    except Exception as exc:
        info["note"] = f"H3 core model not importable ({type(exc).__name__})"
        return info

    if not path or not os.path.isfile(path):
        info["note"] = "H3 core model file not found on disk"
        return info

    info["path"] = path
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as exc:
        info["note"] = f"could not read core model ({type(exc).__name__})"
        return info

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, why in _CORE_PATTERNS:
            if pattern in stripped:
                info["findings"].append(f"L{lineno}: {stripped[:90]}  <- {why}")

    if not info["findings"]:
        info["note"] = "no known high-VRAM patterns found"
    return info


# ---------------------------------------------------------------------------
# launch flags
# ---------------------------------------------------------------------------


def launch_flags_info() -> dict[str, Any]:
    info: dict[str, Any] = {"vram_mode": _UNKNOWN, "flags": [], "pinned_memory": _UNKNOWN}
    try:
        import comfy.cli_args as cli

        args = getattr(cli, "args", None)
        if args is None:
            return info
        for name in (
            "normalvram",
            "highvram",
            "lowvram",
            "novram",
            "cpu",
        ):
            if getattr(args, name, False):
                info["vram_mode"] = name
        for name in (
            "use_sage_attention",
            "use_flash_attention",
            "use_pytorch_cross_attention",
            "disable_smart_memory",
            "cache_none",
            "fast",
        ):
            if getattr(args, name, False):
                info["flags"].append(name)
        disabled_pinned = getattr(args, "disable_pinned_memory", None)
        if disabled_pinned is not None:
            info["pinned_memory"] = "disabled" if disabled_pinned else "enabled"
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def collect() -> dict[str, Any]:
    return {
        "torch": torch_info(),
        "kitchen": comfy_kitchen_info(),
        "attention": attention_info(),
        "layout": packed_layout_info(),
        "core": core_scan_info(),
        "launch": launch_flags_info(),
    }


def format_report(data: dict[str, Any] | None = None) -> str:
    d = data or collect()
    t = d["torch"]
    k = d["kitchen"]
    a = d["attention"]
    lay = d["layout"]
    core = d["core"]
    launch = d["launch"]

    lines: list[str] = []
    add = lines.append

    add("=== MiniMax H3 Director — Environment Report ===")
    add("")
    add("--- Hardware / torch ---")
    add(f"GPU            : {t['gpu_name']} ({t['sm_text']})")
    if t["vram_total_mib"]:
        add(f"VRAM           : {t['vram_total_mib']:.0f} MiB")
    add(f"Driver         : {t['driver']}")
    add(f"torch          : {t['torch_version']} / CUDA {t['torch_cuda']}")
    add("")

    add("--- comfy_kitchen (H3 int8-convrot kernels) ---")
    add(f"backend        : {k['cuda_backend']}")
    if k["native_ops"]:
        add(f"native ops     : {', '.join(k['native_ops'])}")
    if k["emulated_ops"]:
        add(f"emulated ops   : {', '.join(k['emulated_ops'])}")
    add(f"VERDICT        : {k['verdict']}")
    if k["detail"]:
        add(f"detail         : {k['detail']}")
    add("")

    add("--- Attention ---")
    add(f"backend        : {a['backend']}")
    add(f"SageAttention  : {'available' if a['sage_available'] else 'not installed'}")
    add(f"Triton         : {a['triton_version']}")
    add(f"Sol-Attn packs : {', '.join(a['sol_attn_packs']) or 'none installed'}")
    add(f"Sol-Attn ready : {a['sol_attn_eligible']} — {a['sol_attn_note']}")
    add("")

    add("--- H3 PackedLayout ownership (continuity) ---")
    add(f"owner          : {lay['owner']}")
    if lay["module"]:
        add(f"module         : {lay['module']}")
    add(f"note           : {lay['note']}")
    add("")

    add("--- Core regression scan ---")
    add(f"core file      : {core['path']}")
    if core["findings"]:
        for finding in core["findings"]:
            add(f"  ! {finding}")
    else:
        add(f"  {core['note']}")
    add("")

    add("--- ComfyUI launch ---")
    add(f"vram mode      : {launch['vram_mode']}")
    add(f"pinned memory  : {launch['pinned_memory']}")
    add(f"flags          : {', '.join(launch['flags']) or 'none detected'}")

    return "\n".join(lines)


_LOGGED_ONCE = {"done": False}


def log_report_once() -> str:
    """Emit the report to the ComfyUI console once per process. Returns the text."""
    text = ""
    try:
        text = format_report()
    except Exception as exc:
        log.warning("Environment report failed: %s", exc)
        return ""
    if not _LOGGED_ONCE["done"]:
        _LOGGED_ONCE["done"] = True
        for line in text.splitlines():
            log.info("%s", line)
    return text
