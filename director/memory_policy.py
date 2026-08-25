"""Phase 2 memory policy helpers. ``standard`` keeps official behavior."""

from __future__ import annotations

import gc
import logging
from typing import Any

from .memory_debug import normalize_memory_strategy, system_available_gib
from .refine_pack import refine_needs_canvas, refine_uses_h3_latent

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.memory_policy")

# Central RAM pressure thresholds (GiB available physical memory).
RAM_KEEP_HOT_GIB = 12.0
RAM_RELEASE_CACHE_GIB = 8.0
RAM_AGGRESSIVE_GIB = 5.0

W4A8_REFINE_RECOMMENDATION = (
    "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"
)
INT8_REFINE_HINTS = (
    "int8_convrot",
    "int8",
)


def is_balanced_strategy(strategy: str | None) -> bool:
    return normalize_memory_strategy(strategy) == "balanced_20gb"


def can_defer_pre_refine_decode(
    *,
    will_refine: bool,
    trim_frames: int,
    pack: dict[str, Any] | None,
) -> bool:
    """True when first-pass RGB is not needed before refine/upscale."""
    if not will_refine:
        return False
    if int(trim_frames or 0) > 0:
        return False
    if not isinstance(pack, dict):
        return False
    if refine_needs_canvas(pack) and not refine_uses_h3_latent(pack):
        return False
    return True


def soft_release_workspace(*, strategy: str, reason: str) -> None:
    """GC + CUDA cache trim without unloading ComfyUI models."""
    if not is_balanced_strategy(strategy):
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception as exc:
        log.debug("balanced soft release skipped (%s): %s", reason, exc)
        return
    log.debug("MiniMax H3 Director balanced soft release (%s)", reason)


def release_text_encoder(clip, *, strategy: str) -> str:
    """Drop the H3 text encoder once conditioning is done. Returns a report note.

    Measured on a 20GB card: at the moment first sampling starts, the Qwen3-VL
    text encoder is still fully resident (~15GB) while the 20GB H3 diffusion
    model is being staged. The two cannot both fit, so DynamicVRAM pushes ~15GB
    into WDDM shared memory — i.e. system RAM over PCIe — and every sampling step
    pays for it. The conditioning tensors are already computed at this point, so
    the encoder is dead weight.

    Only releases the text encoder. VAEs and the diffusion model are untouched,
    and no generation value changes — this is purely a residency decision.
    ``standard`` keeps the previous behaviour exactly.
    """
    if not is_balanced_strategy(strategy) or clip is None:
        return ""
    try:
        import comfy.model_management as mm
    except Exception as exc:
        log.debug("text encoder release skipped (%s)", exc)
        return ""

    patcher = getattr(clip, "patcher", None) or clip
    loaded = getattr(mm, "current_loaded_models", None)
    if loaded is None:
        return ""

    freed_mib = 0.0
    released = 0
    for entry in list(loaded):
        model = getattr(entry, "model", None)
        if model is None:
            continue
        # Match the wired CLIP first; fall back to the H3 TE class name so a
        # cloned patcher is still recognised.
        inner = getattr(model, "model", None)
        name = f"{type(model).__name__}{type(inner).__name__ if inner is not None else ''}"
        is_te = model is patcher or "TEModel" in name or "MiniMaxH3TE" in name
        if not is_te:
            continue
        size = 0.0
        for attr in ("model_loaded_memory", "loaded_size", "model_size"):
            probe = getattr(entry, attr, None)
            try:
                value = probe() if callable(probe) else probe
            except Exception:
                continue
            if isinstance(value, (int, float)) and value > 0:
                size = float(value) / (1024.0 * 1024.0)
                break
        try:
            unload = getattr(entry, "model_unload", None)
            if callable(unload):
                unload()
            loaded.remove(entry)
            released += 1
            freed_mib += size
        except Exception as exc:
            log.debug("text encoder entry unload failed: %s", exc)

    if not released:
        return ""
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception:
        pass
    note = (
        f"balanced: released text encoder after conditioning "
        f"({released} entr{'y' if released == 1 else 'ies'}, ~{freed_mib:.0f}MiB) "
        "— frees VRAM for sampling; conditioning tensors already computed"
    )
    log.info("MiniMax H3 Director: %s", note)
    return note


def maybe_clear_upscaler_cache(strategy: str) -> bool:
    if not is_balanced_strategy(strategy):
        return False
    avail = system_available_gib()
    if avail is not None and avail >= RAM_RELEASE_CACHE_GIB:
        return False
    from .h3_latent_upscale import clear_h3_latent_upscaler_cache

    cleared = clear_h3_latent_upscaler_cache()
    if cleared:
        log.info(
            "MiniMax H3 Director: cleared H3 latent upscaler cache "
            "(balanced, avail=%.1fGiB)",
            avail if avail is not None else -1.0,
        )
    return cleared


def segment_vram_cleanup(
    *,
    strategy: str,
    enabled: bool = True,
    unload_models: bool = True,
) -> None:
    """Route segment cleanup: standard = official unload path; balanced = RAM-aware."""
    from .vram_cleanup import cleanup_segment_vram

    if not enabled:
        return
    if not is_balanced_strategy(strategy):
        cleanup_segment_vram(enabled=True, unload_models=unload_models)
        return
    avail = system_available_gib()
    if unload_models and avail is not None and avail < RAM_AGGRESSIVE_GIB:
        cleanup_segment_vram(enabled=True, unload_models=True)
        return
    cleanup_segment_vram(enabled=True, unload_models=False)
    soft_release_workspace(strategy=strategy, reason="segment_end")


def run_end_policy(strategy: str) -> list[str]:
    notes: list[str] = []
    if not is_balanced_strategy(strategy):
        return notes
    avail = system_available_gib()
    if avail is not None and avail < RAM_RELEASE_CACHE_GIB:
        if maybe_clear_upscaler_cache(strategy):
            notes.append(
                f"balanced run-end: cleared H3 latent upscaler cache "
                f"(avail={avail:.1f}GiB)"
            )
    soft_release_workspace(strategy=strategy, reason="run_end")
    if avail is not None and avail < RAM_AGGRESSIVE_GIB:
        notes.append(
            f"balanced run-end: soft workspace release (avail={avail:.1f}GiB)"
        )
    return notes


def _gpu_total_gib() -> float | None:
    try:
        from .memory_debug import snapshot_memory

        total = snapshot_memory().get("nvml_total_mib")
        if total is None:
            return None
        return float(total) / 1024.0
    except Exception:
        return None


def _refine_model_name(plan) -> str:
    pack = getattr(plan, "refine", None)
    if not isinstance(pack, dict):
        return ""
    for key in ("refine_model_name", "refine_model", "model_name"):
        val = pack.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    wired = pack.get("refine_model_ref")
    if isinstance(wired, dict):
        name = str(wired.get("name") or wired.get("model_name") or "").strip()
        if name:
            return name
    return ""


def gpu_refine_model_recommendation(plan, *, strategy: str) -> str | None:
    """Advisory only — never changes the wired refine model."""
    if not is_balanced_strategy(strategy):
        return None
    total = _gpu_total_gib()
    if total is not None and total > 24.0:
        return None
    name = _refine_model_name(plan).lower()
    if not name:
        return None
    if W4A8_REFINE_RECOMMENDATION.lower() in name:
        return None
    if not any(h in name for h in INT8_REFINE_HINTS):
        return None
    gpu_note = f"{total:.0f}GB-class GPU" if total is not None else "20GB-class GPU"
    return (
        "[Memory Recommendation]\n"
        f"Detected {gpu_note} with a high-staging refine checkpoint ({name}).\n"
        "For Ref2V refine on this hardware, the W4A8 mixed checkpoint can reduce "
        "DynamicVRAM staging memory with little visible quality change in user testing.\n"
        f"Recommended: {W4A8_REFINE_RECOMMENDATION}\n"
        "(Advisory only — wire the refine_model input manually; Director does not "
        "auto-replace checkpoints.)"
    )
