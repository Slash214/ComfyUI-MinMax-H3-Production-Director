"""Release exported segment pixels without changing default IMAGE outputs."""

from __future__ import annotations

import gc
from pathlib import Path

import torch


def should_release_segment_pixels(export_mode, enabled, confirm_first_pass) -> bool:
    return enabled is True and export_mode == "segments" and not confirm_first_pass


def _pixel_exports_complete(run_dir, index, paths, *, require_pre=False) -> bool:
    """Require successful results from this write and nonempty files, not stale files."""
    if run_dir is None:
        return False
    try:
        expected = [Path(run_dir) / f"seg_{int(index):04d}.mp4"]
        if require_pre:
            expected.append(Path(run_dir) / f"seg_{int(index):04d}_pre.mp4")
        returned = {Path(p).resolve() for p in paths if p}
        return all(
            p.resolve() in returned and p.is_file() and p.stat().st_size > 0
            for p in expected
        )
    except (OSError, TypeError, ValueError):
        return False


def _poster_frame(tensor: torch.Tensor | None) -> torch.Tensor:
    """1-frame stand-in so IMAGE list length stays valid after a pixel release."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or int(tensor.shape[0]) <= 0:
        return torch.full((1, 2, 2, 3), 0.5)
    return tensor[-1:].detach().cpu().contiguous().clone()


def _release_segment_pixels(
    index: int,
    *,
    completed_outputs: dict[int, torch.Tensor],
    completed_pre_refine: dict[int, torch.Tensor],
    completed_refine_passes: dict[int, list[tuple[str, torch.Tensor]]],
    segment_outputs: list[torch.Tensor],
    segment_pre_refine: list[torch.Tensor],
    progress_pos: dict[int, int],
    persisted_segments: set[int],
) -> bool:
    """Drop full-resolution pixels for a finished predecessor. Audio stays.

    Replaces IMAGE-list slots with a 1-frame poster. Safe after mp4 + the
    next segment has already pinned / phase-trimmed this index.
    """
    idx = int(index)
    if idx < 0 or idx not in persisted_segments:
        return False
    chunk = completed_outputs.pop(idx, None)
    pre = completed_pre_refine.pop(idx, None)
    completed_refine_passes.pop(idx, None)
    run_pos = progress_pos.get(idx)
    had = chunk is not None or pre is not None
    if run_pos is not None and run_pos < len(segment_outputs):
        src = chunk if chunk is not None else segment_outputs[run_pos]
        poster = _poster_frame(src)
        segment_outputs[run_pos] = poster
        if run_pos < len(segment_pre_refine):
            if pre is chunk:
                segment_pre_refine[run_pos] = poster
            else:
                segment_pre_refine[run_pos] = _poster_frame(
                    pre if pre is not None else segment_pre_refine[run_pos]
                )
        had = True
    elif chunk is not None or pre is not None:
        had = True
    if had:
        del chunk, pre
        gc.collect()
    return had
