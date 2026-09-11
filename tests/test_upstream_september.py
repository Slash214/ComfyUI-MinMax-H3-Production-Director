"""CPU checks for upstream FPS, remask and aligned segment-output changes."""

import ast
import copy
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def functions_from(path, namespace):
    """Run real helper bodies without importing the ComfyUI server/model stack."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), path, "exec"), namespace)
    return namespace


class OutputAlignmentTests(unittest.TestCase):
    def run_outputs(self, partial, mode, released):
        export = functions_from("director/segment_mp4_export.py", {"torch": torch})
        seen = {}

        def audio(plan, images, **kwargs):
            seen["audio_indices"] = sorted(plan.run_indices) if plan.run_indices is not None else list(range(len(plan.segments)))
            seen["counts"] = kwargs["segment_frame_counts"]
            return [f"audio-{i}" for i in seen["audio_indices"]], None

        def source(plan, images, **kwargs):
            seen["source_indices"] = sorted(plan.run_indices) if plan.run_indices is not None else list(range(len(plan.segments)))
            return [torch.full_like(img, float(i)) for i, img in zip(seen["source_indices"], images)]

        ns = functions_from("nodes/director_common.py", {
            "copy": copy, "torch": torch, "log": logging.getLogger("test"),
            "is_prompt_batch_timeline": lambda *_: True,
            "is_video_batch_task_key": lambda *_: True,
            "released_output_slots": export["released_output_slots"],
            "resolve_audio_mode": lambda _: mode,
            "AUDIO_MODE_GENERATE": "generate",
            "build_director_audio_outputs": audio,
            "source_audio_report_note": lambda *a, **kw: "",
        })
        ns["build_source_images_output"] = source
        plan = SimpleNamespace(
            raw={}, global_task_key="r2v", export_mode="segments", frame_rate=24,
            run_indices=frozenset({1, 3}) if partial else None,
            segments=[object() for _ in range(4 if partial else 2)],
            segment_mp4_run_dir="export", refine=None,
        )
        frames = [torch.zeros(1 if released else 24, 2, 2, 3), torch.ones(48, 2, 2, 3)]
        out = ns["finalize_director_outputs"](
            plan, frames[-1], frames, "", export_source_images=True,
            segment_audios=["first", "second"], segment_frame_counts=[24, 48],
            pre_refine_combined=frames[-1], pre_refine_segments=frames,
        )
        expected = ([3] if partial else [1]) if released else ([1, 3] if partial else [0, 1])
        self.assertEqual(seen["audio_indices"], expected)
        self.assertEqual(seen["source_indices"], expected)
        self.assertEqual(seen["counts"], [48] if released else [24, 48])
        self.assertEqual(out[3], 48 if released else 72)
        self.assertEqual([len(out[i]) for i in (0, 1, 4, 6)], [len(expected)] * 4)
        self.assertEqual(plan.run_indices, frozenset({1, 3}) if partial else None)

    def test_source_and_generated_audio_follow_kept_segments(self):
        for partial in (False, True):
            for mode in ("source", "generate"):
                for released in (False, True):
                    with self.subTest(partial=partial, mode=mode, released=released):
                        self.run_outputs(partial, mode, released)

    def test_short_full_clip_is_not_mistaken_for_poster(self):
        ns = functions_from("director/segment_mp4_export.py", {"torch": torch})
        self.assertFalse(ns["is_released_poster"](torch.zeros(23, 2, 2, 3), 24))
        self.assertTrue(ns["is_released_poster"](torch.zeros(1, 2, 2, 3), 24))


class FrameRateTests(unittest.TestCase):
    def test_duration_uses_requested_fps_instead_of_stale_frame_count(self):
        fl = ModuleType("fps_test.fl2v_timeline")
        fl.__dict__.update(functions_from("director/fl2v_timeline.py", {}))
        ns = functions_from("director/gen_timeline.py", {
            "__name__": "fps_test.gen_timeline", "__package__": "fps_test",
            "VIDEO_BATCH_KEYS": {"r2v", "t2v", "i2v", "fl2v", "mixed"},
            "IMAGE_BATCH_KEYS": set(), "MIN_GEN_VIDEO_FRAMES": 4,
        })
        with patch.dict(sys.modules, {"fps_test.fl2v_timeline": fl}):
            for fps, expected in ((12, 73), (24, 124), (30, 158), (60, 311)):
                for task in ("t2v", "r2v", "mixed"):
                    with self.subTest(fps=fps, task=task):
                        self.assertEqual(ns["_segment_frame_count"](
                            {"durationSec": 5, "frameCount": 124}, default=124,
                            task_key=task, frame_rate=fps,
                        ), expected)


class ContinueMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType("continue_test")
        package.__path__ = [str(ROOT / "director")]
        with patch.dict(sys.modules, {"continue_test": package}):
            spec = importlib.util.spec_from_file_location("continue_test.h3_latent_continue", ROOT / "director/h3_latent_continue.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_remask_preserves_free_tail_and_never_hard_pins_video(self):
        m = self.module
        remask = m._PrefixRemask(4, [1.0, 0.5, 0.0], (1, 1, 7, 2, 2), seam_min=0.65)
        mask = remask.denoise_mask_function(torch.tensor([1.0]), torch.ones(1, 1, 7, 2, 2))
        self.assertEqual(tuple(mask.shape), (1, 1, 7, 2, 2))
        self.assertTrue(torch.all(mask[:, :, :4] >= 0.65))
        self.assertTrue(torch.all(mask[:, :, 4:] == 1.0))
        self.assertEqual(tuple(remask.current_video_mask.shape), (1, 1, 7, 2, 2))

    def test_invalid_strength_is_clamped(self):
        m = self.module
        for value, expected in ((None, 0.10), (float("nan"), 0.10), (-1, 0.0), (0, 0.0), (2, 0.95)):
            self.assertEqual(m.clamp_seam_min_mask(value), expected)

    def test_zero_redraw_hard_locks_seam_but_not_free_tail(self):
        m = self.module
        remask = m._PrefixRemask(4, [1.0, 0.5, 0.0], (1, 1, 7, 2, 2), seam_min=0.0)
        mask = remask.denoise_mask_function(torch.tensor([0.5]), torch.ones(1, 1, 7, 2, 2))
        self.assertTrue(torch.all(mask[:, :, 3:4] == 0))
        self.assertTrue(torch.all(mask[:, :, 4:] == 1))


if __name__ == "__main__":
    unittest.main()
