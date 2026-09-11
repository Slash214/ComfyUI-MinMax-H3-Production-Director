"""CPU checks for September 11 canvas-aware continuity and reference limits."""
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from test_continuity_keep_tail import helpers, ROOT


class CanvasContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType("canvas_test")
        package.__path__ = [str(ROOT / "director")]
        with patch.dict(sys.modules, {"canvas_test": package}):
            spec = importlib.util.spec_from_file_location(
                "canvas_test.h3_motion_context", ROOT / "director/h3_motion_context.py")
            cls.motion = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.motion)

    def av(self, h=4, w=6, fill=0):
        return {"samples": (torch.full((1, 2, 17, h, w), float(fill)), torch.ones(1, 2, 20))}

    def test_choose_first_pass_only_when_final_canvas_differs(self):
        first, final = self.av(), self.av(8, 12)
        choose = self.motion.select_continuity_pin_latent
        self.assertIs(choose(self.av(), first, final), first)
        self.assertIs(choose(self.av(8, 12), first, final), final)
        self.assertIs(choose(self.av(), first, first), first)

    def test_prefix_copy_preserves_body_audio_and_original(self):
        m = self.motion
        target, source = self.av(), self.av(fill=2)
        with patch.object(m, "_repack_av_streams", side_effect=lambda parts, template=None: tuple(parts)):
            out = m.copy_av_tail_into_prefix(target, source, 22)
            head = m.slice_av_prefix(out, 22)
        steps = m.steps_for_frames(22)
        self.assertEqual(head["samples"][0].shape[2], steps)
        self.assertTrue(torch.all(out["samples"][0][:, :, :steps] == 2))
        self.assertTrue(torch.all(out["samples"][0][:, :, steps:] == 0))
        self.assertTrue(torch.all(target["samples"][0] == 0))
        self.assertIs(out["samples"][1], target["samples"][1])
        with self.assertRaises(ValueError):
            m.copy_av_tail_into_prefix(target, self.av(8, 12), 22)

    def test_packed_tensor_is_not_misread_as_av_streams(self):
        with self.assertRaises(ValueError):
            self.motion._streams_from_latent({"samples": torch.zeros(2, 2, 16, 4, 6)})

    def test_reference_presets_preserve_legacy_and_map_official_mode(self):
        ns = helpers("director/plan.py", {
            "normalize_ref_image_size", "official_ref_image_size", "ref_image_long_preset_px",
            "_migrate_ref_image_size"}, REF_IMAGE_SIZE_MATCH="match", REF_IMAGE_SIZE_MAX="max",
            REF_IMAGE_LONG_PRESETS=(1024, 1280, 1536))
        for mode in ("1024", "1280", "1536"):
            self.assertEqual(ns["ref_image_long_preset_px"](mode), int(mode))
            self.assertEqual(ns["official_ref_image_size"](mode), "max")
        self.assertEqual(ns["official_ref_image_size"]("match"), "match")
        self.assertEqual(ns["normalize_ref_image_size"]("bad"), "match")
        self.assertEqual(ns["_migrate_ref_image_size"]("max", {
            "refImageLimitEdge": "long", "refImageLimitPx": 1280}), "1280")

    def test_reference_downscale_and_noop_keep_rank(self):
        ns = helpers("lib/image_prep.py", {"fit_edge_limit"}, torch=torch,
                     common_upscale=lambda x, w, h, *args: F.interpolate(x, size=(h, w), mode="area"))
        resize = ns["fit_edge_limit"]
        for shape in ((32, 64, 3), (1, 32, 64, 3)):
            x = torch.ones(shape)
            self.assertIs(resize(x, 128, edge="long"), x)
            out = resize(x, 32, edge="long")
            self.assertEqual(out.ndim, x.ndim)
            self.assertEqual(tuple(out.shape[-3:-1]), (16, 32))

    def test_missing_or_wrong_source_metadata_prevents_cache_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "seg_0000.pre.av.pt").touch()
            torch_stub = SimpleNamespace(load=Mock(return_value={"samples": "latent"}))
            ns = helpers("director/segment_cache.py", {"load_first_pass_av_latent"},
                         torch=torch_stub, json=json, log=logging.getLogger(__name__),
                         _cache_root=lambda _: root,
                         first_pass_cache_fingerprint=lambda *args: {"source": "new"},
                         _reject_source_stale=lambda *args, **kwargs: True)
            load = ns["load_first_pass_av_latent"]
            self.assertIsNone(load("node", SimpleNamespace(index=0), None, allow_stale=True))
            (root / "seg_0000.pre.meta.json").write_text('{"source":"old"}', encoding="utf-8")
            self.assertIsNone(load("node", SimpleNamespace(index=0), None, allow_stale=True))
            torch_stub.load.assert_not_called()
            (root / "seg_0000.pre.meta.json").write_text('{"source":"new"}', encoding="utf-8")
            self.assertEqual(load("node", SimpleNamespace(index=0), None), {"samples": "latent"})

    def test_opening_grade_changes_only_head_without_mutating_input(self):
        ns = helpers("director/segment_continuity.py", {"match_export_opening_grade"},
                     torch=torch, log=logging.getLogger(__name__),
                     CONTINUITY_EXPORT_GRADE_FRAMES=12, CONTINUITY_EXPORT_GRADE_WEIGHT=0.7,
                     CONTINUITY_EXPORT_GRADE_BLUR=64,
                     _lowfreq_appearance_pull=lambda src, guide, weight, blur: src + weight * guide)
        body = torch.zeros(20, 2, 2, 3)
        out = ns["match_export_opening_grade"](body, torch.ones(1, 2, 2, 3))
        self.assertTrue(torch.all(body == 0))
        self.assertTrue(torch.all(out[12:] == 0))
        self.assertGreater(out[0].mean(), out[11].mean())


if __name__ == "__main__":
    unittest.main()
