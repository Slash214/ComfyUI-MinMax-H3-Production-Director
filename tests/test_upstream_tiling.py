"""CPU coverage for optional tiling and bounded animated previews."""
import base64
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "director" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TilingTests(unittest.TestCase):
    def test_regions_cover_canvas_without_zero_weight(self):
        m = load("spatial_tiled_sampling")
        for size in (25, 33, 64, 97):
            for count in (2, 3, 8):
                for overlap in (0, 16, 128, 2048):
                    plan = m._plan_regions((1, 2, 3, 16, size), count, overlap)
                    if plan is None:
                        continue
                    weights = torch.zeros(size)
                    for start, end, left, right in plan["regions"]:
                        weights[start:end] += m._raised_cosine_1d(
                            end-start, left, right, dtype=torch.float32, device="cpu")
                    self.assertTrue(torch.all(weights > 0), (size, count, overlap))

    def test_partial_payload_failure_can_be_rolled_back(self):
        m = load("spatial_tiled_sampling")
        original_shapes = [(1, 2, 3, 16, 64)]
        original_payload = {"keyframes": []}
        shape = SimpleNamespace(cond=original_shapes)
        payload = SimpleNamespace(cond=original_payload)
        cfg = SimpleNamespace(conds={"positive": [{"model_conds": {
            "latent_shapes": shape, "minimax_payload": payload}}]})
        rollback = []
        with patch.object(m, "_rebuild_layout", side_effect=RuntimeError("layout failure")):
            with self.assertRaises(RuntimeError):
                m._install_tile_payloads(cfg, axis="W", start=0, end=32,
                                         tile_shapes=[(1, 2, 3, 16, 32)],
                                         full_h=16, full_w=64, restorations=rollback)
        m._restore_payloads(rollback)
        self.assertIs(shape.cond, original_shapes)
        self.assertIs(payload.cond, original_payload)

    def test_wrapper_restores_sampler(self):
        m = load("spatial_tiled_sampling")
        original = lambda *args, **kwargs: None
        sampler = SimpleNamespace(sampler_function=original)
        restore = m.wrap_sampler_spatial_tiles(sampler, n_tiles=2, overlap_pixels=0)
        self.assertIsNot(sampler.sampler_function, original)
        restore()
        self.assertIs(sampler.sampler_function, original)
        m.wrap_sampler_spatial_tiles(sampler, n_tiles=1, overlap_pixels=128)()
        self.assertIs(sampler.sampler_function, original)

    def test_temporal_chunks_keep_all_frames_and_blend(self):
        m = load("h3_latent_upscale")
        # Exercise real chunk slicing/blending using a frame-local test resizer.
        class Resizer:
            _temporal_kernel = lambda self: 5
            _forward_seg = lambda self, x, scale, size: F.interpolate(x, size=size, mode="nearest")
        x = torch.arange(61, dtype=torch.float32).reshape(1, 1, 61, 1, 1)
        for enabled in (False, True):
            actual = m.LatentResizer3D.forward(Resizer(), x, 2, (61, 2, 2), enabled)
            torch.testing.assert_close(actual, x.expand(1, 1, 61, 2, 2))

    def test_third_party_upscaler_without_chunking_is_compatible(self):
        m = load("h3_latent_upscale")
        class Legacy(torch.nn.Module):
            def forward(self, x, scale, target_size):
                return x
        x = torch.ones(1)
        self.assertIs(m._forward_upscaler(Legacy(), x, scale=2,
                                         target_size=(1, 2, 2), enable_chunking=True), x)

    def test_preview_frame_limit_and_webp_payload(self):
        m = load("tae_preview")
        indices = m._pick_temporal_indices(1000, m.LIVE_PREVIEW_MAX_FRAMES)
        self.assertEqual(len(indices), 16)
        self.assertEqual((indices[0], indices[-1]), (0, 999))
        frames = [Image.new("RGB", (16, 16), color) for color in ("red", "blue")]
        encoded, mime, width, height = m.encode_preview_payload(frames)
        self.assertEqual((mime, width, height), ("image/webp", 16, 16))
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.n_frames, 2)
            self.assertEqual(image.info["loop"], 0)

    def test_preview_webp_failure_falls_back_to_jpeg(self):
        m = load("tae_preview")
        frames = [Image.new("RGB", (16, 16))] * 2
        with patch.object(m, "encode_animated_webp", return_value=""):
            encoded, mime, _, _ = m.encode_preview_payload(frames)
        self.assertEqual(mime, "image/jpeg")
        self.assertTrue(base64.b64decode(encoded).startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
