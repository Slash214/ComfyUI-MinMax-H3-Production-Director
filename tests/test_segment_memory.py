"""CPU regression tests; run without starting ComfyUI or loading H3 models."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "segment_memory", Path(__file__).resolve().parents[1] / "director" / "segment_memory.py"
)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


class ExportProofTests(unittest.TestCase):
    def test_success_requires_current_write_and_nonempty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "seg_0000.mp4"
            final.write_bytes(b"exported video")
            self.assertTrue(memory._pixel_exports_complete(tmp, 0, [str(final)]))
            # An old file is not evidence that this rewrite succeeded.
            self.assertFalse(memory._pixel_exports_complete(tmp, 0, []))
            final.write_bytes(b"")
            self.assertFalse(memory._pixel_exports_complete(tmp, 0, [str(final)]))

    def test_refine_requires_both_outputs_from_this_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "seg_0000.mp4"
            pre = Path(tmp) / "seg_0000_pre.mp4"
            final.write_bytes(b"final")
            pre.write_bytes(b"pre")
            self.assertFalse(memory._pixel_exports_complete(tmp, 0, [final], require_pre=True))
            self.assertFalse(memory._pixel_exports_complete(tmp, 0, [pre], require_pre=True))
            self.assertTrue(memory._pixel_exports_complete(tmp, 0, [final, pre], require_pre=True))
            pre.unlink()
            self.assertFalse(memory._pixel_exports_complete(tmp, 0, [final, pre], require_pre=True))

    def test_missing_export_directory_never_allows_release(self):
        self.assertFalse(memory._pixel_exports_complete(None, 0, []))


class PixelReleaseTests(unittest.TestCase):
    def release(self, *, persisted, shared_pre=False, index=3):
        frames = torch.rand(20, 8, 8, 3)
        pre = frames if shared_pre else torch.rand(20, 4, 4, 3)
        other = torch.rand(10, 8, 8, 3)
        state = dict(
            completed_outputs={3: frames},
            completed_pre_refine={3: pre},
            completed_refine_passes={3: [("p1", frames)]},
            segment_outputs=[other, frames],
            segment_pre_refine=[other, pre],
            progress_pos={0: 0, 3: 1},
            persisted_segments=persisted,
        )
        result = memory._release_segment_pixels(index, **state)
        return result, state, frames, pre, other

    def test_failed_export_preserves_all_pixels(self):
        result, state, frames, pre, _ = self.release(persisted=set())
        self.assertFalse(result)
        self.assertIs(state["completed_outputs"][3], frames)
        self.assertIs(state["segment_outputs"][1], frames)
        self.assertIs(state["segment_pre_refine"][1], pre)
        self.assertIn(3, state["completed_refine_passes"])

    def test_partial_run_releases_correct_slot_and_separate_pre(self):
        result, state, frames, pre, other = self.release(persisted={3})
        self.assertTrue(result)
        self.assertNotIn(3, state["completed_outputs"])
        self.assertNotIn(3, state["completed_refine_passes"])
        self.assertIs(state["segment_outputs"][0], other)
        for poster, original in [(state["segment_outputs"][1], frames), (state["segment_pre_refine"][1], pre)]:
            torch.testing.assert_close(poster, original[-1:])
            self.assertEqual(poster.untyped_storage().nbytes(), poster.numel() * poster.element_size())
            self.assertNotEqual(poster.untyped_storage().data_ptr(), original.untyped_storage().data_ptr())

    def test_shared_pre_reuses_small_poster(self):
        _, state, _, _, _ = self.release(persisted={3}, shared_pre=True)
        self.assertIs(state["segment_outputs"][1], state["segment_pre_refine"][1])

    def test_one_frame_view_does_not_retain_full_video_storage(self):
        full = torch.rand(100, 8, 8, 3)
        poster = memory._poster_frame(full[-1:])
        self.assertEqual(poster.untyped_storage().nbytes(), 8 * 8 * 3 * 4)
        self.assertNotEqual(poster.untyped_storage().data_ptr(), full.untyped_storage().data_ptr())

    def test_release_is_explicit_and_confirmation_keeps_video(self):
        self.assertFalse(memory.should_release_segment_pixels("segments", False, False))
        self.assertFalse(memory.should_release_segment_pixels("all", True, False))
        self.assertFalse(memory.should_release_segment_pixels("segments", True, True))
        self.assertFalse(memory.should_release_segment_pixels("segments", "true", False))
        self.assertTrue(memory.should_release_segment_pixels("segments", True, False))


if __name__ == "__main__":
    unittest.main()
