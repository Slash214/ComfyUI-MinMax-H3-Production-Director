"""CPU regressions for the September 9 upstream continuity/audio fixes."""

import ast
import importlib.util
import logging
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def helpers(path, names, **namespace):
    """Load selected production helpers without starting ComfyUI or GPU models."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), path, "exec"), namespace)
    return namespace


class KeepTailTests(unittest.TestCase):
    def setUp(self):
        self.motion = helpers(
            "director/h3_motion_context.py",
            {"continuity_export_len", "trim_context_prefix", "handoff_end_frame"},
            torch=torch, FPS=24.0,
        )

    def test_export_budget_and_handoff(self):
        for keep, expected in ((True, 136), (False, 124)):
            with self.subTest(keep=keep):
                count = self.motion["continuity_export_len"](
                    trim_frames=22, sample_len=158, visible_frames=124,
                    target_len=124, keep_tail=keep,
                )
                self.assertEqual(count, expected)
                self.assertEqual(self.motion["handoff_end_frame"](
                    trim_frames=22, export_frames=count), 22 + expected)
        self.assertEqual(self.motion["continuity_export_len"](
            trim_frames=0, sample_len=158, visible_frames=124,
            target_len=124, keep_tail=True), 124)

    def test_decoded_video_and_audio_tail_survive_together(self):
        trim = helpers("director/executor_core.py", {"_trim_decoded_to_export"},
                       torch=torch, trim_context_prefix=self.motion["trim_context_prefix"])["_trim_decoded_to_export"]
        frames = torch.arange(158).reshape(158, 1, 1, 1)
        # One sample per frame makes the exact clipped syllable boundary observable.
        pcm = {"sample_rate": 24, "waveform": torch.arange(158).reshape(1, 1, 158)}
        for keep, count, last in ((True, 136, 157), (False, 124, 145)):
            with self.subTest(keep=keep):
                length = self.motion["continuity_export_len"](
                    trim_frames=22, sample_len=158, visible_frames=124,
                    target_len=124, keep_tail=keep)
                images, audio = trim(frames, pcm, trim_frames=22, export_len=length,
                                     plan=SimpleNamespace(frame_rate=24))
                self.assertEqual(images.shape[0], count)
                self.assertEqual(audio["waveform"].shape[-1], count)
                self.assertEqual(images[-1].item(), last)
                self.assertEqual(audio["waveform"][..., -1].item(), last)

    def test_merged_layout_preserves_tail_only_when_enabled(self):
        layout = helpers("nodes/director_common.py", {"_layout_image_batches"},
                         torch=torch, pad_or_trim_frames=lambda x, n: x[:n])["_layout_image_batches"]
        segments = [torch.zeros(124, 2, 2, 3), torch.ones(136, 2, 2, 3)]
        for enabled, keep, expected in ((True, True, 260), (True, False, 248), (False, True, 248)):
            with self.subTest(enabled=enabled, keep=keep):
                output, count = layout(
                    SimpleNamespace(total_frames=248, continuity_enabled=enabled, continuity_keep_tail=keep),
                    torch.cat(segments), segments, export_segments=False, is_batch=False, video_batch=False)
                self.assertEqual(count, expected)
                self.assertEqual(output[0].shape[0], expected)

    def test_workflow_defaults_and_explicit_opt_out(self):
        resolve = helpers("director/segment_continuity.py", {"resolve_continuity_keep_tail"})["resolve_continuity_keep_tail"]
        for timeline in (None, {}, {"output": {}}, {"output": {"continuityKeepTail": True}}):
            self.assertTrue(resolve(timeline))
        for key in ("continuityKeepTail", "continuity_keep_tail"):
            for value in (False, 0, "false", "0", "no", "off"):
                self.assertFalse(resolve({"output": {key: value}}))

    def test_keep_tail_invalidates_cropped_cache(self):
        fingerprint = helpers(
            "director/segment_cache.py", {"_segment_identity_fingerprint"},
            resolve_ref_image_size=lambda *args: 512, source_video_identity=lambda _: "source",
            SOURCE_VIDEO_FP_KEY="source_video", CONTINUE_PIPELINE_ID="continue",
            CONTINUITY_PIPELINE_ID="guide",
        )["_segment_identity_fingerprint"]
        seg = SimpleNamespace(refs=[], reference_video_meta={}, index=1, start_frame=124,
                              end_frame=248, prompt="test", negative_prompt="", task_key="r2v",
                              reference_video_start_frame=0)
        plan = SimpleNamespace(width=512, height=512, output_mode="long_edge", ref_max_size=512,
                               continuity_enabled=True, continuity_overlap_frames=22)
        full = fingerprint(seg, plan)
        self.assertTrue(full["continuity_keep_tail"])
        plan.continuity_keep_tail = False
        cropped = fingerprint(seg, plan)
        self.assertNotIn("continuity_keep_tail", cropped)
        self.assertNotEqual(full, cropped)
        plan.continuity_enabled = False
        disabled = fingerprint(seg, plan)
        plan.continuity_keep_tail = True
        self.assertEqual(disabled, fingerprint(seg, plan))

    def test_merged_audio_uses_actual_extended_frame_counts(self):
        tree = ast.parse((ROOT / "director/audio_export.py").read_text(encoding="utf-8-sig"))
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        ns = helpers("director/audio_export.py", names, torch=torch, SILENT_SAMPLE_RATE=44100,
                     log=logging.getLogger(__name__),
                     frames_to_audio_samples=lambda n, fps, sr: round(n * sr / fps))
        first = {"sample_rate": 24, "waveform": torch.zeros(1, 2, 124)}
        last = {"sample_rate": 24, "waveform": torch.ones(1, 2, 136)}
        last["waveform"][..., -1] = 0.5
        merged = ns["_merge_generated_segment_audios"](
            SimpleNamespace(), [first, last], total_frames=260, fps=24, frame_counts=[124, 136])
        self.assertEqual(merged["waveform"].shape[-1], 260)
        self.assertTrue(torch.all(merged["waveform"][..., -1] == 0.5))


class ReferenceAudioTests(unittest.TestCase):
    def test_failed_slot_does_not_renumber_following_audio_tag(self):
        ns = helpers("director/plan.py", {"usable_ref_audio_indices", "drop_unusable_audio_prompt_tags"},
                     re=re, log=logging.getLogger(__name__),
                     _AUDIO_PROMPT_TAG_RE=re.compile(r"<\s*Audio\s+(\d+)\s*>", re.IGNORECASE))
        ns["ensure_ref_audio_pcm"] = lambda item, cache=None: item.audio
        slots = [SimpleNamespace(index=0, audio=None),
                 SimpleNamespace(index=1, audio={"waveform": torch.ones(1, 1, 24)})]
        indices = ns["usable_ref_audio_indices"](slots)
        self.assertEqual(indices, [1])
        prompt = ns["drop_unusable_audio_prompt_tags"]("A <Audio 1> B <Audio 2>", indices)
        self.assertNotIn("<Audio 1>", prompt)
        self.assertIn("<Audio 2>", prompt)


class VideoExportTests(unittest.TestCase):
    def test_ffmpeg_exports_decodable_video_and_audio(self):
        spec = importlib.util.spec_from_file_location("keep_tail_video_export", ROOT / "lib/video_export.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ffmpeg = module._ffmpeg_bin()
        if not ffmpeg:
            self.skipTest("ffmpeg unavailable")
        with tempfile.TemporaryDirectory(prefix="h3-tail-test-") as folder:
            path = Path(folder) / "tail.mp4"
            frames = torch.zeros(24, 16, 16, 3)
            pcm = {"sample_rate": 24000, "waveform": torch.ones(1, 2, 24000) * 0.05}
            module.write_frames_to_mp4(path=path, frames=frames, fps=24, audio=pcm)
            self.assertGreater(path.stat().st_size, 0)
            video = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                   capture_output=True, check=True, timeout=30)
            self.assertEqual(len(video.stdout), 24 * 16 * 16 * 3)
            audio = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0",
                                    "-f", "s16le", "pipe:1"],
                                   capture_output=True, check=True, timeout=30)
            self.assertGreaterEqual(len(audio.stdout), 24000 * 2 * 2)


if __name__ == "__main__":
    unittest.main()
