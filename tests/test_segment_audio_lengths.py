"""Verify poster outputs do not shorten generated audio or source audio."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class AudioLengthTests(unittest.TestCase):
    def test_posters_preserve_generated_and_source_audio_duration(self):
        pcm = {"waveform": torch.ones(1, 2, 44100), "sample_rate": 44100}
        io = ModuleType("audio_length_test.lib.audio_io")
        io.frames_to_audio_samples = lambda n, fps, sr: round(n * sr / fps)
        io.extract_timeline_audio = lambda *args, **kwargs: pcm
        io.load_reference_audio = lambda *args, **kwargs: pcm
        io.diagnose_source_audio_failure = lambda *args: "not used"
        spec = importlib.util.spec_from_file_location(
            "audio_length_test.director.audio_export",
            Path(__file__).resolve().parents[1] / "director" / "audio_export.py",
        )
        audio = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {io.__name__: io}):
            spec.loader.exec_module(audio)
        plan = SimpleNamespace(
            frame_rate=24, global_task_key="r2v", raw={}, run_indices=None,
            segments=[
                SimpleNamespace(index=0, start_frame=0, end_frame=24, frame_count=24),
                SimpleNamespace(index=1, start_frame=24, end_frame=72, frame_count=48),
            ],
        )
        # First segment has been released to a poster; last remains full.
        images = [torch.zeros(1, 2, 2, 3), torch.zeros(48, 2, 2, 3)]
        for mode in ("generate", "source"):
            with self.subTest(mode=mode):
                output, fallback = audio.build_director_audio_outputs(
                    plan, images, export_segments=True, audio_mode=mode,
                    segment_audios=[pcm, pcm] if mode == "generate" else None,
                    segment_frame_counts=[24, 48],
                )
                self.assertIsNone(fallback)
                self.assertEqual([a["waveform"].shape[-1] for a in output], [44100, 88200])


if __name__ == "__main__":
    unittest.main()
