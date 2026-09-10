"""Reproduce the upstream rollback defect against the preserved Git revision."""
import inspect
import subprocess
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_upstream_tiling import ROOT, load


class UpstreamComparisonTests(unittest.TestCase):
    def test_local_rolls_back_where_upstream_leaves_tile_shapes(self):
        try:
            source = subprocess.check_output(
                ["git", "show", "5fede41277afea249ca4c4f2775293ff58de45f2:director/spatial_tiled_sampling.py"],
                cwd=ROOT, stderr=subprocess.PIPE,
            ).decode("utf-8-sig")
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.skipTest("Upstream Git history unavailable (ZIP or shallow checkout)")
        upstream = ModuleType("upstream_tiles_comparison")
        exec(compile(source, "upstream_tiles_comparison", "exec"), upstream.__dict__)
        for label, module, expected in (
            ("upstream", upstream, False),
            ("local", load("spatial_tiled_sampling"), True),
        ):
            with self.subTest(version=label):
                original = [(1, 2, 3, 16, 64)]
                shape = SimpleNamespace(cond=original)
                payload = SimpleNamespace(cond={})
                cfg = SimpleNamespace(conds={"positive": [{"model_conds": {
                    "latent_shapes": shape, "minimax_payload": payload}}]})
                rollback = []
                kwargs = dict(axis="W", start=0, end=32, tile_shapes=[(1, 2, 3, 16, 32)],
                              full_h=16, full_w=64)
                # Use each version's own calling convention and rollback path.
                if "restorations" in inspect.signature(module._install_tile_payloads).parameters:
                    kwargs["restorations"] = rollback
                with patch.object(module, "_rebuild_layout", side_effect=RuntimeError("layout failure")):
                    try:
                        rollback = module._install_tile_payloads(cfg, **kwargs)
                    except RuntimeError:
                        module._restore_payloads(rollback)
                    else:
                        self.fail("Expected injected layout failure")
                self.assertEqual(shape.cond is original, expected)


if __name__ == "__main__":
    unittest.main()
