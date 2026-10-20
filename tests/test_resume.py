"""检查随机状态检查点的保存与恢复。"""
from pathlib import Path
import random
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

class ResumeTests(unittest.TestCase):
    def test_random_state_roundtrip(self):
        import numpy as np
        import torch
        from training_state import seed_everything, capture_rng, restore_rng, atomic_save
        from checkpoints import load_checkpoint
        seed_everything(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pth'
            atomic_save({'rng':capture_rng()}, path)
            expected = (random.random(), np.random.rand(), torch.rand(3))
            restore_rng(load_checkpoint(path, mmap=False)['rng'])
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
            self.assertFalse(path.with_name('state.pth.tmp').exists())
