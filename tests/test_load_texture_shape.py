from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_vggt_geometry_tum10 import load_texture  # noqa: E402


def test_load_texture_restores_non_square_saved_shape() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "spoo_texture.npz"
        np.savez_compressed(
            path,
            texture=np.zeros((1, 3, 450, 600), dtype=np.float32),
            spoo_borders_lrtb=np.asarray([0.0, 1.0, 0.9, 1.0], dtype=np.float32),
        )
        args = Namespace(baseline_mask_temperature=1.0)

        texture = load_texture(str(path), torch.device("cpu"), args)

        assert tuple(texture.shape) == (1, 3, 450, 600)
        assert args._texture_hw == (450, 600)
        assert torch.equal(
            args._spoo_borders.cpu(),
            torch.tensor([0.0, 1.0, 0.9, 1.0]),
        )
        assert args._spoo_region_frozen is True
