import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_vggt_geometry_tum10 import mirage_projected_task_scores


args = SimpleNamespace(
    attack_loss="mirage_projected_split_pose",
    mirage_projected_tasks="track",
    mirage_projected_track_cap_px=20.0,
    mirage_projected_track_mode="eval_epe",
    mirage_projected_track_min_points_per_frame=1,
    mirage_projected_track_target_softmin_tau_px=0.25,
)
target = torch.zeros(3, 4, 2)
pred = torch.ones(1, 3, 4, 2)
valid = torch.ones(3, 4, dtype=torch.bool)
scores = mirage_projected_task_scores(
    {"track": pred},
    {
        "mirage_projected_gt": {
            "tracks": target,
            "track_visible": valid,
            "track_known": valid,
        }
    },
    args,
)
assert tuple(scores) == ("track",)
assert torch.allclose(scores["track"], torch.tensor(2.0).sqrt())
print("selective-track regression: PASS")
