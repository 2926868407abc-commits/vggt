"""Does a bigger GT trajectory actually buy headroom above the random baseline?

The metric is ATE / ate_ceiling and the ceiling is the GT RMS radius, so both
numerator and denominator scale with the trajectory. If the random baseline is
~90% because an uncorrelated prediction collapses under Sim(3) alignment, then it
is ~90% on every sequence and picking a wider one changes nothing. Test it on all
eight sequences before spending GPU time.

Also sweeps n_frames, since spurious correlation in a short window is the only
thing keeping random below 100%.
"""
import sys
from pathlib import Path

import numpy as np

VG = Path("/mnt/data/wangqq/vggt")
sys.path.insert(0, str(VG / "scripts"))
from diag_gauge_invariance import ReconsEval, gt_traj_for  # noqa: E402

RECONS = Path("/mnt/data/wangqq/recons_eval")
TUM = RECONS / "data/tum"
CLEAN = VG / "outputs_attack_geometry_aware_tum10/tum10_clean_uniform_l3"

rec = ReconsEval(RECONS)


def random_frac(gt_traj, gt_xyz, n_frames, draws=50, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        c2w = np.tile(np.eye(4), (n_frames, 1, 1))
        c2w[:, :3, 3] = rng.normal(scale=1.0, size=(n_frames, 3))
        try:
            vals.append(rec.ate(rec.evo_utils.get_tum_poses(c2w), gt_traj, True, True))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else float("nan")


WORK = Path("/tmp/survey_work")
WORK.mkdir(parents=True, exist_ok=True)


def frames_for(scene, n):
    """Uniform selection over the 90-frame subset, same rule the pipeline uses."""
    rgb = sorted((TUM / scene / "rgb_90").glob("*.png"))
    if not rgb:
        return None
    return np.linspace(0, len(rgb) - 1, n).round().astype(int).tolist()


scenes = sorted(p.name for p in TUM.iterdir() if p.is_dir())
print(f"{'序列':<40}{'GT半径=上限(m)':>15}{'随机ATE(m)':>12}{'随机/上限':>10}")
print("-" * 80)
for s in scenes:
    idx = frames_for(s, 10)
    if idx is None:
        print(f"{s:<40}  没有 rgb_90")
        continue
    try:
        gt_traj, gt_c2w = gt_traj_for(rec, TUM, s, idx, "groundtruth_90.txt", WORK / s)
    except Exception as e:
        print(f"{s:<40}  跳过: {type(e).__name__} {e}")
        continue
    xyz = np.asarray(gt_c2w, dtype=np.float64)[:, :3, 3]
    ceiling = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
    r = random_frac(gt_traj, xyz, len(xyz))
    print(f"{s:<40}{ceiling:>15.4f}{r:>12.4f}{r/ceiling:>10.1%}")

print("\n\n=== 帧数对随机基线的影响（sitting_xyz）===")
print("随机预测与 GT 零相关；帧数越多伪相关越弱，随机占比应越贴近 100%")
print(f"{'帧数':<8}{'上限(m)':>12}{'随机ATE(m)':>12}{'随机/上限':>10}")
print("-" * 42)
for n in (5, 10, 20, 40, 90):
    idx = frames_for("rgbd_dataset_freiburg3_sitting_xyz", n)
    gt_traj, gt_c2w = gt_traj_for(rec, TUM, "rgbd_dataset_freiburg3_sitting_xyz", idx,
                                  "groundtruth_90.txt", WORK / f"n{n}")
    xyz = np.asarray(gt_c2w, dtype=np.float64)[:, :3, 3]
    ceiling = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
    r = random_frac(gt_traj, xyz, len(xyz))
    print(f"{n:<8}{ceiling:>12.4f}{r:>12.4f}{r/ceiling:>10.1%}")
