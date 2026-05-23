from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AttackWeights:
    depth: float = 1.0
    pose: float = 0.2
    points: float = 0.5

    def as_loss_dict(self) -> dict[str, float]:
        return {"depth": self.depth, "pose": self.pose, "points": self.points}


@dataclass(frozen=True)
class AttackSpec:
    name: str
    attack_type: str = "global"
    steps: int = 10
    eps: float = 8 / 255
    alpha: float = 1 / 255
    random_start: bool = True
    patch_size: int = 96
    patch_alpha: float | None = None
    patch_x: int = -1
    patch_y: int = -1


@dataclass
class AttackResult:
    spec: AttackSpec
    adv_images: torch.Tensor
    history: list[dict[str, float]]
    patch: torch.Tensor | None = None
    patch_meta: dict[str, Any] | None = None


@cache
def _legacy():
    import attack_vggt_new1

    return attack_vggt_new1


def parse_patch_positions(text: str | None, default_x: int = -1, default_y: int = -1) -> list[tuple[int, int]]:
    if not text:
        return [(default_x, default_y)]

    positions: list[tuple[int, int]] = []
    for raw_token in text.replace(";", ",").split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token in {"center", "centre", "middle", "-1:-1"}:
            positions.append((-1, -1))
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid patch position {raw_token!r}. Use 'center' or 'x:y', "
                "for example: center,0:0,128:128"
            )
        x_text, y_text = token.split(":", 1)
        positions.append((int(x_text), int(y_text)))
    if not positions:
        raise ValueError("No valid patch positions were provided")
    return positions


def patch_position_name(x: int, y: int) -> str:
    if x < 0 and y < 0:
        return "patch_center"
    return f"patch_x{x}_y{y}"


def build_attack_specs(
    *,
    attack_type: str,
    steps: int,
    eps: float,
    alpha: float,
    random_start: bool,
    patch_size: int,
    patch_alpha: float | None,
    patch_x: int,
    patch_y: int,
    patch_positions: str | None = None,
) -> list[AttackSpec]:
    if attack_type == "patch":
        return [
            AttackSpec(
                name=patch_position_name(x, y),
                attack_type="patch",
                steps=steps,
                eps=eps,
                alpha=alpha,
                random_start=random_start,
                patch_size=patch_size,
                patch_alpha=patch_alpha,
                patch_x=x,
                patch_y=y,
            )
            for x, y in parse_patch_positions(patch_positions, patch_x, patch_y)
        ]

    return [
        AttackSpec(
            name="global",
            attack_type="global",
            steps=steps,
            eps=eps,
            alpha=alpha,
            random_start=random_start,
            patch_size=patch_size,
            patch_alpha=patch_alpha,
            patch_x=patch_x,
            patch_y=patch_y,
        )
    ]


def run_attack_spec(
    *,
    model,
    images: torch.Tensor,
    reference_preds: dict[str, torch.Tensor],
    dtype: torch.dtype,
    spec: AttackSpec,
    weights: AttackWeights,
) -> AttackResult:
    legacy = _legacy()
    if spec.attack_type == "patch":
        adv_images, history, patch, patch_meta = legacy.patch_attack(
            model=model,
            images=images,
            reference_preds=reference_preds,
            dtype=dtype,
            steps=spec.steps,
            alpha=spec.patch_alpha if spec.patch_alpha is not None else spec.alpha,
            patch_size=spec.patch_size,
            patch_x=spec.patch_x,
            patch_y=spec.patch_y,
            weights=weights.as_loss_dict(),
        )
        return AttackResult(
            spec=spec,
            adv_images=adv_images,
            history=history,
            patch=patch,
            patch_meta=patch_meta,
        )

    adv_images, history = legacy.pgd_attack(
        model=model,
        images=images,
        clean_preds=reference_preds,
        dtype=dtype,
        steps=spec.steps,
        eps=spec.eps,
        alpha=spec.alpha,
        random_start=spec.random_start,
        weights=weights.as_loss_dict(),
    )
    return AttackResult(spec=spec, adv_images=adv_images, history=history)


def save_attack_output(
    *,
    out_dir: Path,
    result: AttackResult,
    clean_images: torch.Tensor,
    adv_preds: dict[str, torch.Tensor],
    clean_preds: dict[str, torch.Tensor],
    image_paths: list[str],
    image_size_hw: tuple[int, int],
    weights: AttackWeights,
    save_images: bool = False,
) -> Path:
    legacy = _legacy()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_np = legacy.tensor_to_numpy(clean_preds, image_size_hw)
    adv_np = legacy.tensor_to_numpy(adv_preds, image_size_hw)
    npz_path = out_dir / "attack_outputs.npz"
    np.savez_compressed(
        npz_path,
        image_paths=np.array([Path(p).name for p in image_paths]),
        clean_images=clean_images.detach().cpu().numpy().astype(np.float16),
        adv_images=result.adv_images.detach().cpu().numpy().astype(np.float16),
        **{f"clean_{key}": value for key, value in clean_np.items()},
        **{f"adv_{key}": value for key, value in adv_np.items()},
    )

    meta = {
        "variant": result.spec.name,
        "attack_type": result.spec.attack_type,
        "attack_spec": asdict(result.spec),
        "weights": asdict(weights),
        "patch": result.patch_meta,
        "history": result.history,
        "npz": str(npz_path),
    }
    with open(out_dir / "attack_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    legacy.save_delta_preview(clean_images, result.adv_images, out_dir)
    legacy.save_patch_image(result.patch, out_dir)
    if save_images:
        legacy.save_adv_images(result.adv_images, image_paths, out_dir)
    return npz_path
