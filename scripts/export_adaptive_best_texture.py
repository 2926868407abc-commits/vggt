#!/usr/bin/env python3
"""Export the best audited texture from an adaptive training checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    parameters = data.get("adaptive_best_parameters")
    if not parameters or len(parameters) != 1:
        raise RuntimeError(
            "Expected exactly one adaptive best parameter (the patch texture), "
            f"found {0 if parameters is None else len(parameters)}"
        )
    texture = parameters[0].detach().float().cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, texture=texture)
    state = data.get("adaptive_state", {})
    print(
        f"saved {output} shape={texture.shape} "
        f"best_update={state.get('best_update')} best_score={state.get('best_score')}"
    )


if __name__ == "__main__":
    main()
