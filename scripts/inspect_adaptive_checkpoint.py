#!/usr/bin/env python3
import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(sorted(data))
    print([(key, type(value).__name__) for key, value in data.items()])
    print(
        "best_shapes",
        [tuple(value.shape) for value in data.get("adaptive_best_parameters", [])],
    )
    texture = data.get("texture")
    print("texture_shape", tuple(texture.shape) if texture is not None else None)
    print("adaptive_state", data.get("adaptive_state"))


if __name__ == "__main__":
    main()
