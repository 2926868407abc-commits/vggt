"""Run this repository's plain assertion tests when pytest is unavailable."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


TEST_MODULES = (
    "test_paper_baseline_losses",
    "test_paper_baseline_placements",
    "test_record_baseline_experiment",
    "test_auto_placement_tools",
    "test_shared_subset_training",
    "test_load_texture_shape",
)


def main() -> None:
    tests_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(tests_dir))
    passed = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        functions = sorted(
            (
                (name, function)
                for name, function in inspect.getmembers(module, inspect.isfunction)
                if name.startswith("test_")
                and not inspect.signature(function).parameters
            ),
            key=lambda item: item[0],
        )
        for _, function in functions:
            function()
            passed += 1
            print(f"PASS {module_name}.{function.__name__}")
    print(f"{passed} direct assertion tests passed")


if __name__ == "__main__":
    main()
