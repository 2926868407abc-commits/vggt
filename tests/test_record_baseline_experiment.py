from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_baseline_experiment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("record_baseline_experiment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replace_block_is_idempotent_and_keeps_other_runs() -> None:
    module = _load_module()
    first = "<!-- BASELINE-RUN:a:START -->\nold\n<!-- BASELINE-RUN:a:END -->"
    second = "<!-- BASELINE-RUN:b:START -->\nkeep\n<!-- BASELINE-RUN:b:END -->"
    text = "# report\n\n" + first + "\n\n" + second + "\n"
    replacement = "<!-- BASELINE-RUN:a:START -->\nnew\n<!-- BASELINE-RUN:a:END -->"

    updated = module.replace_block(text, "a", replacement)

    assert updated.count("BASELINE-RUN:a:START") == 1
    assert "\nnew\n" in updated
    assert "\nold\n" not in updated
    assert second in updated


def test_fmt_uses_dash_only_for_missing_values() -> None:
    module = _load_module()
    assert module.fmt(None) == "—"
    assert module.fmt(1.23456, 3) == "1.235"
