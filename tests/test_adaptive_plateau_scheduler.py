import math

from attack_vggt_geometry_tum10 import (
    normalized_worst_task_score,
    update_adaptive_plateau_state,
)


def _state(lr=2e-3):
    return {
        "lr": lr,
        "best_score": -float("inf"),
        "best_update": 0,
        "plateau_reference": -float("inf"),
        "bad_audits": 0,
        "stopped": False,
        "current_update": 0,
    }


def _step(state, score):
    return update_adaptive_plateau_state(
        state,
        score,
        factor=0.5,
        lr_patience=3,
        stop_patience=5,
        min_lr=1e-5,
        improvement_threshold=0.005,
    )


def test_worst_task_score_is_minimum_log_ratio():
    score, ratios = normalized_worst_task_score(
        {"pose": 2.0, "depth": 4.0, "track": 5.0},
        {"pose": 4.0, "depth": 6.0, "track": 10.0},
    )
    assert set(ratios) == {"depth", "pose", "track"}
    assert math.isclose(ratios["depth"], 1.5, rel_tol=1e-7)
    assert math.isclose(ratios["pose"], 2.0, rel_tol=1e-7)
    assert math.isclose(ratios["track"], 2.0, rel_tol=1e-7)
    assert math.isclose(score, math.log(1.5), rel_tol=1e-7)


def test_improvement_resets_patience_and_records_best_update():
    state = _state()
    state["current_update"] = 100
    state, event = _step(state, 0.1)
    assert event == "improved"
    assert state["bad_audits"] == 0
    assert state["best_score"] == 0.1
    assert state["best_update"] == 100


def test_three_plateaus_halve_learning_rate():
    state = _state()
    state, _ = _step(state, 0.1)
    for update, score in ((200, 0.101), (300, 0.102), (400, 0.103)):
        state["current_update"] = update
        state, event = _step(state, score)
    assert event == "reduce_lr"
    assert math.isclose(state["lr"], 1e-3)
    assert state["bad_audits"] == 0
    assert not state["stopped"]


def test_min_lr_stops_after_five_plateaus():
    state = _state(lr=1e-5)
    state, _ = _step(state, 0.1)
    for index in range(5):
        state["current_update"] = (index + 2) * 100
        state, event = _step(state, 0.1)
    assert event == "early_stop"
    assert state["stopped"]


def test_small_numerical_best_does_not_reset_plateau_patience():
    state = _state()
    state, _ = _step(state, 0.1)
    state["current_update"] = 200
    state, event = _step(state, 0.102)
    assert event == "none"
    assert state["best_score"] == 0.102
    assert state["best_update"] == 200
    assert state["bad_audits"] == 1


def test_cumulative_small_gains_eventually_reset_patience():
    state = _state()
    state, _ = _step(state, 0.1)
    state["current_update"] = 200
    state, event = _step(state, 0.102)
    assert event == "none"
    state["current_update"] = 300
    state, event = _step(state, 0.106)
    assert event == "improved"
    assert state["bad_audits"] == 0
    assert state["plateau_reference"] == 0.106


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"PASS {len(tests)} adaptive scheduler tests")
