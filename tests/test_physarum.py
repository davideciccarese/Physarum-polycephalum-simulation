"""pytest wrapper around the validation functions built into physarum_sim.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import physarum_sim as ps  # noqa: E402


def test_reinforcement_and_decay():
    assert ps.test_reinforcement_and_decay(verbose=False)


def test_disconnected_components():
    assert ps.test_disconnected_components(verbose=False)


def test_two_path_competition():
    r1 = ps.test_two_path_competition(1.0, verbose=False)
    r2 = ps.test_two_path_competition(2.0, verbose=False)
    assert r1 < 1e-2
    assert r2 < r1


def test_full_model_conservation():
    cfg = ps.make_config("A", H=40, W=60, n_steps=150, seed=3)
    sim = ps.PhysarumSimulation(cfg)
    sim.run()
    w = sim.worst
    assert w["residual"] < 1e-8
    assert w["divergence_err"] < 1e-8
    assert w["component_balance"] < 1e-10
    assert w["nutrient_accounting"] < 1e-9
    assert w["adjacency_violations"] == 0
    assert np.all(sim.S >= 0)
    assert np.all(np.isfinite(sim.D)) and np.all(sim.D > 0)
    assert sim.metrics["occupied_area"][-1] > sim.metrics["occupied_area"][0]
