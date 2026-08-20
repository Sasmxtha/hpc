"""Tests for Stackelberg Game (Theorem 4 foundation)."""

import numpy as np
import pytest
from physattest.security.stackelberg import StackelbergGame, GameConfig


@pytest.fixture
def game():
    return StackelbergGame()


@pytest.fixture
def small_game():
    return StackelbergGame(GameConfig(n_sensors=2, n_actuators=2))


class TestImpossibilityBound:
    def test_positive_for_valid_K(self, game):
        K = 2.0 * np.eye(6)
        bound = game.impossibility_bound(K)
        assert bound > 0
        assert np.isfinite(bound)

    def test_smaller_K_gives_larger_bound(self, game):
        K_strong = 5.0 * np.eye(6)
        K_weak = 1.0 * np.eye(6)
        assert game.impossibility_bound(K_weak) > game.impossibility_bound(K_strong)

    def test_singular_K_returns_inf(self, game):
        K = np.zeros((6, 6))
        bound = game.impossibility_bound(K)
        assert bound == float("inf")


class TestAttackerBestResponse:
    def test_attacks_weakest_direction(self, game):
        K = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        response = game.attacker_best_response(K)
        assert response["weakest_eigenvalue"] == pytest.approx(1.0)

    def test_attack_magnitude_respects_budget(self, game):
        K = 2.0 * np.eye(6)
        response = game.attacker_best_response(K)
        delta = response["delta"]
        energy = delta @ K @ delta
        assert energy <= game.config.attack_budget * 1.1

    def test_damage_positive(self, game):
        K = 2.0 * np.eye(6)
        response = game.attacker_best_response(K)
        assert response["damage"] > 0


class TestDefenderOptimal:
    def test_returns_positive_definite(self, game):
        result = game.defender_optimal_K()
        eigvals = np.linalg.eigvalsh(result["K_star"])
        assert np.all(eigvals > 0)

    def test_bound_finite(self, game):
        result = game.defender_optimal_K()
        assert np.isfinite(result["bound"])

    def test_small_game_works(self, small_game):
        result = small_game.defender_optimal_K()
        assert result["K_star"].shape == (2, 2)
        assert np.isfinite(result["bound"])


class TestEquilibrium:
    def test_nash_equilibrium_holds(self, game):
        np.random.seed(42)
        eq = game.verify_equilibrium()
        assert eq["is_equilibrium"], "no random K perturbation should beat K*"

    def test_equilibrium_bound_matches_impossibility(self, game):
        defender = game.defender_optimal_K()
        eq = game.verify_equilibrium()
        assert eq["equilibrium_bound"] == pytest.approx(defender["bound"], abs=1e-6)


class TestEnergyCost:
    def test_energy_ratio_positive(self, game):
        energy = game.energy_cost_analysis()
        assert energy["energy_ratio"] > 0

    def test_min_energy_per_unit_positive(self, game):
        energy = game.energy_cost_analysis()
        assert energy["min_energy_per_unit"] > 0


class TestCouplingEffect:
    def test_damage_inversely_proportional_to_lambda_min(self):
        n = 6
        K = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        config = GameConfig(coupling_matrix=K)
        g = StackelbergGame(config)
        response = g.attacker_best_response(K)
        expected_damage = g.config.attack_budget / response["weakest_eigenvalue"]
        assert response["damage"] == pytest.approx(expected_damage, rel=0.01)
