"""Tests for Theorem Verification (Theorems 4, 5, 6)."""

import numpy as np
import pytest
from physattest.security.theorems import (
    verify_theorem_4, verify_theorem_5, verify_theorem_6,
)


class TestTheorem4:
    def test_verified(self):
        result = verify_theorem_4(n=4, n_trials=200)
        assert result["verified"]
        assert result["trials_beating_optimal"] == 0

    def test_optimal_eigenvalues_equal(self):
        result = verify_theorem_4(n=4, n_trials=10)
        eigvals = result["K_star_eigenvalues"]
        assert all(v == pytest.approx(eigvals[0]) for v in eigvals)

    def test_bound_positive(self):
        result = verify_theorem_4(n=4, n_trials=10)
        assert result["optimal_bound"] > 0


class TestTheorem5:
    def test_verified(self):
        np.random.seed(42)
        result = verify_theorem_5(n_trials=2000)
        assert result["verified"]

    def test_d_kl_positive(self):
        result = verify_theorem_5(n_trials=100)
        assert result["d_kl"] > 0

    def test_evasion_decreases_with_n(self):
        np.random.seed(42)
        result = verify_theorem_5(n_trials=2000)
        ns = sorted(result["per_n_results"].keys())
        evasions = [result["per_n_results"][n]["empirical_evasion"] for n in ns]
        # Evasion should generally decrease (allow some noise)
        assert evasions[-1] <= evasions[0] + 0.05

    def test_n_min_reasonable(self):
        result = verify_theorem_5(n_trials=100)
        assert result["n_min_99_percent"] > 0
        assert result["n_min_99_percent"] < 1000


class TestTheorem6:
    def test_verified(self):
        result = verify_theorem_6()
        assert result["verified"]

    def test_all_scenarios_safe(self):
        result = verify_theorem_6()
        for key, r in result["results"].items():
            assert r["safe"], f"scenario {key} should be safe"

    def test_path_count(self):
        result = verify_theorem_6()
        assert result["total_paths"] > 0

    def test_no_compromise_has_most_paths(self):
        result = verify_theorem_6()
        none_result = result["results"][("none",)]
        assert none_result["valid_paths_remaining"] == result["total_paths"]

    def test_single_compromise_has_valid_path(self):
        result = verify_theorem_6()
        for key, r in result["results"].items():
            if len(r["compromised"]) == 1:
                assert r["valid_paths_remaining"] > 0

    def test_double_compromise_has_valid_path(self):
        result = verify_theorem_6()
        for key, r in result["results"].items():
            if len(r["compromised"]) == 2:
                assert r["valid_paths_remaining"] > 0
