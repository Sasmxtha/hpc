"""Tests for the LangGraph agent graph flow."""

import pytest
from physattest.agents.state import DefenseLevel, AlertSeverity


class TestDefenseLevel:
    def test_ordering(self):
        assert DefenseLevel.L1_MULTI_DOMAIN < DefenseLevel.L2_ACTIVE_PROBING
        assert DefenseLevel.L2_ACTIVE_PROBING < DefenseLevel.L3_FINGERPRINTING
        assert DefenseLevel.L3_FINGERPRINTING < DefenseLevel.L4_CBF_BOUNDING

    def test_comparison_with_int(self):
        assert DefenseLevel.L2_ACTIVE_PROBING >= 2
        assert DefenseLevel.L1_MULTI_DOMAIN < 2


class TestAlertSeverity:
    def test_ordering(self):
        assert AlertSeverity.NONE < AlertSeverity.LOW
        assert AlertSeverity.LOW < AlertSeverity.MEDIUM
        assert AlertSeverity.HIGH < AlertSeverity.CRITICAL

    def test_values(self):
        assert AlertSeverity.NONE == 0
        assert AlertSeverity.CRITICAL == 4


class TestGraphRouting:
    def test_should_probe_at_l2(self):
        from physattest.agents.graph import _should_probe
        state = {"defense_level": DefenseLevel.L2_ACTIVE_PROBING}
        assert _should_probe(state) == "prober"

    def test_should_not_probe_at_l1(self):
        from physattest.agents.graph import _should_probe
        state = {"defense_level": DefenseLevel.L1_MULTI_DOMAIN}
        assert _should_probe(state) == "check_fingerprint"

    def test_should_fingerprint_at_l3(self):
        from physattest.agents.graph import _should_fingerprint
        state = {"defense_level": DefenseLevel.L3_FINGERPRINTING}
        assert _should_fingerprint(state) == "fingerprint"

    def test_should_not_fingerprint_at_l1(self):
        from physattest.agents.graph import _should_fingerprint
        state = {"defense_level": DefenseLevel.L1_MULTI_DOMAIN}
        assert _should_fingerprint(state) == "guardian"

    def test_default_defense_level(self):
        from physattest.agents.graph import _should_probe
        state = {}
        assert _should_probe(state) == "check_fingerprint"


class TestGraphBuild:
    def test_graph_compiles(self):
        from physattest.agents.graph import build_physattest_graph
        graph = build_physattest_graph()
        assert graph is not None
