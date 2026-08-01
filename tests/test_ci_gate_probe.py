def test_ci_gate_is_actually_enforcing():
    """Temporary probe: proves the CI workflow turns red on a failing test."""
    assert 1 == 2
