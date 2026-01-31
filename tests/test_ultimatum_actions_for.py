from llmgt.games import UltimatumGame


def test_ultimatum_actions_for_roles():
    g = UltimatumGame()
    assert g.actions_for("agent_a") == ("L", "F")
    assert g.actions_for("agent_b") == ("A", "R")
