from llmgt.games import (
    PrisonersDilemma,
    StagHunt,
    BattleOfSexes,
    UltimatumGame,
)


def test_stag_hunt_payoffs():
    g = StagHunt()
    assert g.payoff("S", "S") == (4.0, 4.0)
    assert ("S", "S") in g.nash_equilibria()


def test_battle_of_sexes_nash():
    g = BattleOfSexes()
    assert ("O", "O") in g.nash_equilibria()
    assert ("F", "F") in g.nash_equilibria()


def test_ultimatum_game():
    g = UltimatumGame()
    assert g.payoff("F", "A") == (2.0, 2.0)
    assert ("L", "A") in g.nash_equilibria()
    assert ("F", "A") in g.pareto_optima()


def test_ultimatum_action_spaces_are_role_specific():
    g = UltimatumGame()
    assert set(g.actions_a()) == {"L", "F"}
    assert set(g.actions_b()) == {"A", "R"}


