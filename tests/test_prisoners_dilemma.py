from llmgt.games import PrisonersDilemma


def test_pd_payoffs():
    g = PrisonersDilemma()
    assert g.payoff("C", "C") == (3, 3)
    assert g.payoff("C", "D") == (0, 5)
    assert g.payoff("D", "C") == (5, 0)
    assert g.payoff("D", "D") == (1, 1)


def test_pd_theory_sets():
    g = PrisonersDilemma()
    assert ("D", "D") in g.nash_equilibria()
    assert ("C", "C") in g.pareto_optima()
