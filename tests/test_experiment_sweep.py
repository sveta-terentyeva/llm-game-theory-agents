from pathlib import Path

from llmgt.games import PrisonersDilemma
from llmgt.agents.simple import FixedActionAgent
from llmgt.experiments import run_comm_sweep, summarize_by_k, write_csv

def test_comm_sweep_runs():
    g = PrisonersDilemma()
    a = FixedActionAgent(name="A", action="D")
    b = FixedActionAgent(name="B", action="D")

    records = run_comm_sweep(
        game=g,
        agent_a=a,
        agent_b=b,
        k_values=[0, 1, 2],
        n_runs=3,
    )

    assert len(records) == 9
    assert all(r.game == g.name for r in records)

def test_summarize_by_k():
    g = PrisonersDilemma()
    a = FixedActionAgent(name="A", action="D")
    b = FixedActionAgent(name="B", action="D")

    records = run_comm_sweep(
        game=g,
        agent_a=a,
        agent_b=b,
        k_values=[0, 1],
        n_runs=2,
    )

    rows = summarize_by_k(records)

    assert len(rows) == 2
    for row in rows:
        assert row["agreement_rate"] == 1.0
        assert row["nash_rate"] == 1.0

def test_write_csv(tmp_path: Path):
    rows = [
        {"game": "pd", "K": 0, "agreement_rate": 0.5},
        {"game": "pd", "K": 1, "agreement_rate": 1.0},
    ]

    path = tmp_path / "out.csv"
    write_csv(rows, path)

    assert path.exists()
    text = path.read_text()
    assert "agreement_rate" in text
