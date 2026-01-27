from pathlib import Path

from llmgt.games import PrisonersDilemma
from llmgt.agents.simple import FixedActionAgent
from llmgt.logging.jsonl_logger import JsonlLogger
from llmgt.sim.runner import run_episode, run_experiment, summarize_theory_hits


def test_episode_record_fills_fields():
    g = PrisonersDilemma()
    a = FixedActionAgent(name="fixed_C", action="C")
    b = FixedActionAgent(name="fixed_D", action="D")

    rec = run_episode(
        episode_id="t-1",
        game=g,
        agent_a=a,
        agent_b=b,
        mode="no_workflow",
        max_comm_rounds=0,
    )

    assert rec.episode_id == "t-1"
    assert rec.game == g.name
    assert rec.model_a == "fixed_C"
    assert rec.model_b == "fixed_D"

    assert rec.action_a == "C"
    assert rec.action_b == "D"
    assert (rec.payoff_a, rec.payoff_b) == (0.0, 5.0)

    assert rec.finished_at_utc is not None
    assert len(rec.messages) >= 3  # system + 2 actions


def test_experiment_logging_and_theory(tmp_path: Path):
    g = PrisonersDilemma()
    a = FixedActionAgent(name="fixed_D_A", action="D")
    b = FixedActionAgent(name="fixed_D_B", action="D")

    logger = JsonlLogger(out_dir=tmp_path, filename="episodes.jsonl")
    recs = run_experiment(game=g, agent_a=a, agent_b=b, n_episodes=5, logger=logger)

    p = tmp_path / "episodes.jsonl"
    assert p.exists()
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    stats = summarize_theory_hits(recs)
    # для PD: (D,D) має бути Nash
    assert stats["nash_rate"] == 1.0
    assert stats["agreement_rate"] == 1.0
