"""Simulation engine: episode runner, agreement/theory logic, run directories."""

from .runner import run_episode, run_experiment, summarize_experiment, summarize_theory_hits
from .agreement import agreement_hit
from .theory import compute_theory_hits, theory_target_set, TheoryHits
from .run_dir import make_run_dir, RunDir

__all__ = [
    "run_episode",
    "run_experiment",
    "summarize_experiment",
    "summarize_theory_hits",
    "agreement_hit",
    "compute_theory_hits",
    "theory_target_set",
    "TheoryHits",
    "make_run_dir",
    "RunDir",
]

