"""Per-game baseline configurations for rule-based workflow agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llmgt.games.base import Game
from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.games.stag_hunt import StagHunt
from llmgt.games.battle_of_sexes import BattleOfSexes
from llmgt.games.ultimatum import UltimatumGame

from llmgt.agents.workflow import WorkflowProposerAgent, WorkflowResponderAgent


@dataclass(frozen=True)
class WorkflowConfig:
    """
    A simple per-game baseline configuration for "workflow" mode:
    - proposer_pair: what agent_a proposes
    - responder_preferred_pair: what agent_b would counter-propose if proposal is bad
    - responder_min_payoff: minimum payoff for responder to accept
    - fallback_action_b: action used by agent_b if no acceptance is reached
    """
    proposer_pair: tuple[str, str]
    responder_preferred_pair: Optional[tuple[str, str]]
    responder_min_payoff: Optional[float]
    fallback_action_b: str


def workflow_config_for_game(game: Game) -> WorkflowConfig:
    """
    Choose a baseline workflow config that creates non-trivial behavior
    across different games (so plots are not all identical).
    """

    # Prisoner's Dilemma:
    # Responder wants at least 2 payoff (so they won't accept (D,C)=0 etc).
    if isinstance(game, PrisonersDilemma):
        return WorkflowConfig(
            proposer_pair=(game.C, game.C),
            responder_preferred_pair=(game.C, game.C),
            responder_min_payoff=2.0,
            fallback_action_b=game.D,
        )

    # Stag Hunt:
    # Responder is ok with H,H (payoff 3) but prefers S,S (payoff 4).
    if isinstance(game, StagHunt):
        return WorkflowConfig(
            proposer_pair=(game.S, game.S),
            responder_preferred_pair=(game.S, game.S),
            responder_min_payoff=3.0,
            fallback_action_b=game.H,
        )

    # Battle of the Sexes:
    # Responder prefers their favorable equilibrium: (F,F) gives B=2.
    if isinstance(game, BattleOfSexes):
        return WorkflowConfig(
            proposer_pair=(game.O, game.O),            # A proposes their preferred
            responder_preferred_pair=(game.F, game.F), # B counters with their preferred
            responder_min_payoff=1.5,                  # won't accept 1 payoff outcomes <1.5
            fallback_action_b=game.F,
        )

    # Ultimatum:
    # Agent A proposes Fair (F,A). Responder accepts only if they get >=2.
    if isinstance(game, UltimatumGame):
        return WorkflowConfig(
            proposer_pair=(game.F, game.A),
            responder_preferred_pair=(game.F, game.A),
            responder_min_payoff=2.0,
            fallback_action_b=game.R,
        )

    # Default fallback
    actions = game.actions()
    if len(actions) < 2:
        raise ValueError(f"Game has too few actions: {actions}")

    return WorkflowConfig(
        proposer_pair=(actions[0], actions[0]),
        responder_preferred_pair=None,
        responder_min_payoff=None,
        fallback_action_b=actions[0],
    )


def make_rule_based_workflow_agents(game: Game) -> tuple[WorkflowProposerAgent, WorkflowResponderAgent]:
    cfg = workflow_config_for_game(game)

    agent_a = WorkflowProposerAgent(
        name="wf_proposer",
        propose_pair=cfg.proposer_pair,
    )

    agent_b = WorkflowResponderAgent(
        name="wf_responder",
        fallback_action=cfg.fallback_action_b,
        preferred_pair=cfg.responder_preferred_pair,
        min_payoff=cfg.responder_min_payoff,
    )

    return agent_a, agent_b
