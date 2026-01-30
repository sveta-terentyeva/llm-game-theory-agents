from llmgt.agents.llm import LLMAgent
from llmgt.llm import ScriptedLLMClient
from llmgt.games import PrisonersDilemma


def test_llm_agent_act_parses_action_token():
    g = PrisonersDilemma()

    client = ScriptedLLMClient(outputs=[
        "Let's do (C,C).",
        "D",
    ])
    agent = LLMAgent(name="dummy-llm", client=client)

    msg = agent.send_message(g, [])
    assert "C" in msg

    action = agent.act(g, [])
    assert action in g.actions()


def test_llm_agent_act_fallback_on_bad_output():
    g = PrisonersDilemma()

    client = ScriptedLLMClient(outputs=["nonsense output that has no valid token"])
    agent = LLMAgent(name="dummy-llm", client=client)

    action = agent.act(g, [])
    assert action in g.actions()
