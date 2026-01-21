from llmgt.logging import EpisodeRecord, ChatMessage


def test_episode_record_serializes():
    rec = EpisodeRecord(
        episode_id="ep_test",
        game="pd",
        mode="no_workflow",
        max_comm_rounds=2,
        used_comm_rounds=0,
        model_a="dummy",
        model_b="dummy",
    )
    rec.messages.append(ChatMessage(role="agent_a", content="hello"))
    dumped = rec.model_dump()
    assert dumped["game"] == "pd"
    assert len(dumped["messages"]) == 1
