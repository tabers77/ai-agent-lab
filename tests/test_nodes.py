import pytest
from unittest.mock import patch
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import END

# Import your helpers and the nodes/agents under test
import langgraph_project.multi_agents.helpers as ut
from langgraph_project.multi_agents.article_researcher.researchers_agent_base import (
    search_tracker,
    summarize_tracker,
    research_agent,
    writing_agent,
)
from langgraph_project.agents_nodes.custom_nodes import research_node, writing_node


class DummyResponse:
    def __init__(self, messages):
        self.messages = messages
    def __getitem__(self, key):
        if key == "messages":
            return self.messages
        raise KeyError(key)

    def __init__(self, messages):
        self.messages = messages


@pytest.fixture(autouse=True)
def reset_trackers():
    # ensure call counts start at a valid value for each test
    search_tracker.call_count = 1
    summarize_tracker.call_count = 1
    yield
    # optional teardown


@patch.object(summarize_tracker, 'assert_counts', new=lambda self: None)
@patch.object(search_tracker, 'assert_counts', new=lambda self: None)
@patch.object(research_agent, 'invoke')
def test_research_node_succeeds(mock_invoke):
    # Arrange: fake state and fake agent response with two summaries
    state = {
        "messages": [HumanMessage(content="Topic?", name=None)],
        "topic": "AI"
    }
    summaries = ["Summary A", "Summary B"]
    tool_msgs = [
        ToolMessage(content=summaries[0], name='safe_fetch_and_summarize', tool_call_id="1"),
        ToolMessage(content=summaries[1], name='safe_fetch_and_summarize', tool_call_id="2")
    ]
    # agent.invoke returns an object with .messages
    mock_invoke.return_value = DummyResponse(messages=state['messages'] + tool_msgs)

    # Act
    cmd = research_node(state)

    # Assert
    assert isinstance(cmd, Command)
    assert cmd.goto == "writing_node"
    assert cmd.update["research_results"] == summaries
    # messages should include original plus completion notice
    assert any(isinstance(m, AIMessage) and m.content == "Research complete." for m in cmd.update["messages"])


@patch.object(writing_agent, 'invoke')
def test_writing_node_succeeds(mock_invoke):
    # Arrange: fake state with research_results
    state = {
        "messages": [HumanMessage(content="Intro", name=None)],
        "topic": "AI",
        "research_results": ["S1", "S2"]
    }
    fake_article = "# Article Text"
    tool_msg = ToolMessage(content=fake_article, name='generate_article', tool_call_id="3")
    mock_invoke.return_value = DummyResponse(messages=state['messages'] + [tool_msg])

    # Act
    cmd = writing_node(state)

    # Assert
    assert isinstance(cmd, Command)
    assert cmd.goto == END
    assert cmd.update["article"] == fake_article
    assert any(isinstance(m, AIMessage) and m.content == "Article drafted." for m in cmd.update["messages"])
