"""
Multi-agent base example.
IMPORTANT!!! Use this as base example to add new techniques and test approaches.
This example uses react agent approach which allows the agent to choose tools dynamically.
"""


import logging

from langgraph.graph import StateGraph, START

from utils import llm

from langchain_core.messages import HumanMessage
from langgraph_project.agents_nodes.agent_factory import make_agent
from langgraph_project.agents_nodes.custom_nodes import (
    make_research_node,
    make_writing_node,
)
from langgraph_project.multi_agents.AgentState import MultiState
import langgraph_project.multi_agents.helpers as ut
import langgraph_project.tools.tools as tools
import prompts.multiagents_prompts as prompts


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------
# 1. LLM SETTINGS
# ----------------
research_llm = llm
writing_llm = llm

research_llm.temperature = 0.8  # higher temp for exploration
writing_llm.temperature = 0  # lower temp for focused writing


# ----------------
# 2. TOOL TRACKERS
# ----------------
# Define tools needed
# Observe that here we could add nodes with tools

def create_tracker(tool_fn, min_calls: int = 1, max_calls: int = 10):
    return ut.ToolInvocationTracker(tool_fn, min_calls=min_calls, max_calls=max_calls)


search_tracker = create_tracker(tools.web_search)
summarize_tracker = create_tracker(tools.safe_fetch_and_summarize)


# ----------------
# 3. AGENT FACTORY
# ----------------

# Note: Observe that with create_react_agent you cant guarantee the model must pick one of your registered tools.


# # --- Meta-Planning Agent ---
# planning_agent = make_agent(
#     model=research_llm,
#     tool_list=[],
#     system_prompt=(
#         "You're a planning agent.\n"
#         "1) Break down the user's high-level objective into a list of independent subgoals.\n"
#         "2) Return JSON: {{\"subgoals\": [string]}}."
#     )
# )

research_agent = make_agent(
    model=research_llm,
    tool_list=[search_tracker.wrapped_tool, summarize_tracker.wrapped_tool],
    system_prompt=prompts.research_system_org,
)

writing_agent = make_agent(
    model=writing_llm,
    tool_list=[tools.generate_article],
    system_prompt=prompts.writing_system_org,
)

# ----------------
# NODE DEFINITIONS
# ----------------

research_node = make_research_node(
    search_tool=search_tracker,
    summarize_tool=summarize_tracker,
    research_agent=research_agent,
)
writing_node = make_writing_node(
    writing_agent=writing_agent,
    article_tool=tools.generate_article
)

# --------------------------
# STATE GRAPH CONSTRUCTION
# --------------------------
builder = StateGraph(MultiState)
builder.add_edge(START, "research_node")

builder.add_node("research_node", research_node)
builder.add_node("writing_node", writing_node)
graph = builder.compile()


# -------
# INVOKE
# -------

initial_state = {
    "messages": [HumanMessage(content="Write me an article on the future of AI.")],
    "topic": "X",
}
answer = graph.invoke(input=initial_state)

# print('***DEBUG***: answer from graph.invoke', answer)
