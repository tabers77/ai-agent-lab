"""This example uses react agent approach"""
import json

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.constants import END
from langgraph.types import Command

import logging

from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START

from langgraph_project.agents_nodes.custom_nodes import make_writing_node, make_research_node
from langgraph_project.multi_agents.AgentState import MultiState  # Using a custom state class
import langgraph_project.tools.tools as tools
import langgraph_project.multi_agents.temp_helpers as ut

import prompts.multiagents_prompts as prompts
from utils import llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------
# LLM SETTINGS
# -------------
research_llm = llm
writing_llm = llm

research_llm.temperature = 0.8  # higher temp for exploration
writing_llm.temperature = 0  # lower temp for focused writing


# ----------------
# TOOL TRACKERS
# ----------------

def create_tracker(tool_fn, min_calls: int = 1, max_calls: int = 10):
    return ut.ToolInvocationTracker(tool_fn, min_calls=min_calls, max_calls=max_calls)


search_tracker = create_tracker(tools.web_search)
summarize_tracker = create_tracker(tools.safe_fetch_and_summarize)


# ----------------
# AGENT FACTORY
# ----------------

# Note: Observe that with create_react_agent you cant guarantee the model must pick one of your registered tools.

def make_agent(model, tool_list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    return create_react_agent(model=model, tools=tool_list, prompt=prompt)




# TEST
# --- Meta-Planning Agent ---
planning_agent = make_agent(
    model=research_llm,
    tool_list=[],
    system_prompt=(
        "You're a planning agent.\n"
        "1) Break down the user's high-level objective into a list of independent subgoals.\n"
        "2) Return JSON: {{\"subgoals\": [string]}}."
    )
)
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
