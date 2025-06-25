"""This example uses react agent approach"""

import logging
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from langgraph_project.agents_nodes.custom_nodes import make_writing_node, make_research_node
from langgraph_project.multi_agents.AgentState import MultiState
import langgraph_project.tools.tools as tools
import langgraph_project.multi_agents.helpers as ut
import prompts.multiagents_prompts as prompts
from utils import llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm.temperature = 0.8

def create_tracker(tool_fn, min_calls: int = 1, max_calls: int = 10):
    return ut.ToolInvocationTracker(tool_fn, min_calls=min_calls, max_calls=max_calls)

search_tracker = create_tracker(tools.web_search)
summarize_tracker = create_tracker(tools.safe_fetch_and_summarize)

def make_agent(model, tool_list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    return create_react_agent(model=model, tools=tool_list, prompt=prompt)

research_agent = make_agent(
    model=llm,
    tool_list=[search_tracker.wrapped_tool, summarize_tracker.wrapped_tool],
    system_prompt=prompts.research_system_org,
)

writing_agent = make_agent(
    model=llm,
    tool_list=[tools.generate_article],
    system_prompt=prompts.writing_system_org,
)

research_node = make_research_node(
    search_tool=search_tracker,
    summarize_tool=summarize_tracker,
    research_agent=research_agent,
)
writing_node = make_writing_node(
    writing_agent=writing_agent,
    article_tool=tools.generate_article
)

builder = StateGraph(MultiState)
builder.add_edge(START, "research_node")
builder.add_node("research_node", research_node)
builder.add_node("writing_node", writing_node)
graph = builder.compile()

initial_state = {
    "messages": [HumanMessage(content="Write me an article on the future of AI.")],
    "topic": "X",
}
answer = graph.invoke(input=initial_state)