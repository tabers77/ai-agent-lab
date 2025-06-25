"""   Plan‐and‐execute (Tree‐of‐Thought) """

import langgraph_project.tools.tools as tools

from langgraph_project.multi_agents.AgentState import State
from utils import llm

from langgraph.graph import StateGraph, START, END

import json

import langgraph_project.multi_agents.helpers as ut
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Guarantee tool invocation
# original tools
search_tool = tools.web_search
summarize_tool = tools.fetch_and_summarize

# trackers enforce 1–2 calls
search_tracker = ut.ToolInvocationTracker(search_tool, min_calls=1, max_calls=2)
summarize_tracker = ut.ToolInvocationTracker(summarize_tool, min_calls=1, max_calls=2)


# Plan‐and‐execute (Tree‐of‐Thought)
# --- 1) Define a small schema for planning ---


class QueryPlan(BaseModel):
    queries: list[str] = Field(
        ...,
        description="Exactly the 1–2 search queries you plan to run"
    )


# --- 2) Create a parser and get format instructions ---
plan_parser = PydanticOutputParser(pydantic_object=QueryPlan)
plan_instructions = plan_parser.get_format_instructions().replace('{', '{{').replace('}', '}}')

# --- 3) Build a dedicated planning agent ---
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You’re a planning agent. Your job is to break the user’s topic into specific search queries (max 2).\n"
        "Output **only** valid JSON matching this schema:\n"
        f"{plan_instructions}")),
    ("placeholder", "{messages}")
])
planning_agent = create_react_agent(model=llm, tools=[], prompt=planning_prompt)


# ------------
# BUILD AGENTS
# ------------

# Note: Observe that with create_react_agent you cant guarantee the model must pick one of your registered tools.
def make_agent(llm, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    return create_react_agent(model=llm, tools=tools, prompt=prompt)


# -----------
# ORCHESTRATE
# -----------


# Plan‐and‐execute (Tree‐of‐Thought)

def research_node(state):
    # Reset trackers before each run
    search_tracker.call_count = 0
    summarize_tracker.call_count = 0

    # Phase 1: ask for a query plan
    plan_response = planning_agent.invoke({"messages": state["messages"]})
    plan_json = next(
        msg.content for msg in plan_response["messages"]
        if isinstance(msg, AIMessage)
    )

    plan = plan_parser.parse(plan_json)
    queries = plan.queries

    # Phase 2: execute each planned query using the *wrapped* tools
    research_summaries = []

    for query in queries:
        print('query', query)
        # NOTE: use the wrapped versions here!
        search_results = search_tracker.wrapped_tool(query)
        print('search_results', search_results)

        if not search_results:
            continue
        top_url = search_results[0]["url"]

        summary = summarize_tracker.wrapped_tool(top_url)
        research_summaries.append(summary)

    # Enforce that we did exactly the right number of calls
    search_tracker.assert_counts()
    summarize_tracker.assert_counts()

    # Update state and move on
    return Command(
        update={
            "research_results": research_summaries,
            "messages": state["messages"] + [
                AIMessage(content="Research complete.", name="research_node")
            ]
        },
        goto="writing_node"
    )


writing_agent = make_agent(
    llm=llm,
    tools=[tools.generate_article],
    system_prompt=(
        "You’re a writing agent. Your job is to take research summaries\n"
        "and craft a Medium-ready article, with headings, intro, conclusion,\n"
        "and a friendly yet authoritative tone.\n\n"
        "When you respond, **you must** call the `generate_article` tool exactly once\n"
        "with the arguments `topic` (string) and `research_summaries` (list of strings),\n"
        "and then return its result as your final output. Do not write any free-form text."
    )
)


def writing_node(state):
    # summaries = state["research_results"]
    summaries = state.get("research_results", [])
    print('***DEBUG***: summaries in writing_node', summaries)

    # Pass topic & summaries into generate_article via messages
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Summaries: {json.dumps(summaries)}\n"
        "Please write the article now."
    )
    # you could also use agent.invoke with explicit tool inputs
    res = writing_agent.invoke(
        {"messages": state["messages"] + [HumanMessage(content=prompt)]}
    )
    print('***DEBUG***: res in writing_node', res)

    article_text = None
    for msg in res["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "generate_article":
            article_text = msg.content
            break

    if article_text is None:
        raise RuntimeError("generate_article did not return a ToolMessage")

    print('***DEBUG***: article_text', article_text)

    return Command(
        update={
            "article": article_text,  # the markdown article from your tool
            "messages": state["messages"] + [
                AIMessage(content="Article drafted.", name="writing_node")
            ]
        },
        goto=END
    )


# ------------------------------
# Build and compile & add nodes
# ------------------------------

builder = StateGraph(State)
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

print('***DEBUG***: answer from graph.invoke', answer)
