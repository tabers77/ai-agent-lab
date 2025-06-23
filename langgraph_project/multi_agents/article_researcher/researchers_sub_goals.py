"""This example shows Meta - Planning & Subgoal Decomposition"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts.chat import ChatPromptTemplate

from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

from langgraph_project.multi_agents.AgentState import MultiState2  # Using a custom state class
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

# TEST
def planning_node(state):
    """
    Decompose high-level goal into subgoals and store in state.
    """
    user_goal = state.get("goal")
    llm_input = [HumanMessage(content=user_goal)]
    res = planning_agent.invoke({"messages": llm_input})
    # Extract JSON list of subgoals
    text = next((m.content for m in res["messages"]
                 if isinstance(m, AIMessage)), None)
    data = json.loads(text)
    subgoals = data.get("subgoals", [])
    # Update state
    new_msgs = state["messages"] + [AIMessage(content="Planning complete.", name="planning_node")]
    return Command(
        update={"subgoals": subgoals, "messages": new_msgs},
        goto="research_node"
    )



# TEST
def research_node(state):
    "Perform research on the current subgoal"
    # Pop next subgoal
    subgoals = state["subgoals"]
    results = state.get("research_results", [])

    # 1) If there’s nothing left to research, go write
    if not subgoals:
        print('DEBUG: No subgoals left, transitioning to writing_node')
        return Command(update=state, goto="writing_node")

    # 2) Otherwise, pull the next subgoal and do your work
    current = subgoals.pop(0)

    print("DEBUG: subgoals in research_node", subgoals)


    # Reset trackers
    for tracker in (search_tracker, summarize_tracker):
        tracker.call_count = 0

    # inject subgoal prompt
    prompt = f"Research this specific task: {current}"
    print("DEBUG: prompt in research_node", prompt)

    msgs = state["messages"] + [HumanMessage(content=prompt)]
    res = research_agent.invoke({"messages": msgs})
    # Invoke agent for research
    search_tracker.assert_counts()
    summarize_tracker.assert_counts()

    # Extract summaries
    research_summaries = [
        msg.content
        for msg in res["messages"]
        # Note: It is important to match the name
        if isinstance(msg, ToolMessage) and msg.name == 'safe_fetch_and_summarize'
    ]

    # Update state with results
    new_messages = state["messages"] + [
        AIMessage(content="Research complete.", name="research_node")
    ]

    # 3) Accumulate this subgoal’s results
    results.append({
        "subgoal": current,
        "summaries": research_summaries
    })

    update = {
        "subgoals": subgoals,  # ⚡️ propagate the shortened list
        "research_results": results,  # ⚡️ accumulate
        "messages": new_messages
    }

    # 4) If this one “failed”, re-enqueue it for refinement
    # if len(research_summaries) < 2:
    #     subgoals.insert(0, current)
    #     update["messages"].append(
    #         AIMessage("Refining my queries…", name="research_node")
    #     )

    # 5) Always loop back here until subgoals is empty
    print("DEBUG: loop back in research node")
    return Command(update=update, goto="research_node")





def writing_node(state):
    """
    Generates the final article based on research summaries.
    """

    print("DEBUG: 1. writing_node invoked")

    summaries = state.get("research_results", [])

    # NOTE: temporarily deactivated
    # if len(summaries) < 2:
    #     raise ValueError("Insufficient research summaries to proceed to writing stage.")

    # Construct prompt for article generation
    payload = json.dumps({"topic": state.get("topic"), "summaries": summaries})
    human_msg = HumanMessage(content=f"Write an article with this data: {payload}")

    response = writing_agent.invoke({"messages": state["messages"] + [human_msg]})

    # Find generated article
    article = next(
        (msg.content for msg in response["messages"]
         if isinstance(msg, ToolMessage) and msg.name == tools.generate_article.name),
        None
    )
    if not article:
        raise RuntimeError("generate_article tool did not return a result.")

    final_msgs = state["messages"] + [AIMessage(content="Article drafted.", name="writing_node")]
    print("DEBUG: 2. writing_node: final article text form writing node", article)

    return Command(update={"article": article, "messages": final_msgs}, goto=END)


# --------------------------
# STATE GRAPH CONSTRUCTION
# --------------------------


# --- Build StateGraph ---

builder = StateGraph(MultiState2)
builder.add_node("planning_node", planning_node)
builder.add_node("research_node",  research_node)
builder.add_node("writing_node",   writing_node)

builder.add_edge(START,           "planning_node")
builder.add_edge("planning_node", "research_node")
builder.add_edge("research_node", "research_node")    # self-loop only
# <-- no more research_node → writing_node edge
builder.add_edge("writing_node",  END)
graph = builder.compile()


# --- Invoke ---
initial_state = {
    "messages": [],
    "goal": "Learn all techniques, methods and more of agents in LangChain and LangGraph",
    "subgoals": [],
    "research_results": []
}

answer = graph.invoke(input=initial_state)
print(answer)