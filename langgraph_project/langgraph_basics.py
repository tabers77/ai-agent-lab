"""Refer to https://langchain-ai.github.io/langgraph/concepts/why-langgraph/#learn-langgraph-basics"""

from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import utils as ut
from conf.configs import Cfg

load_dotenv()
# **************** Global Configurations ****************
use_memory = True

# **************** Global Configurations ****************

tool_search = TavilySearchResults(max_results=5)
# tools = [tool]

cfg_instance = Cfg()

cfg_instance.llm_configs.llm_deployment = "gpt-app"  # "langchain_model"
cfg_instance.llm_configs.openai_api_version = "2024-02-15-preview"  # Use this version for gpt4 # "2023-07-01-preview"

llm = ut.get_llm_instance(configs=cfg_instance.llm_configs)


# ---------------
# ADD STATE CLASS
# ---------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# -----------------------
# ADD FIRST NODE AND EDGE
# -----------------------

## The first argument is the unique node name
## The second argument is the function or object that will be called whenever
## the node is used.

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
# graph = graph_builder.compile()


# # -----------------
# # ADD TOOLS TO LLM
# # -----------------
# llm_with_tools = llm.bind_tools(tools)
#
# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}
#
# graph_builder.add_node("chatbot", chatbot)

# -------------
# RUN THE TOOLS
# -------------
# NOTE: If you do not want to build this yourself in the future, you can use LangGraph's prebuilt ToolNode.


import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool_search])
graph_builder.add_node("tools", tool_node)


# ------------------------
# DEFINE CONDITIONAL EDGES
# ------------------------

def route_tools(
        state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# graph = graph_builder.compile()

# -----------
# ADD MEMORY 1
# -----------


from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph1 = graph_builder.compile(checkpointer=memory) if use_memory else graph_builder.compile()


# -------------
# ASK QUESTIONS
# -------------

def stream_graph_updates(user_input_: str, graph_, use_memory=False):
    if not use_memory:
        for event in graph_.stream({"messages": [{"role": "user", "content": user_input_}]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
    else:
        config = {"configurable": {"thread_id": "1"}}

        for event in graph_.stream({"messages": [{"role": "user", "content": user_input_}]},
                                   config,
                                   stream_mode="values",

                                   ):
            event["messages"][-1].pretty_print()


# # **** RUN COMMAND ****
# def run_command():
#     while True:
#         try:
#             user_input = input("User: ")
#             if user_input.lower() in ["quit", "exit", "q"]:
#                 print("Goodbye!")
#                 break
#             stream_graph_updates(user_input, graph1, use_memory)
#         except:
#             # fallback if input() is not available
#             user_input = "What do you know about LangGraph?"
#             print("User: " + user_input)
#             stream_graph_updates(user_input, graph1, use_memory)
#             break

# run_command()

# -----------
# BASE AGENT
# -----------

def base_agent(llm):
    # TODO: MOVE STATE CLASS TO A SEPARATE FILE
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    tools = [tool_search]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")

    tool_node = BasicToolNode(tools=[tool_search])
    graph_builder.add_node("tools", tool_node)

    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory) if use_memory else graph_builder.compile()

    def execute_command():
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                stream_graph_updates(user_input, graph, use_memory)
            except:
                # fallback if input() is not available
                user_input = "What do you know about LangGraph?"
                print("User: " + user_input)
                stream_graph_updates(user_input, graph, use_memory)
                break

    execute_command()

