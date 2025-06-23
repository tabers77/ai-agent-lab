""" https://langchain-ai.github.io/langgraph/agents/multi-agent/#supervisor
https://github.com/anurag-mishra899/Multi-Agents-Appointment-Booking/blob/main/Back-End/agents/base.py
"""
# env_path = 'Path to Env File'
# load_dotenv(env_path)
# from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph_project.multi_agents.AgentState import State
from langgraph.types import Command

from langgraph_project.tools.tools import book_hotel, book_flight
from temp_helpers import llm

import json

# cfg_instance = Cfg()
#
# cfg_instance.llm_configs.llm_deployment = "gpt-app"  # "langchain_model"
# cfg_instance.llm_configs.openai_api_version = "2024-08-01-preview"  # "2024-02-15-preview"  # Use this version for gpt4 # "2023-07-01-preview"
#
# llm = get_llm_instance(configs=cfg_instance.llm_configs)


# ---------------
# 1. DEFINE TOOLS
# ---------------
# from langchain_openai import ChatOpenAI
# from langgraph_supervisor import create_supervisor


# ------------
# Step-2: Define Agents Layer
# ------------

def create_agent(llm: AzureChatOpenAI, tools: list, system_prompt: str):
    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            ("placeholder", "{messages}"),
        ]
    )
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent


information_agent = create_agent(
    llm=llm,
    tools=[book_flight],
    system_prompt="You are specialized agent to provide information related to availbility of doctors or any FAQs related to hospital based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."
)

booking_agent = create_agent(
    llm=llm,
    tools=[book_hotel],
    system_prompt="You are specialized agent to set, cancel or reschedule appointment based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."
)


def information_node(state: State):
    result = information_agent.invoke(state)
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="information_node")
            ]
        },
        goto="supervisor",
    )


def booking_node(state: State):
    result = booking_agent.invoke(state)
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="booking_node")
            ]
        },
        goto="supervisor",
    )


# -------------------------------
# Step-3: Define supervisor Layer:
# -------------------------------
members_dict = {
    'information_node': 'specialized agent to provide information related to availbility of doctors or any FAQs related to hospital.',
    'booking_node': 'specialized agent to only to book, cancel or reschedule appointment'}
options = list(members_dict.keys()) + ["FINISH"]
worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in
                           members_dict.items()]) + '\n\nWORKER: FINISH \nDESCRIPTION: If User Query is answered and route to Finished'

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers.\n"
    "Your job is to route each user query to either `information_node`, `booking_node`, or `FINISH`.\n"
    "When you reply, you **must** return strictly valid JSON with exactly two fields:\n"
    "{\n"
    '  "next": "<information_node | booking_node | FINISH>",\n'
    '  "reasoning": "<a brief plain-text explanation>"\n'
    "}\n"
    "Do not include any extra text, Markdown, or commentary—only the JSON object.\n\n"
    "### Workers:\n"
    "WORKER: information_node\n"
    "DESCRIPTION: specialized agent to provide information related to availability of doctors or hospital FAQs.\n"
    "WORKER: booking_node\n"
    "DESCRIPTION: specialized agent to book, cancel, or reschedule appointments.\n"
    "WORKER: FINISH\n"
    "DESCRIPTION: if the user’s request is answered and no further action is needed.\n\n"
    "Always consider the current year is 2024.\n"
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH. and provide reasoning for the routing"""
    next: Annotated[
        Literal["information_node", "booking_node", "FINISH"], ..., "worker to route to next, route to FINISH"]
    reasoning: Annotated[str, ..., "Support proper reasoning for routing to the worker"]


def supervisor_node(state: State) -> Command:
    messages = [
                   {"role": "system", "content": system_prompt},
               ] + [state["messages"][-1]]

    result = llm.invoke(
        messages,
        response_format={"type": "json_object"})
    # llm.invoke(...) returns an AIMessage, so grab its .content directly
    raw = result.content

    # 2) Attempt to parse that JSON string:
    try:
        parsed: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        # If parsing fails, default to FINISH
        worker = "FINISH"
        reasoning = "Failed to parse JSON from supervisor; defaulting to FINISH."
    else:
        worker = parsed.get("next", "FINISH")
        if worker not in ("information_node", "booking_node", "FINISH"):
            worker = "FINISH"
        reasoning = parsed.get("reasoning", "")

    goto = END if worker == "FINISH" else worker

    # 3) Build and return the Command as before:
    return Command(
        goto=goto,
        update={
            "next": worker,
            "cur_reasoning": reasoning,
            # if it’s the first turn, include “query” and so on here
        },
    )


# -----------------------------------------
# Step-4: Connect All Nodes and Build Graph
# -----------------------------------------

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("information_node", information_node)
builder.add_node("booking_node", booking_node)
graph = builder.compile()

inputs = [
    HumanMessage(content='can you check and make a booking if any cosmetic dentist available on 8 August 2024 at 9 AM?')
]

config = {"configurable": {"thread_id": "1", "recursion_limit": 10}}

state = {'messages': inputs, 'id_number': 10232303}
answer = graph.invoke(input=state, config=config)

print(answer)
