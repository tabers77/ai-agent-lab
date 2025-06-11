
# -------------
# ADD MEMORY 2
# -------------

# from langgraph.checkpoint.memory import MemorySaver
#
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
#
# config = {"configurable": {"thread_id": "1"}}
#
# user_input = "Hi there! My name is Will."
#
# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [{"role": "user", "content": user_input}]},
#     config,
#     stream_mode="values",
# )
#
# # **** RUN COMMAND ****
# for event in events:
#     event["messages"][-1].pretty_print()

# # ----------------
# # HUMAN ASSISTANCE
# # ----------------
#
# from typing import Annotated
#
# from langchain_core.tools import tool
# from typing_extensions import TypedDict
#
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.types import Command, interrupt
#
#
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#
#
# graph_builder = StateGraph(State)
#
#
# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]
#
#
# # tool = TavilySearch(max_results=2)
# tools = [tool_search, human_assistance]
# llm_with_tools = llm.bind_tools(tools)
#
#
# def chatbot(state: State):
#     message = llm_with_tools.invoke(state["messages"])
#     assert (len(message.tool_calls) <= 1)
#     return {"messages": [message]}
#
#
# graph_builder.add_node("chatbot", chatbot)
#
# tool_node = ToolNode(tools=tools)
# graph_builder.add_node("tools", tool_node)
#
# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")
#
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
#
# # **** RUN COMMAND ****
# user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
# config = {"configurable": {"thread_id": "1"}}
#
# events = graph.stream(
#     {"messages": [{"role": "user", "content": user_input}]},
#     config,
#     stream_mode="values",
# )
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()
#
# human_response = (
#     "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
#     " It's much more reliable and extensible than simple autonomous agents."
# )
#
# human_command = Command(resume={"data": human_response})
#
# events = graph.stream(human_command, config, stream_mode="values")
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()
#
