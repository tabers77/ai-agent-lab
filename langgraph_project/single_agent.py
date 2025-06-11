"""This script demonstrates how to use the LangGraph library to create a simple Single React Agent"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from conf.configs import Cfg

import utils as ut

# from langchain.agents.agent import AgentExecutor

# Initialize your LLM

tool_search = TavilySearchResults(max_results=5)
# tools = [tool]

cfg_instance = Cfg()

cfg_instance.llm_configs.llm_deployment = "gpt-app"  # "langchain_model"
cfg_instance.llm_configs.openai_api_version = "2024-02-15-preview"  # Use this version for gpt4 # "2023-07-01-preview"

llm = ut.get_llm_instance(configs=cfg_instance.llm_configs)


# 2. Define your “tool” agents

# -----------------------------
# SINGLE REACT AGENT WITH TOOLS
# -----------------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web for `query` and return a brief summary.
    (Here we stub it out; you could hook in SerpAPI, GoogleSearch, etc.)
    """
    # In a real app you’d call an API here
    return f"Top search result summary for '{query}'."


@tool
def square_number(number: float) -> float:
    """Return the square of the given number."""
    return number * number


@tool
def fallback_agent(state: Annotated[dict, InjectedState]) -> str:
    """
    Handle any other user inputs.
    InjectedState gives us the full conversation state (including "input").
    """
    user_input = state["input"]
    return f"I’m a generalist. You said: '{user_input}'."


# 2) Build the supervisor
agent = create_react_agent(
    model=llm,
    tools=[web_search, square_number, fallback_agent],
    prompt="You are a helpful assistant that can search the web and do math.",
    debug=True,
    name="agent"
)

query = "Please use your square_number tool to square 4."

# 3) Invoke it properly
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": query}
        ]
    }
)

# 2) Check for any ToolMessage
from langchain_core.messages.tool import ToolMessage

called = any(isinstance(m, ToolMessage) for m in result["messages"])
print("Tools were called!" if called else "No tools called.")

# 4) Extract reasoning vs. final answer
all_messages = result["messages"]  # List of BaseMessage
final_answer_msg = all_messages[-1]  # the last AIMessage
reasoning_msgs = all_messages[:-1]  # everything before it

print("=== REASONING TRACE ===")
for msg in reasoning_msgs:
    pass
    # this will include system prompt (if any), tool calls, function outputs, etc.
    # print(f"{msg.role}: {msg.content!r}")

print("\n=== FINAL ANSWER ===")
print(final_answer_msg.content)
