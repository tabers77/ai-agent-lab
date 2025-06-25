from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent


def make_agent(model, tool_list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    return create_react_agent(model=model, tools=tool_list, prompt=prompt)
