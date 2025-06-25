# Dynamic Tool Discovery & Registry Agent
# ----------------------------------------
# This code adds a registration_node into your StateGraph that:
# 1. Crawls the LangChain + LangGraph GitHub repos
# 2. Parses Python files for any `class ...Agent` or `@tool` definitions
# 3. Auto-registers these into the orchestrator's `state.tools_registry`

import os
import re
import logging

from git import Repo  # GitPython for cloning repos

from langchain_core.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from utils import llm
import prompts.multiagents_prompts as prompts

from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START

from langgraph_project.agents_nodes.custom_nodes import make_writing_node, make_research_node
from langgraph_project.multi_agents.AgentState import MultiState  # Using a custom state class
import langgraph_project.tools.tools as tools
import langgraph_project.multi_agents.helpers as ut

logger = logging.getLogger(__name__)

# -------------
# LLM SETTINGS
# -------------
research_llm = llm
writing_llm = llm


# ------------------
# Helper functions
# ------------------

def crawl_repo(repo_url: str, root_dir: str = None) -> list[str]:
    """
    Clone (if needed) and list all Python files in the given GitHub repo.
    """

    # 1) Derive the repo name from the URL
    repo_name = repo_url.rstrip('/').split('/')[-1]

    # 2) Compute parent folder (default "./repos") and actual clone path
    parent_dir = root_dir or os.path.join('.', 'repos')
    clone_dir = os.path.join(parent_dir, repo_name)

    # Clone repo if missing
    if not os.path.isdir(os.path.join(clone_dir, '.git')):
        os.makedirs(clone_dir, exist_ok=True)
        logger.info(f"Cloning {repo_url} into {clone_dir}")
        Repo.clone_from(repo_url, clone_dir)

    # Walk to collect .py files
    python_files = []
    for dirpath, _, filenames in os.walk(clone_dir):
        for fn in filenames:
            if fn.endswith('.py'):
                python_files.append(os.path.join(dirpath, fn))
    return python_files


def parse_agent_tools(file_paths: list[str]) -> list[dict]:
    """
    Parse each Python file and extract classes ending in 'Agent' or functions decorated with @tool.
    Returns a list of metadata dicts: {name, type, source}
    """
    registry = []
    class_pattern = re.compile(r'class\s+(?P<name>\w+Agent)\s*\(')
    tool_pattern = re.compile(r'@tool\s*\ndef\s+(?P<name>\w+)\s*\(')

    for path in file_paths:
        try:
            src = open(path, 'r', encoding='utf-8').read()
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue

        for match in class_pattern.finditer(src):
            registry.append({
                'name': match.group('name'),
                'type': 'agent_class',
                'source': path
            })
        for match in tool_pattern.finditer(src):
            registry.append({
                'name': match.group('name'),
                'type': 'tool_function',
                'source': path
            })

    print('DEBUG: Registry:', registry)
    return registry


# ------------------
# Registration Node
# ------------------

def registration_node(state):
    """
    Crawls repos and registers any new Agent/tools in state.tools_registry.

    state.tools_registry: a dict mapping tool names to metadata.
    """
    print('DEBUG1: Running registration_node')
    repo_urls = [
        'https://github.com/hwchase17/langchain',
        'https://github.com/langchain-ai/langgraph'
    ]

    all_files = []
    for url in repo_urls:
        files = crawl_repo(repo_url=url, root_dir=r'C:\Users\delacruzribadenc\Documents\Repos')
        all_files.extend(files)

    parsed = parse_agent_tools(all_files)
    print("DEBUG1: Parsed tools and agents:", parsed)

    # merge into existing registry
    registry = state.get('tools_registry', {})
    added = []
    for entry in parsed:
        if entry['name'] not in registry:
            registry[entry['name']] = entry
            added.append(entry['name'])

    print('DEBUG: Added new tools:', added)

    new_messages = state['messages'] + [AIMessage(content=f"Registered tools: {added}", name="registration_node")]
    update = {
        'tools_registry': registry,
        'messages': new_messages
    }

    # Proceed to research or planning phase
    return Command(update=update, goto='research_node')


def make_agent(model, tool_list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    return create_react_agent(model=model, tools=tool_list, prompt=prompt)


def create_tracker(tool_fn, min_calls: int = 1, max_calls: int = 10):
    return ut.ToolInvocationTracker(tool_fn, min_calls=min_calls, max_calls=max_calls)


search_tracker = create_tracker(tools.web_search)
summarize_tracker = create_tracker(tools.safe_fetch_and_summarize)

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
# Integrate into your graph
# --------------------------

builder = StateGraph(MultiState)
# Start with registration
builder.add_edge(START, 'registration_node')
builder.add_node('registration_node', registration_node)
builder.add_node('research_node', research_node)  # existing from your code
builder.add_node('writing_node', writing_node)  # existing from your code
builder.add_edge('registration_node', 'research_node')
builder.add_edge('research_node', 'writing_node')
builder.add_edge('writing_node', END)

# Compile and invoke as usual
graph = builder.compile()

initial_state = {
    'messages': [HumanMessage(content="Write me an article on the future of AI.")],
    'topic': 'future of AI',
    # initialize empty registry
    'tools_registry': {}
}

answer = graph.invoke(input=initial_state)
print(answer)
