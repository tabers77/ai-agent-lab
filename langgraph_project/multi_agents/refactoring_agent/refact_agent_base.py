import langgraph_project.tools.tools as t

import re
import ast

from typing import Tuple

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from langgraph_project.agents_nodes.custom_nodes import validated_node
from langgraph_project.multi_agents.AgentState import RefactorState
from utils import llm

# # Send all LangGraph debug logs to the console
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(name)s %(levelname)s %(message)s"
# )
# # (Optional) just dial up LangGraph and down everything else
# logging.getLogger("langgraph").setLevel(logging.DEBUG)
# logging.getLogger("langchain").setLevel(logging.WARNING)


# ---------------
# DEFINE STATES
# ---------------
# TODO: MOVE TO STATES
# class RefactorState(TypedDict, total=False):
#     filename: str
#     task: str
#     original_code: str
#     refactored_raw: str
#     refactored_code: str
#     temp_path: str
#     exec_stdout: str
#     exec_stderr: str


# ---------------------------------------------------------
# TEST

# 1. Define a Pydantic model for state with validation

# TEST
from typing import Dict, Any

# 2. Helper: decorator to validate inputs/outputs of nodes


# ---------------------------------------------------------


builder = StateGraph(RefactorState)


# ----------------------
# DEFINE NODES & TOOLS
# ----------------------
# TODO: MOVE NODES TO A SEPARATE FILE
@validated_node
def read_file_node(state: RefactorState) -> Dict[str, Any]:
    print('Executing read_file_node...')
    #code = t.read_file(state["filename"])
    code = t.read_file(state.filename)  # TEST

    return {"original_code": code}


builder.add_node("read_file", read_file_node)


@validated_node
def refactor_llm_node(state: RefactorState) -> Dict[str, Any]:
    print('Executing refactor_llm_node...')
    prompt = (
        f"You are a Python assistant. The user wants to '{state.task}' in {state.filename}.\n"
        "Please output ONLY the full, valid Python code for the refactored file, with no markdown fences, no commentary, and no explanations. "
        "Here is the current code:\n```python\n"
        f"{state.original_code}\n```"

    )
    raw = llm.predict(prompt)
    return {"refactored_raw": raw}


builder.add_node("refactor_llm", refactor_llm_node)


@validated_node
def clean_validate_node(state: RefactorState) -> Dict[str, Any]:
    print('Executing clean_validate_node...')
    match = re.search(r"```(?:python)?\n([\s\S]*?)```", state.refactored_raw)
    code = match.group(1) if match else state.refactored_raw
    ast.parse(code)
    return {"refactored_code": code}


builder.add_node("clean_validate", clean_validate_node)


@validated_node
def write_temp_node(state: RefactorState) -> Dict[str, Any]:
    print('Executing write_temp_node...')
    temp = state.filename + ".refactored"
    t.write_file(temp, state.refactored_code)
    return {"temp_path": temp}


builder.add_node("write_temp", write_temp_node)


@validated_node
def execute_module_node(state: RefactorState) -> Dict[str, Any]:
    import sys
    from pathlib import Path
    import os
    import subprocess
    print('Executing execute_module_node...')

    # 1) Use the same Python interpreter as this process
    PYTHON = sys.executable
    test_file = r"C:\Users\delacruzribadenc\Documents\Repos\ai-agent-lab\langgraph_project\multi_agents\refactoring_agent\demo_file.py"

    # 2) Preserve venv site-packages while adding project root
    env = os.environ.copy()
    project_root = str(Path(state.filename).parent.parent)
    orig_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{orig_py}" if orig_py else project_root

    # 3) Execute the freshly written refactored file
    proc = subprocess.run(
        # [PYTHON, state["temp_path"]],
        [PYTHON, test_file],  # TEST
        capture_output=True,
        text=True,
        env=env,
    )

    # 4) If anything landed on stderr, halt the graph with that error
    if proc.stderr:
        return Command(update={"exec_stderr": proc.stderr}, goto=END)

    # 5) Otherwise return stdout
    return {"exec_stdout": proc.stdout}


builder.add_node("execute_module", execute_module_node)


@validated_node
def write_original_node(state: RefactorState) -> Dict[str, Any]:
    print('Executing write_original_node...')
    t.write_file(state.filename, state.filename)
    return {}


# --------------------------
# STATE GRAPH CONSTRUCTION
# --------------------------
builder.add_node("write_original", write_original_node)

# 9. Wire up edges for linear flow

builder.add_edge(START, "read_file")
builder.add_edge("read_file", "refactor_llm")
builder.add_edge("refactor_llm", "clean_validate")
builder.add_edge("clean_validate", "write_temp")
builder.add_edge("write_temp", "execute_module")
builder.add_edge("execute_module", "write_original")
builder.add_edge("write_original", END)

# 10. Compile the graph
refactor_graph = builder.compile()


# Usage helper
def refactor_file(filepath: str, task: str) -> Tuple[str, str]:
    initial = {"filename": filepath, "task": task}
    out = refactor_graph.invoke(initial)
    # If exec_stderr in state, return stderr
    stderr = out.get("exec_stderr", "")
    stdout = out.get("exec_stdout", "")
    return stdout, stderr


if __name__ == "__main__":
    fp = r"C:\Users\delacruzribadenc\Documents\Repos\ai-agent-lab\langgraph_project\multi_agents\refactoring_agent\demo_file.py"
    stdout, stderr = refactor_file(filepath=fp, task="for better performance")
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
