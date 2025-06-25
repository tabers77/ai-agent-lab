
# # -----------
# # SUPERVISOR
# # -----------
# import subprocess
# import sys
#
# def run_flake8(path: str) -> str:
#     """Return the stdout+stderr of flake8 (empty string = no lint errors)."""
#     cmd = [sys.executable, "-m", "flake8", path]
#     proc = subprocess.run(cmd, capture_output=True, text=True)
#     return proc.stdout + proc.stderr
#
# def run_mypy(path: str) -> str:
#     """Return the stdout+stderr of mypy (empty string = no type issues)."""
#     cmd = [sys.executable, "-m", "mypy", path]
#     proc = subprocess.run(cmd, capture_output=True, text=True)
#     return proc.stdout + proc.stderr
#
# def run_pytest(path: str) -> bool:
#     """
#     Run pytest on just that file (or its containing tests).
#     Return True if exit code == 0.
#     """
#     cmd = [
#         sys.executable, "-m", "pytest",
#         "--maxfail=1", "--disable-warnings", "-q", path
#     ]
#     proc = subprocess.run(cmd, capture_output=True, text=True)
#     return proc.returncode == 0
#
#
# def create_critique_prompt(
#         original: str,
#         refactored: str,
#         lint: str,
#         types: str
# ) -> str:
#     msg = (
#         "The previous refactoring still has issues.\n"
#         "Original code:\n```python\n"
#         f"{original}\n```\n"
#         "Refactored code:\n```python\n"
#         f"{refactored}\n```\n"
#         "Please fix the following problems without adding commentary:\n"
#     )
#     if lint:
#         msg += f"* Lint errors:\n```\n{lint}\n```\n"
#     if types:
#         msg += f"* Type-check failures:\n```\n{types}\n```\n"
#     msg += "\nOutput ONLY the full, valid Python source.\n"
#     return msg
#
#
# # --- 1. Define the Supervisor node
#
# def supervise_node(state: RefactorState) -> Dict[str, Any]:
#     print("Executing supervise_node…")
#     temp = state["temp_path"]
#     lint_issues = run_flake8(temp)
#     type_issues = run_mypy(temp)
#     tests_ok = run_pytest(temp)
#
#     # If any checks fail, ask the LLM to refine again
#     if lint_issues or type_issues or not tests_ok:
#         critique = create_critique_prompt(
#             state["original_code"],
#             state["refactored_code"],
#             lint_issues,
#             type_issues
#         )
#         new_raw = llm.predict(critique)
#         # Update the raw refactor and loop back
#         return Command(
#             update={"refactored_raw": new_raw},
#             goto="refactor_llm"
#         )
#
#     # Otherwise proceed normally
#     return {}
#
#
# builder.add_node("supervise", supervise_node)

