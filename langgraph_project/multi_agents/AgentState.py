"""
Note: Avoid having local imports here to avoid circular imports.
"""
from typing import Annotated, Literal, Optional, Any, Dict

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict
from typing import List

from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


# ----------------
# STATE TEMPLATES
# ----------------
# These are templates
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "get_info",
                "appointment_info",
            ]
        ],
        update_dialog_stack,
    ]


class AgentState(BaseModel):
    # -- Core, always-present fields --
    step: str = Field(..., description="Current pipeline step")
    history: Optional[list[str]] = Field(default_factory=list, description="Log of events or messages")
    error: Optional[str] = Field(None, description="Last error, if any")

    # -- Catch-all for custom data per agent --
    data: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific payload")

    class Config:
        extra = "ignore"


# ----------------
# CUSTOM STATES
# ----------------
class MultiState(TypedDict, total=False):
    """
    Conversation state for the multi-agent pipeline.

    Optional fields:
      - topic: the article topic extracted or provided
      - research_results: list of summaries from the research agent
    """
    messages: Annotated[List[AnyMessage], add_messages]
    dialog_state: Annotated[
        List[
            Literal[
                "assistant",
                "get_info",
                "appointment_info",
            ]
        ],
        update_dialog_stack,
    ]
    topic: str
    research_results: List[str]


# --- define the shape of each research result ---
class ResearchResult(TypedDict):
    subgoal: str
    summaries: List[str]


class MultiState2(TypedDict, total=False):
    """
    Conversation state for the meta-planning → research → writing pipeline.

    Fields:
      - messages: the growing chat history
      - dialog_state: your existing dialog-stack tracking
      - goal: the user’s high-level objective
      - subgoals: list of sub-tasks produced by planning_node
      - research_results: for each subgoal, the summaries
      - article: final markdown string from writing_node
    """
    messages: Annotated[List[AnyMessage], add_messages]
    dialog_state: Annotated[
        List[
            Literal["assistant", "get_info", "appointment_info"]
        ],
        update_dialog_stack,
    ]
    goal: str
    subgoals: List[str]
    research_results: List[ResearchResult]
    article: str


class RefactorState(BaseModel):
    filename: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    original_code: Optional[str] = None
    refactored_raw: str = Field("", description="Raw LLM output with potential fences")
    refactored_code: str = Field("", description="Clean, validated Python code")
    temp_path: str = Field("", description="Path to temporary refactored file")
    exec_stdout: str = Field("", description="Captured stdout of execution")
    exec_stderr: str = Field("", description="Captured stderr of execution or errors")

    @field_validator("original_code")
    def original_code_not_empty(cls, v):
        # Only validate after read_file has set original_code
        if v is None:
            return v
        if not v.strip():
            raise ValueError("`original_code` must not be empty")
        return v
