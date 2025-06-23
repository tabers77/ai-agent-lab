from typing import Annotated, Literal, Optional
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


# -----------------------------------------
# TEST
# --- define the shape of each research result ---
class ResearchResult(TypedDict):
    subgoal: str
    summaries: List[str]

# --- updated conversation state ---
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


# -----------------------------------------
