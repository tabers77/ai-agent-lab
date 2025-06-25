import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

import langgraph_project.multi_agents.AgentState as States
from typing import (
    Any, Dict, Type, Union, Callable, TypeVar
)
from pydantic import BaseModel
from langgraph.types import Command


def make_research_node(search_tool, summarize_tool, research_agent):
    @validated_command_node(States.MultiState)  # TEST
    def research_node(state):
        """
        Executes research using search and summarization tools, ensuring minimum results.
        """
        # Reset trackers
        for tracker in (search_tool, summarize_tool):
            tracker.call_count = 0

        # Invoke agent for research
        res = research_agent.invoke(state)
        search_tool.assert_counts()
        summarize_tool.assert_counts()

        # Extract summaries
        research_summaries = [
            msg.content
            for msg in res["messages"]
            # Note: It is important to match the name
            if isinstance(msg, ToolMessage) and msg.name == 'safe_fetch_and_summarize'
        ]

        # Update state with results
        new_messages = state["messages"] + [
            AIMessage(content="Research complete.", name="research_node")
        ]
        update = {
            "research_results": research_summaries,
            "messages": new_messages
        }
        # -----------------------
        # CONDITIONAL TRANSITION:
        # -----------------------
        # If insufficient summaries, loop back
        if len(research_summaries) < 2:
            # optionally append a follow-up question to steer the agent
            update["messages"].append(
                AIMessage(
                    content="I only found one result; let me refine my queries.",
                    name="research_node"
                )
            )
            return Command(update=update, goto="research_node")
        else:
            return Command(update=update, goto="writing_node")

    return research_node


def make_writing_node(writing_agent, article_tool):
    def writing_node(state):
        """
        Generates the final article based on research summaries.
        """

        summaries = state.get("research_results", [])

        # NOTE: temporarily deactivated
        if len(summaries) < 2:
            raise ValueError("Insufficient research summaries to proceed to writing stage.")

        # Construct prompt for article generation
        payload = json.dumps({"topic": state.get("topic"), "summaries": summaries})
        human_msg = HumanMessage(content=f"Write an article with this data: {payload}")

        response = writing_agent.invoke({"messages": state["messages"] + [human_msg]})

        # Find generated article
        article = next(
            (msg.content for msg in response["messages"]
             if isinstance(msg, ToolMessage) and msg.name == article_tool.name),
            None
        )
        if not article:
            raise RuntimeError("generate_article tool did not return a result.")

        final_msgs = state["messages"] + [AIMessage(content="Article drafted.", name="writing_node")]
        print("DEBUG: 2. writing_node: final article text form writing node", article)

        return Command(update={"article": article, "messages": final_msgs}, goto=END)

    return writing_node


def validated_node(fn: Callable[[States.RefactorState], Dict[str, Any]]) -> Callable[
    [Union[States.RefactorState, Dict[str, Any]]], Dict[str, Any]]:
    def wrapper(raw_state: Union[States.RefactorState, Dict[str, Any]]) -> Dict[str, Any]:
        # Normalize to dict
        state_data = raw_state.model_dump() if isinstance(raw_state, BaseModel) else raw_state
        # Validate incoming state
        state = States.RefactorState(**state_data)
        # Run node logic
        updates = fn(state)
        # Merge and re-validate full state
        merged = state.copy(update=updates)
        merged = States.RefactorState(**merged.model_dump())
        # Return only changed fields
        diffs = {}
        for k, v in merged.model_dump().items():
            if state_data.get(k) != v:
                diffs[k] = v
        return diffs

    return wrapper


StateModel = TypeVar("StateModel", bound=Union[BaseModel, dict])


def validated_command_node(
        state_cls: Type[StateModel]
) -> Callable[[Callable[[StateModel], Command]], Callable[[Union[StateModel, Dict[str, Any]]], Command]]:
    """
    Wraps a node fn(state)->Command so that:
      - If state_cls is a Pydantic model, we run runtime validation on both incoming state
        and on the .update dict inside the Command.
      - If state_cls is a TypedDict (i.e. subclass of dict), we skip validation entirely.
    """

    def decorator(
            fn: Callable[[StateModel], Command]
    ) -> Callable[[Union[StateModel, Dict[str, Any]]], Command]:

        def wrapper(
                raw_state: Union[StateModel, Dict[str, Any]]
        ) -> Command:
            # 1) Normalize to a plain dict
            if isinstance(raw_state, BaseModel):
                state_data = raw_state.model_dump()
            else:
                state_data = raw_state  # already a mapping

            # 2) Validate incoming if it's a Pydantic class
            if issubclass(state_cls, BaseModel):
                state: StateModel = state_cls(**state_data)  # type: ignore
            else:
                # TypedDicts are just dict subclasses at runtime
                state = state_data  # type: ignore

            # 3) Run node logic; must return a Command
            result = fn(state)
            if not isinstance(result, Command):
                raise TypeError(
                    f"{fn.__name__} must return a Command, got {type(result)}"
                )

            # 4) Validate only the .update payload if using Pydantic
            if issubclass(state_cls, BaseModel):
                updates = result.update or {}
                merged = state.copy(update=updates)  # type: ignore
                # This will raise if any validator fails
                state_cls(**merged.model_dump())  # type: ignore

            # 5) Return the (validated) Command
            return result

        return wrapper

    return decorator
