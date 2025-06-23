import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.graph import END


def make_research_node(search_tool, summarize_tool, research_agent):
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
