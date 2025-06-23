"""TODO: ADD CODE THAT HAS NOT BEING IMPLEMENTED YET"""

# # ------
# # TOOLS
# # ------
# # 1) Wrap your existing functions as LangChain Tools / OpenAI functions
# search_tool = Tool.from_function(
#     func=tools.web_search,
#     name="web_search",
#     description="Search the web for a query; returns a JSON list of top URLs."
# )
# summarize_tool = Tool.from_function(
#     func=tools.fetch_and_summarize,
#     name="fetch_and_summarize",
#     description="Fetch a URL's content and return a concise summary."
# )


#
# # -----------------------------------------------------------
# # MEMORY AGENT: OPTIONAL
# from langchain import LLMChain, PromptTemplate
# from langchain.tools import Tool
#
# # 1) Define a simple LLMChain that summarizes arbitrary text into bullets.
# summary_prompt = PromptTemplate(
#     template=(
#         "You are a helpful assistant. "
#         "Summarize the following conversation into a concise bullet list:\n\n"
#         "{conversation}"
#     ),
#     input_variables=["conversation"]
# )
# summarizer_chain = LLMChain(llm=llm, prompt=summary_prompt)
#
#
# # 2) Wrap it as a LangChain Tool that *takes text* (not a URL).
# def summarize_text_block(conversation: str) -> str:
#     return summarizer_chain.predict(conversation=conversation)
#
#
# summarize_tool = Tool.from_function(
#     func=summarize_text_block,
#     name="summarize_text",
#     description="Summarize a block of text (the full conversation) into a concise bullet list."
# )
#
# # 3) Use that in your memory_agent instead of fetch_and_summarize.
# memory_agent = create_react_agent(
#     model=llm,
#     tools=[summarize_tool],
#     prompt=ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a memory summarizer. Given a long conversation, "
#          "produce a concise summary of the key points (3–5 bullets)."
#          ),
#         ("placeholder", "{messages}")
#     ])
# )
#
#
# # ------------------
# # NODE DEFINITIONS
# # ------------------
# def memory_node(state):
#     # Only summarize if conversation is long
#     if len(state["messages"]) < 10:
#         return Command(goto="research_node")
#
#     # Invoke memory_agent to compress state["messages"]
#     res = memory_agent.invoke({"messages": state["messages"]})
#     summary = None
#     for msg in res["messages"]:
#         if isinstance(msg, ToolMessage) and msg.name == "summarize_text":
#             summary = msg.content
#             break
#     if not summary:
#         # If summarization failed, skip
#         return Command(goto="research_node")
#
#     # Update memory and prune messages to keep only recent context
#     pruned_history = state["messages"][-5:]
#     new_messages = [AIMessage(content=summary, name="memory_node")] + pruned_history
#
#     return Command(
#         update={
#             "memory": summary,
#             "messages": new_messages
#         },
#         goto="research_node"
#     )
#
#
# # -----------------------------------------------------------

# # ---------------
# # Using pydantic
# # --------------
# # TEST
# system_prompt = (
#     "You’re a research agent. Output **only** valid JSON matching this schema:\n"
#     f"{escaped_schema}\n\n"
#     "1) Break the user’s topic into specific search queries (max 2)\n"
#     "2) Use web_search (max 2 times)\n"
#     "3) Summarize with fetch_and_summarize\n"
#     "Ask follow-ups if ambiguous."
# )
#
# research_agent = create_react_agent(
#     model=llm,
#     tools=[tools.web_search, tools.fetch_and_summarize],
#     prompt=ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("placeholder", "{messages}")
#     ])
# )


# -----------
# ORCHESTRATE
# -----------


# def research_node(state):
#     res = research_agent.invoke(state)
#
#     print('***DEBUG***: state messages', state['messages'])
#
#     # res = research_agent.invoke({"messages": state["messages"]})
#
#     print('***DEBUG***: res from research agent', res)
#
#     # collect only the ToolMessage contents (these are your summaries)
#     research_summaries = [
#         msg.content
#         for msg in res["messages"]
#         if isinstance(msg, ToolMessage) and msg.name == "fetch_and_summarize"
#     ]
#
#     print('***DEBUG***: research_summaries', research_summaries)
#     # After gathering summaries, we return a Command that does two things:
#     # 1) Updates the shared `state` by:
#     #    • Storing our list of summaries under `"research_results"`, so downstream nodes can access them.
#     #    • Appending an AIMessage “Research complete.” to the `"messages"` list, preserving conversation context.
#     # 2) Tells the orchestrator to transition next into the `"writing_node"`.
#     return Command(
#         update={
#             "research_results": research_summaries,  # list of summaries
#             "messages": state["messages"] + [
#                 AIMessage(content="Research complete.", name="research_node")
#             ]
#         },
#         goto="writing_node"
#     )
#

# ---------------------------------------------------------------------------------------