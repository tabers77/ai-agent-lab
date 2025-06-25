"""TODO: ADD CODE THAT HAS NOT BEING IMPLEMENTED YET"""

# # --------------
# # DEFINE TOOLS
# # --------------
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

# ----------------
# # MEMORY AGENT:
# ----------------
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

#

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

