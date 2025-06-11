import asyncio
import os

from conf.configs import Cfg
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_core.tools import FunctionTool
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport*")

cfg = Cfg()
cfg.llm_configs.llm_deployment = 'gpt-4o'

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=cfg.llm_configs.llm_deployment,
    model="gpt-4o",
    api_version=cfg.llm_configs.openai_api_version,
    azure_endpoint=cfg.llm_configs.endpoint,
    api_key=cfg.llm_configs.key,  # Ensure you have this API key or the correct token provider
    # azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
    # extra_params={"max_tokens": 1000}  # Reduce response size
)

# web_surfer = FunctionTool(
#     MultimodalWebSurfer, description="WebSurfer"
# )


web_surfer = MultimodalWebSurfer("WebSurfer", az_model_client)
web_surfer_tool = FunctionTool(web_surfer.run, name="WebSurfer", description="Multimodal web search and browsing tool.")

# Ensure compatibility with Windows event loop
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def main() -> None:
    # Define the CustomerResearcher agent: Responsible for market analysis and identifying counter-arguments.
    customer_researcher = AssistantAgent(
        name="CustomerResearcher",
        model_client=az_model_client,
        tools=[web_surfer_tool],
        description="Conducts market analysis and identifies potential counter-arguments.",
        system_message=(
            "You are the CustomerResearcher. Your responsibilities include:\n"
            "1. Retrieving and analyzing data on industry trends, customer demand forecasts "
            "(for customers like Valio, Rørosmeieriet AS, Arla), and competitor outlook (e.g., Tetra Pak).\n"
            "2. Identifying potential counter-arguments that Elopak might use (e.g., citing bulk purchasing benefits or long-term contracts).\n"
            "3. Listing and justifying the selected data sources (such as industry reports, news articles, and annual reports).\n"
            "Provide your analysis in a structured and detailed report."
        )
    )
    # customer_researcher.reflect_on_tool_outputs = True  # TEST Use this if there are other tools

    # Define the FinanceExpert agent: Responsible for cross-verification and proposing pricing scenarios.
    finance_expert = AssistantAgent(
        name="FinanceExpert",
        model_client=az_model_client,
        description="Analyzes internal data and suggests pricing scenarios along with counter-counterarguments.",
        system_message=(
            "You are the FinanceExpert. Your tasks include:\n"
            "1. Cross-verifying market findings provided by the CustomerResearcher.\n"
            "2. Proposing three pricing scenarios for the product segments: LPB Fresh, LPB Fresh Brown, and LPB Aseptic, "
            "with pricing percentages, cost analysis, and margin justifications.\n"
            "3. Developing counter-counterarguments that emphasize Stora Enso’s unique value proposition and market-driven cost increases.\n"
            "Provide your analysis in a structured and detailed format."
        )
    )

    # Define the ReportAgent: Responsible for synthesizing all the insights into a comprehensive negotiation report.
    report_agent = AssistantAgent(
        name="ReportAgent",
        model_client=az_model_client,
        description="Synthesizes inputs from both the CustomerResearcher and FinanceExpert into a comprehensive report.",
        system_message=(
            "You are the ReportAgent. Your task is to compile and synthesize the outputs from the CustomerResearcher "
            "and FinanceExpert into a comprehensive negotiation report. The report should include:\n"
            "1. A market analysis covering industry trends, customer demand forecasts, and competitor outlook.\n"
            "2. A cross-verification of the market analysis.\n"
            "3. Three proposed pricing scenarios with detailed justifications.\n"
            "4. Potential counter-arguments along with counter-counterarguments.\n"
            "When you are done generating the report, reply with TERMINATE."
        )
    )

    # Create a Round Robin team that includes all three agents.
    team = RoundRobinGroupChat(
        [customer_researcher, finance_expert, report_agent],
        max_turns=5  # Adjust max_turns as needed
    )

    # Define the negotiation task prompt.
    task_prompt = (
        "Generate a negotiation report for Stora Enso's pricing discussions. The report should include:\n"
        "1. A market analysis with trends, customer demand forecasts (for customers like Valio, Rørosmeieriet AS, Arla), "
        "and competitor outlook (e.g., Tetra Pak).\n"
        "2. A cross-verification of the market analysis.\n"
        "3. Three pricing scenarios for 2026-2028 with justifications, including price increase percentages, cost analysis, "
        "and margin justifications for LPB Fresh, LPB Fresh Brown, and LPB Aseptic segments.\n"
        "4. Identification of potential counter-arguments from Elopak and corresponding counter-counterarguments.\n"
        "Synthesize all the information into a comprehensive, structured report. When complete, reply with TERMINATE."
    )

    # Run the team conversation stream and display it using the Console UI.
    stream = team.run_stream(task=task_prompt)
    await Console(stream)


# Run the main asynchronous function.
# asyncio.run(main())

def run_main_safely():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
        loop.run_until_complete(web_surfer._playwright.stop())  # correct way to call an async method
    except Exception:
        pass
    finally:
        loop.close()


run_main_safely()
