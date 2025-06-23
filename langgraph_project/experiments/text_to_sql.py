import utils
from conf.configs import Cfg
from langchain_community.utilities import SQLDatabase

# ---------------------------------
cfg_instance = Cfg()

cfg_instance.llm_configs.llm_deployment = "gpt-app"  # "langchain_model"
cfg_instance.llm_configs.openai_api_version = "2024-02-15-preview"  # Use this version for gpt4 # "2023-07-01-preview"

llm = utils.get_llm_instance(configs=cfg_instance.llm_configs)

# --------------------------------

# 1. Point it at your DB (no manual SQL)

db = SQLDatabase.from_uri(cfg_instance.database_configs.pg_connection_string)

# 2. Wrap it as a toolkit
#toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# from langchain import hub
#
# # 3. Pull & format the system prompt
# prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
#
# print('prompt_template',prompt_template)
# # it expects two parameters: your SQL dialect, and how many tables to show at once (top_k)
# system_message = prompt_template.format(
#     dialect=db.dialect,
#     top_k=5,
# )
# print('system_message',system_message)
# # 3. Spin up your agent
#
# #print(toolkit.get_context())
#
# # 4. Create your agent
# agent = create_react_agent(
#     model=llm,
#     tools=toolkit.get_tools(),
#     state_modifier=system_message
# )
#
query = "How many rows in total are in the langchain_pg_embedding table?"
#
# # 5. Run natural-language queries
# response = agent.invoke({"messages":[{"role":"user","content":query}]})
#
# print('response', response)


# OPTION 2
from langchain_community.agent_toolkits import create_sql_agent

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",  # or ZERO_SHOT_REACT_DESCRIPTION, etc.
    verbose=True,
)

# then exactly the same invoke:
response = agent.invoke(query)

print('response', response['output'])
