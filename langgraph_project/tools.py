# from langgraph.prebuilt import create_react_agent  # agent factory
# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# import utils as ut
from langchain_community.tools import TavilySearchResults
from conf.configs import Cfg

configs_ = Cfg()
connection_string = configs_.database_configs.pg_connection_string

tool_search = TavilySearchResults(max_results=5)
