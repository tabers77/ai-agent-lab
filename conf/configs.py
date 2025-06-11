from dataclasses import dataclass
from dataclasses_json import dataclass_json
from langchain_text_splitters import CharacterTextSplitter
import langchain_text_splitters as t_splitters
from dotenv import load_dotenv
import os


@dataclass_json
@dataclass(frozen=True)
class ChatbotConfigs:
    load_dotenv()  # Load environment variables from .env file
    embeddings_deployment: str = "text-embedding-ada-002"
    llm_deployment: str = "langchain_model"
    llm_type: str = 'azure_openai'
    openai_api_version: str = "2023-07-01-preview"

    storage_type: str = 'blob'
    blob_container_name: str = "policies-container"
    blob_connection_string = os.getenv("CONNECTION_STRING")
    pg_connection_string: str = f"postgresql://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    pg_collection_name: str = "state_of_union_vectors2"

    is_authentication_for_prod: bool = True
    saver_version: bool = False

    answer_prompt: str = None
    text_splitter: CharacterTextSplitter = CharacterTextSplitter
    embeddings_type: str = 'azure_openai'


class LLMConfigs:
    load_dotenv()  # Load environment variables from .env file
    key: str = os.getenv("AZURE_OPENAI_API_KEY")  # Fetch the key from .env
    endpoint: str = os.getenv('AZURE_OPENAI_ENDPOINT')
    embeddings_deployment: str = "text-embedding-ada-002"
    llm_deployment: str = "gpt-4o-HR-Test"  # "gpt-4o-mini"#"langchain_model"
    llm_type: str = 'azure_chat_openai'
    embeddings_type: str = 'azure_openai'
    openai_api_version: str = "2024-05-01-preview"  # "2023-07-01-preview"
    return_retriever: bool = True
    text_splitter: t_splitters = None
    use_dynamic_prompt_recognizer: bool = True


class DatabaseConfigs:
    storage_type: str = 'blob'
    blob_container_name: str = "policies-container"
    blob_connection_string = os.getenv("CONNECTION_STRING")
    pg_connection_string: str = f"postgresql+psycopg://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    pg_collection_name: str = None


class FlaskAppConfigs:
    is_authentication_for_prod: bool = True
    saver_version: bool = False


class PromptConfigs:
    answer_prompt: str = None
    agent_template_standard: str = None


class WikiConfigs:
    organization_url: str = 'https://dev.azure.com/storaenso-data-services'
    project_name: str = 'Azure Data Platform'  # 'Azure Data Platform', 'Data Science Products and Projects'
    wiki_identifier: str = '2c16fdea-8163-4a0f-b9ac-42231128570e'  # 'a06910c4-0843-4bea-aae1-38a104ef6d37' , '2c16fdea-8163-4a0f-b9ac-42231128570e'
    personal_access_token: str = os.getenv("AZURE_DEVOPS_PAT")


@dataclass_json
@dataclass(frozen=True)
class Cfg:
    llm_configs: LLMConfigs = LLMConfigs()
    database_configs: DatabaseConfigs = DatabaseConfigs()
    flask_app_configs: FlaskAppConfigs = FlaskAppConfigs()
    prompt_configs: PromptConfigs = PromptConfigs()
    wiki_configs: WikiConfigs = WikiConfigs()
