"""All utility functions"""
from langchain_community.llms import AzureOpenAI, HuggingFaceHub

from langchain_openai import AzureChatOpenAI
from conf.configs import Cfg


def get_llm_instance(configs, *args, **kwargs):
    """
      Obtain the language model type based on the specified configuration.

      Args:
          configs : Configuration parameters.

      Returns:
          LLM: Language model based on the specified configuration.

      LLMs, or Large Language Models, are like supercharged text generators. They understand language,
      complete sentences, and can write various types of content. However, they are best suited for pure text-based
      tasks.

      If you need an AI that feels like a real conversation partner, then you need a Chat Model. Chat models
      understand context, remember past interactions, and are specifically designed for conversational flow. They are
      best suited for chatbots, virtual assistants, or any application requiring continuous dialogue.

      Usage:
          - For simple applications, using an LLM is suitable.
          - For applications requiring conversational flow, a Chat Model is recommended.
      """

    if configs.llm_type == 'azure_chat_openai':
        return AzureChatOpenAI(
            azure_deployment=configs.llm_deployment,
            openai_api_version=configs.openai_api_version,
            # temperature=0,
            # TODO: ADD THIS TO CONFIGS
            # model_name="gpt-3.5-turbo",  # To specify the version name is necessary for CustomConversationBufferMemory
            *args,
            **kwargs
        )

    elif configs.llm_type == 'azure_openai':
        return AzureOpenAI(
            openai_api_type="azure_ad",
            deployment_name=configs.llm_deployment  # Name of the deployment for identification
        )

    elif configs.llm_type == 'hugging_face':
        return HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    else:
        raise ValueError('LLM type is not recognized')


cfg_instance = Cfg()

cfg_instance.llm_configs.llm_deployment = "gpt-app"  # "langchain_model"
cfg_instance.llm_configs.openai_api_version = "2024-08-01-preview"  # "2024-02-15-preview"  # Use this version for gpt4 # "2023-07-01-preview"

llm = get_llm_instance(configs=cfg_instance.llm_configs)
