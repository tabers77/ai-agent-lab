"""All utility functions"""

# from langchain_community.llms import HuggingFaceHub
# from langchain_community.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI


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
            temperature=0,
            # TODO: ADD THIS TO CONFIGS
            model_name="gpt-3.5-turbo",  # To specify the version name is necessary for CustomConversationBufferMemory
            *args,
            **kwargs
        )


    # else:
    #     raise ValueError('LLM type is not recognized')
