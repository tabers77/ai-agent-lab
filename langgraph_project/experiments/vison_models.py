# ------
# LINKS
# ------
# https://medium.com/@astropomeai/title-llama-3-vision-alpha-how-to-convert-llama-3-into-a-vision-model-2f078f0ed1bf
# https://ai.gopubby.com/unveiling-azure-openai-gpt-4-turbo-vision-visionary-model-representing-tomorrows-intelligence-6ebe2deedb32
# https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource


# -----------------
# IMPORTANT TO KNOW
# -----------------
# SET max_tokens to LLM


import base64
from langchain.chains.transform import TransformChain
from langchain_core.messages.human import HumanMessage
from pydantic import BaseModel, Field
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from taberspilotml.conf.llm_configs import Cfg
import taberspilotml.experimental.llms.lang_models as comps

from typing import List

cfg_instance = Cfg()

Cfg.llm_configs.llm_type = "azure_chat_openai"
Cfg.llm_configs.llm_deployment = "gpt4-vision"
Cfg.llm_configs.openai_api_version = "2024-02-15-preview"

llm = comps.get_llm_instance(configs=cfg_instance.llm_configs, max_tokens=1600)

# input_prompt = """
#
# Carefully examine the provided Partial Dependence Plot (PDP). Analyze the relationship between the feature (on the x-axis) and the model's predicted outcome (on the y-axis). Address the following points in your analysis:
#
# 1. **Overall Trend**: Identify whether the relationship is linear, non-linear, or stepwise. Describe the general direction of the trend (e.g., increasing, decreasing, or stable).
#
# 2. **Thresholds and Breakpoints**: Identify if there are any distinct thresholds, breakpoints, or non-linear regions where the model’s prediction changes significantly. Note any specific values or ranges where these shifts occur.
#
# 3. **Magnitude of Change**: Assess the size of the change in the model's predicted outcome as the feature value changes. Is the change in prediction gradual or abrupt?
#
# 4. **Anomalies or Irregular Patterns**: Look for any unusual patterns or outliers that deviate from the expected trend. For example, if the plot shows unexpected jumps, plateaus, or noise, highlight and explain these anomalies.
#
# 5. **Interpretation of Feature Influence**: Based on the trends, thresholds, and anomalies, explain how the feature affects the model’s prediction. What does this suggest about the relationship between the feature and the target variable? Does the feature have a strong, weak, or moderate influence on the prediction?
#
# 6. **Practical Implications**: Based on the analysis, provide insights into how the feature could impact decision-making or operations. Would increasing or decreasing this feature lead to significant changes in the prediction?
#
# Provide a detailed, data-driven analysis with high-quality insights. Aim for a comprehensive explanation that captures both the overall behavior and any nuances or unexpected findings in the plot.
#
# """

input_prompt = """
You are an expert analytics engineer focused on improving steam energy efficiency per unit of production. Your objective is to reduce 'PK2 Steam energy/production' (measured in GJ/t), which is shown on the y-axis of the provided plot.

   - Identify trends, non-linear effects, thresholds, and any anomalies.

   - Determine if increasing or decreasing the feature is likely to result in significant improvements in steam energy consumption.
   - Highlight any trade-offs, practical constraints, or interactions with other operational variables that might affect the feasibility of adjustments.

   - Assess whether the observed range of 'PK2 Steam energy/production' is within acceptable or typical levels for efficient operations.
   - Provide insights on whether optimizing this feature could meaningfully contribute to reducing steam energy consumption.
"""


def load_images(inputs: dict) -> dict:
    """Load multiple images from file paths and encode them as base64."""
    image_paths = inputs["image_paths"]  # Expecting a list of image paths

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Encode each image
    images_base64 = [encode_image(image_path) for image_path in image_paths]

    return {"images": images_base64}


load_image_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)


class ImageInformation(BaseModel):
    """Information about an image."""
    image_description: str = Field(description="a short description of the image")


parser = JsonOutputParser(pydantic_object=ImageInformation)

globals.set_debug(True)


@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with multiple images and a prompt."""

    model = llm

    # Prepare image messages
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        for image_base64 in inputs["images"]
    ]

    msg = model.invoke(
        [HumanMessage(
            content=[
                {"type": "text", "text": inputs["prompt"]},
                *image_messages,  # Append multiple images
            ]
        )]
    )

    # --------------------------------------------
    # Format the response in Markdown
    markdown_response = f"""
    **Model Response:**

    ```markdown
    {msg.content}
    ```
    """

    return markdown_response


def get_image_information(image_paths: List[str], prompt: str) -> dict:
    """Invoke the vision chain with multiple images."""
    vision_chain = load_image_chain | image_model  # | parser

    result = vision_chain.invoke({'image_paths': image_paths, 'prompt': prompt})

    return result


shap_prompt = """

Help me to interpret all the details of this shap plot, where the target var'PK2 Steam energy/production' . 
The goal is to reduce 'PK2 Steam energy/production'.


"""
image_path_1 = r"C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\data\img_1.png"
# image_path_2 = r"C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\data\img.png"

output = get_image_information(image_paths=[image_path_1],
                               prompt=shap_prompt
                               )

print(output)
