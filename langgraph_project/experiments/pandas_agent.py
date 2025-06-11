"""PANDAS"""

import taberspilotml.experimental.llms.lang_models as comps
import taberspilotml.experimental.llms.agents_tools as llmtools
import pandas as pd
import taberspilotml.experimental.llms.prompts.pre_subfixes as pre_sub_prompts

# from taberspilotml.conf.llm_configs import Cfg
# import langchain_text_splitters as t_splitters


# ----------------------------------------------------------------------------------------------------
path = r'C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\data\preprocessed.csv'

path_sample = r'C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\data\sample_df.csv'
df1 = pd.read_csv(path,
                  index_col=0)

sample = pd.read_csv(path_sample,
                     index_col=0)

prep = pd.read_csv(
    r'C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\experimental\llms_results\preprocessed_dataset.csv',
    index_col=0)


def column_selector(df, target):
    """Selects columns based on the target variable."""
    col_exclusions = {
        'PK2 Steam energy/production': ['Steam specific consumption', 'PK2 Steam energy/production~^0',
                                        'PK2 6kV Energy/production',
                                        'PK2 Steam energy/production * PK2 Production speed from quality system'],
        'Steam specific consumption': ['PK2 Steam energy/production~^0', 'PK2 Steam energy/production'],
        'PK2 Steam energy/production~^0': ['PK2 Steam energy/production', 'Steam specific consumption']
    }

    if target not in col_exclusions:
        raise ValueError("Invalid target column name.")
    cols = list(set(df.columns) - set(col_exclusions[target]))

    return df[cols]


df1 = column_selector(df1, 'PK2 Steam energy/production')


# ------------------
# INITIALIZE AGENTS
# ------------------


def user_input_footer(user_input):
    mandatories = """
  **Important Final Step:**
  - Save all scripts used for the analysis in a Python file named **scripts.py**.
  - Organize the file with well-commented sections that follow the analysis workflow, clearly explaining each script.  
  - Store the file in the directory **llms_results**.
  - Confirm that the script has been successfully saved before completing the task.
    """

    user_input += mandatories
    return user_input.strip()


# dfs = {'df1': df1, 'prep': prep}
#
# constructor = pre_sub_prompts.PrefixSuffixConstructor(multi_datasets=False)
# prefix, suffix = constructor.build_prompt()
#
# agent_custom = llmtools.create_pandas_dataframe_agent_custom_experimental(llm=llm,
#                                                                           dfs=df1,  # dfs,
#                                                                           prefix=prefix,
#                                                                           # pre_sub_prompts.PREFIX_PANDAS_DF,
#                                                                           suffix=suffix,
#                                                                           # pre_sub_prompts.SUFFIX_PANDAS,
#                                                                           return_intermediate_steps=False,
#                                                                           verbose=True,
#                                                                           max_iterations=50
#                                                                           )

# test_input = """
# Problem Overview: You are working on a predictive modeling task to forecast steam consumption over time. The target variable is "PK2 Steam energy/production" (measured in GJ/t), which is already provided in the dataset.
#
# The best baseline models so far have identified the following top features (based on feature importance) as the most significant:
#
# Features to Remove: From the dataset, you should remove the following columns as they are redundant or problematic:
#
# 'Steam specific consumption'
# 'PK2 Steam energy/production~^0'
# 'PK2 6kV Energy/production'
# 'PK2 Steam energy/production * PK2 Production speed from quality system'
# 'PK2 5 bar steam'
# 'PK2 2 bar steam'
# Currently, these features are resulting in an R² value of 43%.
#
# Your Tasks:
#
# Data Analysis and Quality Check:
# Analyze each feature in the dataset and examine its distribution.
# Investigate whether any of the features contain outliers, zero values, or other anomalies that might affect the model’s performance.
# Ensure that the feature distributions make sense given the context of the problem.
#
# Partial Dependence Plot Investigation:
# Review the Partial Dependence Plots (PDPs) for the top features identified by the baseline model:
#
# PK2 Production speed from quality system (t/h) - Importance: 0.418077
# 4th press line pressure (kN/m) - Importance: 0.246065
# Hood exhaust air humidity 2 (g/kg) - Importance: 0.123898
# Hood replacement air 2 temperature, control (%) - Importance: 0.106357
# PK2 POPE Moisture A (%) - Importance: 0.105604
# We have used a RandomForest Regressor. These plots currently do not make sense based on the data you have. Investigate what could be causing these discrepancies.
#
# Dataset Accuracy:
# Ensure that the dataset is free of issues that could negatively impact model accuracy. This includes checking for:
# Incorrect data types (e.g., numeric columns being treated as categorical).
# Missing or incorrectly labeled data.
# Inconsistent values that could distort the feature relationships.
#
# Finally, create a detailed summary of all your analysis and provide recommendations on what to do next.
#
# """

import taberspilotml.experimental.llms.prompts.input_cases_examples as input_prompts

# user_input_ = user_input_footer(test_input)
# print('user_input_', user_input_)
#
# chat_agents = llmtools.ConversationalAgents()
#
# final_answer = chat_agents.chat_with_agent(
#     agent=agent_custom,
#     user_input=user_input_  # optional
# )


## ------------------- TEST PEAKON SURVEY GOLDEN DATASET GENERATOR -------------------

prefix = pre_sub_prompts.PREFIX_PANDAS
suffix = pre_sub_prompts.SUFFIX_PANDAS

# -------------------------------
# path = r"C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\experimental\llms_results\anonymized_dataset.csv"
#
# df = pd.read_csv(path)
#
# print()
# -------------------------------


path = r"C:\Users\delacruzribadenc\Documents\Repos\autopilot\taberspilotml\data\Peakon Comments - Sample.csv"

df = pd.read_csv(path)

agent_peakon = llmtools.create_pandas_dataframe_agent_custom_experimental(llm=llm,
                                                                          dfs=df,  # dfs,
                                                                          prefix=prefix,
                                                                          # pre_sub_prompts.PREFIX_PANDAS_DF,
                                                                          suffix=suffix,
                                                                          # pre_sub_prompts.SUFFIX_PANDAS,
                                                                          return_intermediate_steps=False,
                                                                          verbose=True,
                                                                          max_iterations=50
                                                                          )

# user_input_ = (
# "Analyze this dataset. The goal is to create a similar dataset with 200 rows, replacing any names or sensitive "
# "information in the 'Comment' column without losing context. Analyze the content of the 'Comment' column to generate "
# "similar content. Save the results in the llms_results directory"
# )

user_input_ = (
    "1. Load and Analyze the Dataset:\n\n"
    "Load the provided dataset, paying special attention to the 'Comment' column.\n\n"
    "Analyze the 'Comment' column to identify key themes, context, and common patterns.\n"
    "(For example, themes related to teamwork, cooperation, and work dynamics.)\n\n"

    "2. Generate a New Dataset:\n\n"
    "Create a new dataset with 200 rows that matches the structure of the original dataset.\n\n"
    "For the 'Comment' column, generate new comments that preserve the identified themes and context, "
    "while replacing any names or sensitive data with general, anonymized content.\n\n"
    "Ensure the generated content maintains the original sentiment and context without including any specific "
    "or sensitive information.\n\n"

    "3. Save the Results:\n\n"
    "Save the final generated dataset as a CSV file in the directory llms_results."
)
chat_agents = llmtools.ConversationalAgents()

final_answer = chat_agents.chat_with_agent(
    agent=agent_peakon,
    user_input=user_input_  # optional
)

# ------------------------------------------------------------
# TEST from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# ------------------------------------------------------------


# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#
# agent_executor = create_pandas_dataframe_agent(
#     llm,
#     df1,
#     agent_type="tool-calling",
#     verbose=True,
#     allow_dangerous_code=True
# )
#
# print(agent_executor.invoke({'input': u_i}))


# --------
# APPENDIX
# --------
