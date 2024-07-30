print(f"Importing from {__file__}")

from .models import get_gpt3_model, get_gpt4o_model
from .utils import get_llms
from .prompts import load_prompt

print("Finished loading llm/__init__.py")

# # EXPERIMENT 0
# from llm.experiment0 import experiment0_llm_pipeline, process_llms_and_df_0
# # or

# # EXPERIMENT 1
# from llm.experiment1 import experiment1_llm_pipeline, process_llms_and_df
# from llm.experiment1b import experiment1_llm_pipeline_b, process_llms_and_df_b


