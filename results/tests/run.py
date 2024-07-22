import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from config.settings import TEMPERATURE
from config.llm_config import llms
from llm.models import get_gpt3_model
from llm.prompts import load_prompt
from llm.experiment1 import experiment1
from llm.utils import extract_price

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_loading():
    logger.info("Testing LLM loading...")
    try:
        model = get_gpt3_model()
        assert model is not None
        logger.info("LLM loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading LLM: {str(e)}")
        raise

def test_prompt_loading():
    logger.info("Testing prompt loading...")
    try:
        system_prompt = load_prompt('experiment1', 'system_prompt.txt')
        user_prompt = load_prompt('experiment1', 'user_prompt.txt')
        specific_question = load_prompt('experiment1', 'specific_question.txt')
        assert all([system_prompt, user_prompt, specific_question])
        logger.info("Prompts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        raise

def test_basic_inference():
    logger.info("Testing basic LLM inference...")
    try:
        model = get_gpt3_model()
        test_input = "What is the capital of France?"
        response = model.invoke(test_input)
        assert response is not None and "Paris" in response.content
        logger.info("Basic inference test passed.")
    except Exception as e:
        logger.error(f"Error in basic inference: {str(e)}")
        raise

def test_experiment1_function():
    logger.info("Testing experiment1 function...")
    try:
        model = get_gpt3_model()
        test_case = "A 35-year-old patient presents with fever and cough."
        test_question = "What is the most likely diagnosis?"
        test_options = "A. Common cold\nB. Influenza\nC. COVID-19\nD. Bacterial pneumonia"
        response, _, _, _, _, _, _, _ = experiment1(model, "system prompt", "user prompt", test_case, test_question, test_options, "gender")
        assert response is not None
        logger.info("experiment1 function test passed.")
    except Exception as e:
        logger.error(f"Error in experiment1 function: {str(e)}")
        raise

def run_all_tests():
    logger.info("Starting test suite...")
    try:
        test_llm_loading()
        test_prompt_loading()
        test_basic_inference()
        test_experiment1_function()
        logger.info("All tests passed successfully!")
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
    finally:
        logger.info("Test suite completed.")

if __name__ == "__main__":
    run_all_tests()