import os

def load_prompt(experiment, filename):
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', experiment, filename)
    with open(prompt_path, 'r') as file:
        return file.read().strip()

# Load experiment1 prompts
exp1_system_prompt = load_prompt('experiment1', 'exp1_system_prompt.txt')
exp1_user_prompt = load_prompt('experiment1', 'exp1_user_prompt.txt')
exp1_specific_question = load_prompt('experiment1', 'exp1_specific_question.txt')