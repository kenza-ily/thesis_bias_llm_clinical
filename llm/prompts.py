import os

def load_prompt(experiment, filename):
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', experiment, filename)
    with open(prompt_path, 'r') as file:
        return file.read().strip()

# Load experiment1 prompts
exp1_system_prompt = load_prompt('experiment1', 'exp1_system_prompt.txt')
exp1_user_prompt = load_prompt('experiment1', 'exp1_user_prompt.txt')
exp1_specific_question = load_prompt('experiment1', 'exp1_specific_question.txt')

# # Load experiment2 prompts
# exp2_system_prompt = load_prompt('experiment2', 'system_prompt.txt')
# exp2_user_prompt = load_prompt('experiment2', 'user_prompt_1.txt')
# exp2_specific_question = load_prompt('experiment2', 'specific_question.txt')