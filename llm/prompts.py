import os

def load_prompt(experiment, filename):
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', experiment, filename)
    with open(prompt_path, 'r') as file:
        return file.read().strip()


# ============ PIPELINE 0
# Experiment 0
exp0_system_prompt = load_prompt('experiment0', 'exp0_system_prompt.txt')
exp0_user_prompt = load_prompt('experiment0', 'exp0_user_prompt.txt')

# ============ PIPELINE 1
# Experiment 1
exp1_system_prompt = load_prompt('experiment1', 'exp1_system_prompt.txt')
exp1_user_prompt = load_prompt('experiment1', 'exp1_user_prompt.txt')
exp1_specific_question = load_prompt('experiment1', 'exp1_specific_question.txt')

# ============ PIPELINE 2
# Experiment 2
exp2_system_prompt = load_prompt('experiment2', 'exp2_system_prompt.txt')
exp2_user_prompt = load_prompt('experiment2', 'exp2_user_prompt.txt')

# Experiment 3
exp3_system_prompt = load_prompt('experiment3', 'exp3_system_prompt.txt')
exp3_user_prompt = load_prompt('experiment3', 'exp3_user_prompt.txt')

# # Experiment 4
exp4_system_prompt = load_prompt('experiment4', 'exp4_system_prompt.txt')
exp4_user_prompt = load_prompt('experiment4', 'exp4_user_prompt.txt')

# ============ PIPELINE 3 -> NO LABEL
# Experiment 5
exp5_system_prompt = load_prompt('experiment5', 'exp5_system_prompt.txt')
exp5_user_prompt = load_prompt('experiment5', 'exp5_user_prompt.txt')


# ======== EXPERIMENT 6 -> FT

## MCQ
exp6_system_prompt_mcq= load_prompt('experiment6_ft_mcq', 'exp6_system_prompt_mcq.txt')
exp6_user_prompt_mcq = load_prompt('experiment6_ft_mcq', 'exp6_user_prompt_mcq.txt')

## XPL
exp6_system_prompt_xpl = load_prompt('experiment6_ft_xpl', 'exp6_system_prompt_xpl.txt')
exp6_user_prompt_xpl = load_prompt('experiment6_ft_xpl', 'exp6_user_prompt_xpl.txt')