def load_prompt(filename):
    with open(f'llm/{filename}', 'r') as file:
        return file.read().strip()

system_prompt_1 = load_prompt('system_prompt_1.txt')
user_prompt_1 = load_prompt('user_prompt_1.txt')
specific_question = load_prompt('specific_question.txt')