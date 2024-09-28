def get_word_importance_dict(text, model, tokenizer):
    word_importance = calculate_word_importance(text, model, tokenizer)
    return {word: importance for word, importance in word_importance}
