import numpy as np

def bias_score(emb, gender_direction, word_importance, gender_words):
    bias = 0
    words = list(word_importance.keys())
    for word in words:
        if word not in gender_words:
            importance = word_importance[word]
            bias += max(0, np.dot(emb, gender_direction)) * importance
    return bias / len(words)