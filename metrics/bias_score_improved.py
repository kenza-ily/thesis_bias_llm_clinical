

def improved_bias_score(emb, gender_direction, word_importance, gender_words, directional=True):
    bias = []
    words = list(word_importance.keys())
    for word in words:
        if word not in gender_words:
            importance = word_importance[word] / 100.0  # Divide by 100 to match the scale
            word_bias = torch.cosine_similarity(torch.tensor(emb), torch.tensor(gender_direction), dim=0)
            weighted_bias = word_bias * importance
            bias.append(weighted_bias.item())
    
    if directional:
        score_pos = np.sum([b for b in bias if b > 0])
        score_neg = np.sum([b for b in bias if b < 0])
        return score_pos, score_neg
    else:
        return np.sum([abs(b) for b in bias])