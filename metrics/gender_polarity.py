def gender_polarity(emb, gender_direction):
    return np.dot(emb, gender_direction) / np.linalg.norm(gender_direction)

