from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_score(emb1, emb2):
    # Ensure both embeddings are 2D
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]
