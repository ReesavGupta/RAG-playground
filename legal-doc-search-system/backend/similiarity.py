from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

def cosine_search(vectordb, query, k=5):
    return vectordb.similarity_search(query, k=k)

def euclidean_search(query_embedding, doc_embeddings, k=5):
    distances = euclidean_distances([query_embedding], doc_embeddings)[0]
    top_k = np.argsort(distances)[:k]
    return top_k

def mmr_search(vectordb, query, k=5, fetch_k=15):
    return vectordb.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

def hybrid_score(cos_score, legal_entity_score, alpha=0.6):
    return alpha * cos_score + (1 - alpha) * legal_entity_score