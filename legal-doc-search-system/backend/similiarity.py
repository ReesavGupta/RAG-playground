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

def precision_at_k(results, relevant_docs, k=5):
    retrieved = results[:k]
    relevant_retrieved = [doc for doc in retrieved if doc in relevant_docs]
    return len(relevant_retrieved) / k if k else 0

def recall_at_k(results, relevant_docs, k=5):
    retrieved = results[:k]
    relevant_retrieved = [doc for doc in retrieved if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

def diversity_score(results):
    # Simple diversity: count of unique first 10 words in each result
    unique_phrases = set()
    for doc in results:
        words = tuple(doc.page_content.split()[:10])
        unique_phrases.add(words)
    return len(unique_phrases) / len(results) if results else 0