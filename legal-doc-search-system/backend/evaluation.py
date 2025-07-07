def precision_at_k(results, relevant_docs, k=5):
    return len(set(results[:k]) & set(relevant_docs)) / k

def recall(results, relevant_docs):
    return len(set(results) & set(relevant_docs)) / len(relevant_docs)

def diversity_score(results):
    topics = [doc.metadata.get("topic", "") for doc in results]
    return len(set(topics)) / len(topics)