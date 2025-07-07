def legal_entity_match(query, doc):
    return 1 if any(term in doc.page_content for term in query.split()) else 0

