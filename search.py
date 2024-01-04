import os
import numpy as np
from langchain.vectorstores import FAISS

def cosine_search(query, embedding_model, text):
    vectorstore = FAISS.from_documents(text, embedding_model())
    cosine_search = vectorstore.similarity_search_with_score(query, k=5)
    return cosine_search

def lexical_search(index, query, chunks, k):
    query_tokens = query.lower().split()  # preprocess query
    scores = index.get_scores(query_tokens)  # get best matching (BM) scores
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]  # sort and get top k
    lexical_context = [
        {"id": chunks[i][0], "text": chunks[i][1], "source": chunks[i][2], "score": scores[i]}
        for i in indices
    ]
    return lexical_context

