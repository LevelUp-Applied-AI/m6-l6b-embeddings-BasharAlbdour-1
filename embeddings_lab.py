"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import torch
from transformers import AutoTokenizer, AutoModel



   
def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix=vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    similarity_matrix=sklearn_cosine(tfidf_matrix)
    return similarity_matrix


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings={}
    with open(filepath,'r',encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split()
            words=parts[0]
            vector=np.array(parts[1:],dtype=np.float32)
            embeddings[words]=vector
        return embeddings
    


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words=text.lower().split()
    vectors=[embeddings[word] for word in words if word in embeddings]
    if not vectors:
        return np.zeros(50)
    return np.mean(vectors, axis=0)
    


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    inputs=tokenizer(text, return_tensors='pt', truncation=True,max_length=512)
    with torch.no_grad():
        outputs=model(**inputs)
    last_hidden_state=outputs.last_hidden_state
    mask=inputs['attention_mask'].unsqueeze(-1)
    sum_embeddings = (last_hidden_state * mask).sum(dim=1) 
    sum_mask = mask.sum(dim=1)  
    embedding = (sum_embeddings / sum_mask).squeeze().numpy()  
    return embedding
    


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """

    corpus_glove = np.array([text_to_glove(t, glove_embeddings) for t in texts])
    corpus_bert = np.array([extract_bert_embedding(t, bert_tokenizer, bert_model) for t in texts])

    results = {}

    for query in queries:
        query_idx = texts.index(query)
        entry = {}

        tfidf_scores = tfidf_sim[query_idx].copy()
        tfidf_scores[query_idx] = -1
        top_tfidf = np.argsort(tfidf_scores)[::-1][:3]
        entry["tfidf"] = [(texts[i], tfidf_scores[i]) for i in top_tfidf]

        q_glove = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(q_glove, corpus_glove).flatten()
        glove_scores[query_idx] = -1
        top_glove = np.argsort(glove_scores)[::-1][:3]
        entry["glove"] = [(texts[i], glove_scores[i]) for i in top_glove]

        q_bert = extract_bert_embedding(query, bert_tokenizer, bert_model).reshape(1, -1)
        bert_scores = sklearn_cosine(q_bert, corpus_bert).flatten()
        bert_scores[query_idx] = -1
        top_bert = np.argsort(bert_scores)[::-1][:3]
        entry["bert"] = [(texts[i], bert_scores[i]) for i in top_bert]

        results[query] = entry

    return results



if __name__ == "__main__":

    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")
        glove_embeddings = np.array([text_to_glove(t, glove) for t in texts])
        glove_sim = sklearn_cosine(glove_embeddings)
        print(f"GloVe similarity matrix shape: {glove_sim.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")
    bert_embeddings = np.array([extract_bert_embedding(t, tokenizer, model) for t in texts])
    bert_sim = sklearn_cosine(bert_embeddings)
    print(f"BERT similarity matrix shape: {bert_sim.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q, methods in comparison.items():
                print(f"\n{'='*80}")
                print(f"Query: {q[:100]}...")
                print(f"{'='*80}")
                print(f"{'Rank':<6} {'TF-IDF':<40} {'GloVe':<40} {'BERT':<40}")
                print(f"{'-'*126}")
                for i in range(3):
                    tfidf_text = methods["tfidf"][i][0][:37] + "..." if len(methods["tfidf"][i][0]) > 37 else methods["tfidf"][i][0]
                    glove_text = methods["glove"][i][0][:37] + "..." if len(methods["glove"][i][0]) > 37 else methods["glove"][i][0]
                    bert_text  = methods["bert"][i][0][:37] + "..."  if len(methods["bert"][i][0]) > 37  else methods["bert"][i][0]
                    print(f"{i+1:<6} {tfidf_text:<40} {glove_text:<40} {bert_text:<40}")

    
    corpus_words = set()
    oov_words = set()
    total_tokens = 0
    oov_tokens = 0

    for text in texts:
        words = text.lower().split()
        total_tokens += len(words)
        for w in words:
            corpus_words.add(w)
            if w not in glove:
                oov_words.add(w)
                oov_tokens += 1

    print(f"Total tokens: {total_tokens}")
    print(f"OOV tokens: {oov_tokens}")
    print(f"OOV token rate: {oov_tokens/total_tokens:.2%}")
    print(f"Unique corpus words: {len(corpus_words)}")
    print(f"Unique OOV words: {len(oov_words)}")
    print(f"OOV type rate: {len(oov_words)/len(corpus_words):.2%}")
    print(f"\nSample OOV words:")
    print(list(oov_words)[:30])