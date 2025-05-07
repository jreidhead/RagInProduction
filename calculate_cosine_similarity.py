import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
 
 
def calculate_cosine_similarity(sentences, model, batch_size=8):
    sentence_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_embeddings = model.encode(sentences[i:i + batch_size])
        sentence_embeddings.extend(batch_embeddings)
 
    sentence_embeddings = np.array(sentence_embeddings)
    cosine_similarities = cosine_similarity(sentence_embeddings)
    return cosine_similarities
 