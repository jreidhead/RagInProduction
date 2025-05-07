 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import sent_tokenize, TextTilingTokenizer
 
def find_threshold(cosine_similarities, percentile=25):
    similarities = cosine_similarities.flatten()
    threshold = np.percentile(similarities, percentile)
    return threshold
 
 
def anchor_based_segmentation(sentences, cosine_similarities, adaptive_percentile=25):
    threshold = find_threshold(cosine_similarities, adaptive_percentile)
    chunks = []
    current_chunk = [sentences[0]]
    anchor_idx = 0
 
    for i in range(1, len(sentences)):
        similarity = cosine_similarities[anchor_idx, i]
        if similarity < threshold:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            anchor_idx = i
        else:
            current_chunk.append(sentences[i])
 
    if current_chunk:
        chunks.append(' '.join(current_chunk))
 
    return chunks
 
def fixed_size_chunking(text, chunk_size=128):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
 
def fixed_size_chunking_with_overlap(text, chunk_size=128, overlap=30):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks