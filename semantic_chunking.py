import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
 
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
 
def break_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences