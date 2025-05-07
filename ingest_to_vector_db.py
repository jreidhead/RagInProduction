import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from get_metadata import generate_keywords

load_dotenv()

# configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URI = os.getenv("QDRANT_URI", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "my_test_collection"
 
qdrant_client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API_KEY)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
 
def ingest_chunks_with_metadata_to_qdrant(qdrant_client, all_chunks):
    try:
        collections = qdrant_client.get_collections()
        if COLLECTION_NAME not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 768, "distance": "Cosine"}
            )
    except Exception as e:
        print("Error in collection handling:", e)
        return
 
    points = []
    for i, item in enumerate(all_chunks):
        chunk = item["chunk"]
        keywords_list = generate_keywords(chunk)
        embedding = model.encode(chunk)
 
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk": chunk,
                "keywords_list": keywords_list
            }
        )
        points.append(point)
  
def ingest_chunks_without_metadata_to_qdrant(all_chunks):
    try:
        collections = qdrant_client.get_collections()
        if COLLECTION_NAME not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 768, "distance": "Cosine"}
            )
    except Exception as e:
        print("Error in collection handling:", e)
        return
 
    points = []
    for i, item in enumerate(all_chunks):
        chunk = item["chunk"]
        embedding = model.encode(chunk)
 
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk": chunk,
            }
        )
        points.append(point)
 