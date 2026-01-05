import chromadb
import uuid
from typing import List, Dict, Any
from langchain_ollama import OllamaEmbeddings

# Initialize simple persistent client
# We move the DB path to a cleaner location relative to the app execution or keep it global
DB_PATH = "./chroma_db"
COLLECTION_NAME = "incident_memory"

class MemoryManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        # We will use Ollama for embeddings to keep it local
        self.embedding_fn = OllamaEmbeddings(model="llama3.2:3b") 
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def save_incident(self, description: str, resolution: str):
        """Saves an incident and its resolution to memory."""
        doc_id = str(uuid.uuid4())
        text_to_embed = f"Issue: {description}\nResolution: {resolution}"
        
        vector = self.embedding_fn.embed_query(text_to_embed)
        
        self.collection.add(
            ids=[doc_id],
            documents=[text_to_embed],
            embeddings=[vector],
            metadatas=[{"type": "incident"}]
        )
        return doc_id

    def search_incidents(self, query: str, n_results: int = 2) -> List[str]:
        """Searches for similar past incidents."""
        vector = self.embedding_fn.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results
        )
        
        return results["documents"][0] if results["documents"] else []
