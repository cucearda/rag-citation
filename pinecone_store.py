"""
Pinecone integration for storing and searching embedded chunks.

This module provides functions to:
- Store embedded chunks in Pinecone
- Search for similar chunks
- Manage Pinecone indexes
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from models import EmbeddedChunk, Chunk


def get_pinecone_api_key() -> str:
    """Load Pinecone API key from api_key.json or environment."""
    api_key_path = os.path.join(os.path.dirname(__file__), 'api_key.json')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            keys = json.load(f)
            pinecone_key = keys.get('pinecone')
            if pinecone_key:
                return pinecone_key
    
    # Fall back to environment variable
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in api_key.json or environment")
    return api_key


class PineconeStore:
    """Manages Pinecone operations for storing and searching embedded chunks."""
    
    def __init__(self, index_name: str = "rag-citation-index"):
        """
        Initialize Pinecone client and connect to index.
        
        Args:
            index_name: Name of the Pinecone index to use
        """
        self.api_key = get_pinecone_api_key()
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.index = None
        
    def get_index(self):
        """Get or create the index connection."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        return self.index
    
    def store_chunks(
        self, 
        embedded_chunks: List[EmbeddedChunk], 
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Store embedded chunks in Pinecone using vector-based indexing.
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects to store
            namespace: Pinecone namespace to store chunks in
            
        Returns:
            Dictionary with operation results
        """
        if not embedded_chunks:
            return {"stored": 0, "message": "No chunks to store"}
        
        index = self.get_index()
        
        # Prepare vectors for Pinecone (vector-based indexing)
        vectors = []
        for emb_chunk in embedded_chunks:
            chunk = emb_chunk.chunk
            vector = {
                "id": emb_chunk.vector_id,
                "values": emb_chunk.embedding,  # 1024-dimensional vector from Voyage AI
                "metadata": {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "author": chunk.author,
                    "title": chunk.title,
                    "section_title": chunk.section_title,
                    "section_level": chunk.section_level,
                    "pages": ",".join(map(str, chunk.pages)),  # Pinecone metadata must be flat
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            }
            vectors.append(vector)
        
        # Upsert vectors in batches (max 1000 for vectors)
        batch_size = 1000
        total_stored = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=namespace)
                total_stored += len(batch)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                raise
        
        return {
            "stored": total_stored,
            "namespace": namespace,
            "message": f"Successfully stored {total_stored} chunks"
        }
    
    def search(
        self,
        query_text: str,
        namespace: str = "default",
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector-based semantic search.
        
        This method automatically embeds the query using Voyage AI and performs
        vector similarity search in Pinecone.
        
        Args:
            query_text: Text query to search for
            namespace: Namespace to search in
            top_k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"author": "John Doe"})
            
        Returns:
            List of search results with chunks and metadata
        """
        from embedding import get_voyage_client
        
        # Embed the query using Voyage AI
        vo = get_voyage_client()
        query_embedding = vo.embed(
            texts=[query_text], 
            model="voyage-4-lite", 
            input_type="query"  # Use "query" for search queries
        ).embeddings[0]
        
        # Query Pinecone using vector search
        index = self.get_index()
        
        try:
            results = index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
        except Exception as e:
            print(f"Search error: {e}")
            raise
        
        # Format results
        formatted_results = []
        for match in results.matches:
            result = {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "chunk_id": match.metadata.get("chunk_id"),
                "author": match.metadata.get("author", ""),
                "title": match.metadata.get("title", ""),
                "section_title": match.metadata.get("section_title", ""),
                "pages": match.metadata.get("pages", "").split(",") if match.metadata.get("pages") else [],
                "start_char": match.metadata.get("start_char"),
                "end_char": match.metadata.get("end_char"),
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        index = self.get_index()
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "namespaces": dict(stats.namespaces) if hasattr(stats, 'namespaces') else {},
            "dimension": stats.dimension if hasattr(stats, 'dimension') else None,
        }


def create_index_if_not_exists(
    index_name: str = "rag-citation-index",
    dimension: int = 1024,  # Voyage-4-lite dimension
    metric: str = "cosine"
) -> bool:
    """
    Check if a Pinecone index exists.
    
    Note: For vector-based indexes (using pre-computed embeddings from Voyage AI),
    use the setup_index.py script to create the index programmatically.
    
    Args:
        index_name: Name of the index
        dimension: Vector dimension (1024 for voyage-4-lite)
        metric: Similarity metric (cosine, euclidean, dotproduct)
        
    Returns:
        True if index exists, False otherwise
    """
    api_key = get_pinecone_api_key()
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"✓ Index '{index_name}' already exists")
        return True
    
    print(f"✗ Index '{index_name}' does not exist.")
    print("\nPlease create it using one of these methods:")
    print("\n1. Run the setup script:")
    print("   python3 setup_index.py")
    print("\n2. Or use the Pinecone CLI:")
    print(f"   pc index create -n {index_name} -m {metric} -d {dimension} -c aws -r us-east-1")
    return False
