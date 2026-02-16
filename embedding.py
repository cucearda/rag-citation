import voyageai
import json
import os
from typing import List
from models import Chunk, EmbeddedChunk


def get_voyage_client():
    """Initialize Voyage AI client with API key from api_key.json or environment."""
    # Try to load from api_key.json first
    api_key_path = os.path.join(os.path.dirname(__file__), 'api_key.json')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            keys = json.load(f)
            voyage_key = keys.get('voyage')
            if voyage_key:
                return voyageai.Client(api_key=voyage_key)
    
    # Fall back to environment variable
    return voyageai.Client()


vo = get_voyage_client()


def embed_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    """
    Embed chunks using Voyage AI and return EmbeddedChunk objects.
    
    Args:
        chunks: List of Chunk objects to embed
        
    Returns:
        List of EmbeddedChunk objects with embeddings
    """
    if not chunks:
        return []
    
    embedded_chunks: List[EmbeddedChunk] = []
    inputs: List[str] = []
    for chunk in chunks:
        inputs.append(chunk.text)

    # Get embeddings from Voyage AI
    embds_obj = vo.embed(
        texts=inputs, model="voyage-4-lite", input_type="document"
    )
    
    # Extract embeddings from the response
    embeddings = embds_obj.embeddings
    
    # Create EmbeddedChunk objects
    for i, chunk in enumerate(chunks):
        # Generate a unique vector ID for Pinecone
        vector_id = f"chunk_{chunk.chunk_id}"
        
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            vector_id=vector_id,
            embedding=embeddings[i]
        )
        embedded_chunks.append(embedded_chunk)
    
    return embedded_chunks