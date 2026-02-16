#!/usr/bin/env python3
"""
Setup script to create the Pinecone index for RAG citation system.

This script creates a serverless Pinecone index if it doesn't already exist.
Run this before running main.py for the first time.
"""

import sys
from pinecone import Pinecone, ServerlessSpec
from pinecone_store import get_pinecone_api_key


def create_pinecone_index():
    """Create the Pinecone index for vector storage."""
    
    # Configuration
    index_name = "rag-citation-index"
    dimension = 1024  # voyage-4-lite default dimension
    metric = "cosine"
    cloud = "aws"
    region = "us-east-1"
    
    print(f"Initializing Pinecone client...")
    api_key = get_pinecone_api_key()
    pc = Pinecone(api_key=api_key)
    
    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"✓ Index '{index_name}' already exists")
        
        # Get index details
        index_info = pc.describe_index(index_name)
        print(f"  Dimension: {index_info.dimension}")
        print(f"  Metric: {index_info.metric}")
        print(f"  Cloud: {index_info.spec.serverless.cloud}")
        print(f"  Region: {index_info.spec.serverless.region}")
        return True
    
    # Create the index
    print(f"Creating index '{index_name}'...")
    print(f"  Dimension: {dimension}")
    print(f"  Metric: {metric}")
    print(f"  Cloud: {cloud}")
    print(f"  Region: {region}")
    
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        
        print(f"✓ Successfully created index '{index_name}'")
        print(f"\nIndex is ready to use!")
        return True
        
    except Exception as e:
        print(f"✗ Error creating index: {e}")
        return False


if __name__ == "__main__":
    success = create_pinecone_index()
    sys.exit(0 if success else 1)
