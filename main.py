import argparse
import time
from pathlib import Path
from chunking import chunk_pdf
from embedding import embed_chunks
from pinecone_store import PineconeStore, create_index_if_not_exists


def main() -> None:
    """Main pipeline: chunk PDF -> embed chunks -> index in Pinecone."""

    parser = argparse.ArgumentParser(description="RAG Citation Pipeline")
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip chunking, embedding, and storing steps (use when index is already populated)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Citation Pipeline")
    print("=" * 60)

    # Step 3: Verify index exists
    print("\nChecking Pinecone index...")
    if not create_index_if_not_exists("rag-citation-index", dimension=1024):
        print("\n✗ Index does not exist. Please run setup_index.py first.")
        return

    store = PineconeStore(index_name="rag-citation-index")

    if not args.skip_ingestion:
        # Step 1: Chunk the PDF
        print("\n[1/3] Chunking PDF...")
        chunks = chunk_pdf('./hope.pdf', w=35, k=10, smoothing_width=2, development=True)
        print(f"✓ Created {len(chunks)} chunks")

        # Step 2: Embed chunks
        print("\n[2/3] Embedding chunks...")
        embedded_chunks = embed_chunks(chunks)
        print(f"✓ Embedded {len(embedded_chunks)} chunks")

        # Step 3: Store in Pinecone
        print("\n[3/3] Storing chunks in Pinecone...")
        result = store.store_chunks(embedded_chunks, namespace="default")
        print(f"✓ {result['message']}")

        # Wait for indexing to complete (required before searching)
        print("\nWaiting 10 seconds for indexing to complete...")
        time.sleep(10)
    else:
        print("\n⚡ Skipping ingestion (--skip-ingestion flag set)")

    # Verify storage
    print("\nVerifying storage...")
    stats = store.get_stats()
    print(f"✓ Index stats:")
    print(f"  - Total vectors: {stats['total_vectors']}")
    print(f"  - Namespaces: {stats['namespaces']}")

    # Test search
    print("\nTesting search functionality...")
    test_query = "What is hope theory in entrepreneurship?"
    results = store.search(test_query, namespace="default", top_k=3)
    print(f"\n✓ Search results for: '{test_query}'")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (score: {result['score']:.4f}):")
        print(f"    Section: {result['section_title']}")
        print(f"    Pages: {', '.join(result['pages'])}")
        print(f"    Text preview: {result['text'][:150]}...")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()