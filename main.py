import json
from pathlib import Path
from chunking import chunk_pdf


def main() -> None:
    result = chunk_pdf('./hope.pdf', w=35, k=10, smoothing_width=2)
    
    print(f"Source PDF: {result['source_pdf']}")
    print(f"Pages: {result['num_pages']}")
    print(f"Chunks: {result['num_chunks']}")
    print(f"Sections detected: {result['num_sections']}")
    print("\n" + "="*80)
    
    # Show first few chunks with metadata
    for i, chunk in enumerate(result['chunks'][:3], 1):
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Pages: {chunk['pages']}")
        print(f"  Author: {chunk['author']}")
        print(f"  Title: {chunk['title']}")
        print(f"  Section: {chunk['section_title']} (level {chunk['section_level']})")
        print(f"  Text preview: {chunk['text'][:100]}...")
    
    if len(result['chunks']) > 3:
        print(f"\n... and {len(result['chunks']) - 3} more chunks")
    
    # Optionally save to JSON
    # with open('chunks.json', 'w') as f:
    #     json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()