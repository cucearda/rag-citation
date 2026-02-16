# Citation System Usage Guide

## Overview

The citation system breaks paragraphs into claims and retrieves supporting evidence from your indexed papers in Pinecone.

## How It Works

### 1. Claim Extraction Logic

- **Long sentences (≥15 words)**: Become standalone claims
- **Short sentences (<15 words)**: Combined using sliding window approach
  - Combined with previous sentence(s) until ≥15 words
  - Combined with next sentence(s) until ≥15 words
  - Creates overlapping claims for better context coverage

### 2. Evidence Retrieval

- Each claim is embedded using Voyage AI
- Queries Pinecone with `top_k=15` to retrieve supporting evidence
- Returns structured data with citation metadata (title, author, pages, sections)

## Quick Start

### Basic Usage

```python
from citation import extract_claims_with_citations
from pinecone_store import PineconeStore

# Your input text
paragraph = """
Hope theory has significant implications for entrepreneurship.
Entrepreneurs face uncertainty. They must maintain goals despite setbacks.
Research shows that hopeful individuals are more likely to persist.
"""

# Connect to Pinecone
store = PineconeStore(index_name="rag-citation-index")

# Extract claims with evidence
claims = extract_claims_with_citations(
    paragraph=paragraph,
    store=store,
    namespace="default",
    top_k=15
)

# Use the results
for claim in claims:
    print(f"Claim {claim.index}: {claim.text}")
    print(f"Evidence: {len(claim.evidence)} sources found")
    for evidence in claim.evidence[:3]:  # Top 3
        print(f"  - {evidence.title} (score: {evidence.score:.4f})")
```

## Run the Demo

```bash
python3 test_citation.py
```

This will demonstrate the full pipeline with a sample paragraph.

## Data Structures

### Claim Object

```python
@dataclass
class Claim:
    index: int                      # Sequential claim number
    text: str                       # The claim text
    sentence_indices: List[int]     # Which original sentences form this claim
    word_count: int                 # Total words in the claim
    evidence: List[ClaimEvidence]   # Supporting evidence from papers
```

### ClaimEvidence Object

```python
@dataclass
class ClaimEvidence:
    chunk_id: str        # Unique chunk identifier
    score: float         # Similarity score (0-1)
    text: str           # Evidence text from the paper
    author: str         # Paper author
    title: str          # Paper title
    section_title: str  # Section where evidence was found
    pages: List[int]    # Page numbers
```

## Advanced Usage

### Custom Top-K

```python
claims = extract_claims_with_citations(
    paragraph=paragraph,
    store=store,
    top_k=20  # Get more evidence per claim
)
```

### Filter by Metadata

```python
# In citation.py, modify the search call to add filters:
search_results = store.search(
    query_text=claim.text,
    namespace=namespace,
    top_k=top_k,
    filter_dict={"author": "Specific Author"}  # Filter by metadata
)
```

### Custom Namespace

```python
claims = extract_claims_with_citations(
    paragraph=paragraph,
    store=store,
    namespace="my-custom-namespace"
)
```

## Next Steps (Future Implementation)

As noted in the plan, the next phase will involve:

1. **LLM-based reranking**: Use an LLM to evaluate and rerank the evidence
2. **Confidence scoring**: Assess how well evidence supports each claim
3. **Citation suggestions**: Generate formatted citations
4. **Conflict detection**: Identify contradictory evidence

## File Structure

```
rag-citation/
├── models.py              # Data models (Claim, ClaimEvidence)
├── citation.py            # Core citation logic
├── test_citation.py       # Demo/example usage
├── pinecone_store.py      # Pinecone integration (existing)
└── CITATION_USAGE.md      # This file
```

## Requirements

- Python 3.8+
- nltk (automatically downloads punkt tokenizer on first use)
- Existing Pinecone index with embedded documents
- Voyage AI API key for embeddings

## Troubleshooting

### "Index does not exist"
Make sure you've run `setup_index.py` and indexed your documents with `main.py`.

### "NLTK punkt not found"
The system automatically downloads it, but you can manually run:
```python
import nltk
nltk.download('punkt')
```

### Empty evidence lists
- Verify your Pinecone index has data: check with `store.get_stats()`
- Ensure you're using the correct namespace
- Wait a few seconds after indexing for Pinecone to update
