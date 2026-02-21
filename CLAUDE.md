# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# One-time NLTK setup
python -c "import nltk; nltk.download('punkt')"

# Initialize Pinecone index
python setup_index.py

# Run full ingestion pipeline (chunk PDF → embed → store in Pinecone)
python main.py

# Run citation extraction demo
python test_citation.py
```

## API Keys

Keys are loaded from `api_key.json` (gitignored) or environment variables:

```json
{
  "pinecone": "...",
  "voyage": "..."
}
```

`ANTHROPIC_API_KEY` is loaded from `.env` via `python-dotenv`.

## Architecture

This is a two-phase RAG pipeline:

### Phase 1: Ingestion (`main.py`)
1. **`chunking.py`** — Extracts text from a PDF page-by-page (PyPDF), sends the PDF to Claude to detect section headings, then applies NLTK's TextTiling algorithm (topic-based segmentation) to produce `Chunk` objects with author/title/section/page metadata.
   - Set `development=True` in `chunk_pdf()` to skip Claude heading detection and use hardcoded headings (faster for local iteration).
2. **`embedding.py`** — Embeds each `Chunk` using Voyage AI (`voyage-4-lite`, 1024 dims, input type `"document"`), producing `EmbeddedChunk` objects.
3. **`pinecone_store.py`** — Batch-upserts `EmbeddedChunk` vectors into a Pinecone serverless index (`rag-citation-index`, cosine metric, AWS us-east-1).

### Phase 2: Citation Extraction (`citation.py`)
Given a paragraph of text:
1. Tokenize into sentences (NLTK `sent_tokenize`).
2. Build **claims** using a sliding window: combine adjacent sentences until the claim is ≥15 words.
3. For each claim, embed it with Voyage AI (input type `"query"`) and query Pinecone for top-k similar chunks.
4. Return a list of `Claim` objects, each with a list of `ClaimEvidence` items (text, score, author, title, section, pages).

### Data Models (`models.py`)
- `Chunk` — raw text chunk with metadata
- `EmbeddedChunk` — chunk + 1024-dim vector + Pinecone vector ID
- `Claim` — extracted claim text + sentence indices + evidence list
- `ClaimEvidence` — a single evidence item from Pinecone (text, score, metadata)

### Key Parameters
| Parameter | Value | Location |
|---|---|---|
| TextTiling window (w) | 35 | `main.py` → `chunk_pdf()` |
| TextTiling block (k) | 10 | `chunking.py` |
| Min chunk length | 200 chars | `chunking.py` |
| Min claim length | 15 words | `citation.py` |
| Default top_k | 15 | `citation.py` |
| Embedding dims | 1024 | `embedding.py`, `setup_index.py` |
