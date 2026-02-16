"""
Citation module for extracting claims from text and retrieving supporting evidence.

This module provides functionality to:
- Break paragraphs into claims using a sliding window approach
- Query Pinecone for supporting evidence for each claim
- Return structured claim data with citations
"""

import re
from typing import List
from models import Claim, ClaimEvidence
from pinecone_store import PineconeStore

# Download nltk punkt tokenizer on first import
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK setup issue: {e}")


def parse_into_sentences(text: str) -> List[str]:
    """
    Parse text into sentences using NLTK's sentence tokenizer.
    
    Args:
        text: Input text to parse
        
    Returns:
        List of sentences
    """
    # Clean up text: remove extra whitespace, normalize line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Use NLTK's sentence tokenizer for robust sentence boundary detection
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback to simple regex-based splitting if NLTK fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def count_words(sentence: str) -> int:
    """
    Count words in a sentence using simple whitespace splitting.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Number of words
    """
    return len(sentence.split())


def build_claims(sentences: List[str]) -> List[Claim]:
    """
    Build claims from sentences by concatenating until >= 15 words.
    
    Logic:
    - Start with current sentence as claim_sentences
    - While claim_sentences has < 15 words: append next sentence
    - Create claim when claim_sentences >= 15 words, then advance past consumed sentences
    
    Args:
        sentences: List of sentences to process
        
    Returns:
        List of Claim objects (without evidence populated)
    """
    if not sentences:
        return []
    
    claims = []
    claim_index = 0
    i = 0
    
    while i < len(sentences):
        claim_sentences = sentences[i]
        sentence_indices = [i]
        j = i + 1
        
        # Keep adding next sentences until we have >= 15 words
        while count_words(claim_sentences) < 15 and j < len(sentences):
            claim_sentences = claim_sentences + " " + sentences[j]
            sentence_indices.append(j)
            j += 1
        
        claim = Claim(
            index=claim_index,
            text=claim_sentences,
            sentence_indices=sentence_indices,
            word_count=count_words(claim_sentences),
            evidence=[]
        )
        claims.append(claim)
        claim_index += 1
        i = j  # Move past all consumed sentences
    
    return claims


def retrieve_evidence_for_claims(
    claims: List[Claim],
    store: PineconeStore,
    namespace: str = "default",
    top_k: int = 15
) -> List[Claim]:
    """
    Query Pinecone for supporting evidence for each claim.
    
    Args:
        claims: List of Claim objects (evidence will be populated)
        store: PineconeStore instance
        namespace: Pinecone namespace to search
        top_k: Number of evidence chunks to retrieve per claim
        
    Returns:
        List of Claim objects with evidence populated
    """
    for claim in claims:
        # Query Pinecone with the claim text
        search_results = store.search(
            query_text=claim.text,
            namespace=namespace,
            top_k=top_k
        )
        
        # Convert search results to ClaimEvidence objects
        evidence_list = []
        for result in search_results:
            evidence = ClaimEvidence(
                chunk_id=result['chunk_id'],
                score=result['score'],
                text=result['text'],
                author=result['author'],
                title=result['title'],
                section_title=result['section_title'],
                pages=result['pages']
            )
            evidence_list.append(evidence)
        
        claim.evidence = evidence_list
    
    return claims


def extract_claims_with_citations(
    paragraph: str,
    store: PineconeStore,
    namespace: str = "default",
    top_k: int = 15
) -> List[Claim]:
    """
    Main entry point: Extract claims from a paragraph and retrieve supporting evidence.
    
    This function orchestrates the full pipeline:
    1. Parse paragraph into sentences
    2. Build claims using sliding window approach
    3. Query Pinecone for supporting evidence for each claim
    
    Args:
        paragraph: Input text to extract claims from
        store: PineconeStore instance for querying
        namespace: Pinecone namespace to search
        top_k: Number of evidence chunks to retrieve per claim
        
    Returns:
        List of Claim objects with evidence populated
        
    Example:
        >>> store = PineconeStore(index_name="rag-citation-index")
        >>> paragraph = "Hope theory has significant implications..."
        >>> claims = extract_claims_with_citations(paragraph, store, top_k=15)
        >>> for claim in claims:
        ...     print(f"Claim {claim.index}: {claim.text}")
        ...     print(f"Evidence sources: {len(claim.evidence)}")
    """
    # Step 1: Parse into sentences
    sentences = parse_into_sentences(paragraph)
    
    # Step 2: Build claims
    claims = build_claims(sentences)
    
    # Step 3: Retrieve evidence
    claims_with_evidence = retrieve_evidence_for_claims(
        claims=claims,
        store=store,
        namespace=namespace,
        top_k=top_k
    )
    
    return claims_with_evidence
