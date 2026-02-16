"""
Data models for PDF chunking and processing.

This module defines the core data structures used throughout the RAG citation system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Section:
    """Represents a detected heading/section in the PDF."""
    title: str
    page_num: int  # 1-based, if known; 0 if unknown
    char_offset_start: int
    char_offset_end: int
    level: int  # 1 top-level, 2 subsection, 3 sub-subsection


@dataclass
class Chunk:
    """Represents a text chunk extracted from a PDF."""
    chunk_id: int
    start_char: int
    end_char: int
    pages: List[int]
    text: str
    author: str
    title: str
    section_title: str
    section_level: int

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector_id: str
    embedding: List[float]


@dataclass
class ClaimEvidence:
    """Evidence supporting a claim from a paper."""
    chunk_id: str
    score: float
    text: str
    author: str
    title: str
    section_title: str
    pages: List[int]


@dataclass
class Claim:
    """Represents a claim extracted from input text."""
    index: int
    text: str
    sentence_indices: List[int]  # Which sentences form this claim
    word_count: int
    evidence: List[ClaimEvidence]
