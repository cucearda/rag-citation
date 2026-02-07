#!/usr/bin/env python3
"""
PDF -> Text -> TextTiling chunks (topic-based segmentation)

Requirements:
  pip install pypdf pdfplumber nltk

  # one-time in Python:
  python -c "import nltk; nltk.download('punkt')"

Usage:
  python chunking.py input.pdf
  from chunking import chunk_pdf
  result = chunk_pdf('input.pdf')
  
Features:
  - Uses pdfplumber for heading detection (preserves layout)
  - Uses pypdf for clean text extraction (removes layout noise for TextTiling)
  - Adds metadata: author, title, page number, section title
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from pypdf import PdfReader
import pdfplumber
import nltk 
from nltk.tokenize.texttiling import TextTilingTokenizer

# Check for required NLTK data at import time
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError as e:
    raise SystemExit(
        "NLTK 'punkt' tokenizer data not found.\n"
        "Run: python -c \"import nltk; nltk.download('punkt')\""
    ) from e

# try:
#     _ = nltk.data.find("stopwords")
# except LookupError as e:
#     raise SystemExit(
#         "NLTK 'stopwords' data not found.\n"
#         "Run: python -c \"import nltk; nltk.download('stopwords')\""
#     ) from e


# --------- PDF extraction ---------
def extract_text_per_page(pdf_path: str) -> List[str]:
    """
    Extract text page-by-page using PyPDF.
    If your PDFs are scanned images, this will return little/no text (needs OCR).
    """
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append(txt)
    return pages


# --------- Cleaning / normalization ---------
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

def normalize_page_text(text: str) -> str:
    """
    Mild cleaning:
    - fix hyphenation across line breaks: "exam-\nple" -> "example"
    - normalize spaces
    - collapse excessive blank lines
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def build_document_with_offsets(pages: List[str]) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Build a single document string with page separators, and track char offsets.

    Returns:
      doc_text: concatenated pages
      page_spans: list of (page_index, start_char, end_char) in doc_text
    """
    doc_parts: List[str] = []
    page_spans: List[Tuple[int, int, int]] = []
    cursor = 0

    for idx, raw in enumerate(pages):
        txt = normalize_page_text(raw)
        # Add a clear separator so TextTiling can “feel” boundaries without being too disruptive.
        # Keep it simple and consistent.
        sep = "\n\n" if idx == 0 else "\n\n"  # same separator
        if idx != 0:
            doc_parts.append(sep)
            cursor += len(sep)

        start = cursor
        doc_parts.append(txt)
        cursor += len(txt)
        end = cursor
        page_spans.append((idx, start, end))

    return "".join(doc_parts), page_spans


def pages_for_span(page_spans: List[Tuple[int, int, int]], span: Tuple[int, int]) -> List[int]:
    """Given (start,end) char span in doc_text, return 1-based page numbers overlapped."""
    s, e = span
    overlapped = []
    for page_idx, ps, pe in page_spans:
        if e <= ps:
            break
        if s < pe and e > ps:
            overlapped.append(page_idx + 1)  # 1-based for humans
    return overlapped


# --------- TextTiling chunking ---------
def texttiling_segments(doc_text: str, w: int = 20, k: int = 10, smoothing_width: int = 2) -> List[Tuple[int, int]]:
    """
    Run NLTK TextTiling, then convert segments to character spans (start,end).

    Notes:
    - TextTiling returns token-based segments; we approximate spans by searching segment strings
      in the original doc_text in order.

    Parameters:
      w, k: TextTiling window sizes (typical defaults: w=20, k=10)
      smoothing_width: smoothing for boundary detection
    """
    tokenizer = TextTilingTokenizer(w=w, k=k, smoothing_width=smoothing_width)
    segments = tokenizer.tokenize(doc_text)

    # Convert segment strings -> char spans by incremental search.
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        pos = doc_text.find(seg, cursor)
        if pos == -1:
            # Fallback: be more forgiving by collapsing whitespace in both strings.
            # This helps if PDF extraction produces slight spacing differences.
            def squash_ws(s: str) -> str:
                return re.sub(r"\s+", " ", s).strip()

            squashed_doc = squash_ws(doc_text[cursor:])
            squashed_seg = squash_ws(seg)
            pos2 = squashed_doc.find(squashed_seg)
            if pos2 == -1:
                # Last resort: approximate by taking remaining text as a segment.
                start = cursor
                end = len(doc_text)
                spans.append((start, end))
                cursor = end
                break
            else:
                # Map back approximately by taking a proportional slice.
                # Not perfect, but usually rare and acceptable.
                # We’ll just append the remaining text as the final chunk.
                start = cursor
                end = len(doc_text)
                spans.append((start, end))
                cursor = end
                break

        start = pos
        end = pos + len(seg)
        spans.append((start, end))
        cursor = end

    # Merge tiny trailing chunk if needed (optional, conservative)
    merged: List[Tuple[int, int]] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        prev_s, prev_e = merged[-1]
        s, e = span
        if (e - s) < 200:  # tiny segment threshold
            merged[-1] = (prev_s, e)
        else:
            merged.append(span)

    return merged


# --------- Heading Detection with pdfplumber ---------
def extract_text_with_pdfplumber(pdf_path: str) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Extract text using pdfplumber, preserving layout information.
    Returns normalized text and page spans similar to build_document_with_offsets.
    """
    pages_text: List[str] = []
    page_spans: List[Tuple[int, int, int]] = []
    cursor = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = normalize_page_text(txt)
            
            sep = "\n\n" if idx == 0 else "\n\n"
            if idx != 0:
                pages_text.append(sep)
                cursor += len(sep)
            
            start = cursor
            pages_text.append(txt)
            cursor += len(txt)
            end = cursor
            page_spans.append((idx, start, end))
    
    return "".join(pages_text), page_spans


def detect_headings_with_pdfplumber(pdf_path: str, pypdf_text: str, pypdf_page_spans: List[Tuple[int, int, int]]) -> List[Section]:
    """
    Detect headings using pdfplumber's layout information.
    
    Uses heuristics:
    - Font size larger than body text (mode font size)
    - Bold or specific font families
    - Position patterns (left-aligned)
    - Text patterns (short lines, title case, numbering)
    
    Returns list of Section objects with character offsets aligned to pypdf text.
    """
    sections: List[Section] = []
    
    # Extract text with pdfplumber to get character-level details
    pdfplumber_text, pdfplumber_page_spans = extract_text_with_pdfplumber(pdf_path)
    
    # Collect font information from all pages
    font_sizes: List[float] = []
    font_names: List[str] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            chars = page.chars
            for char in chars:
                if char.get('size'):
                    font_sizes.append(char['size'])
                if char.get('fontname'):
                    font_names.append(char['fontname'])
    
    if not font_sizes:
        return sections  # No font info available
    
    # Calculate mode font size (most common size, likely body text)
    size_counter = Counter(round(s, 1) for s in font_sizes)
    mode_size = size_counter.most_common(1)[0][0] if size_counter else 0
    
    # Extract lines with their properties
    candidate_headings: List[Dict[str, Any]] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            words = page.extract_words()
            if not words:
                continue
            
            # Group words into lines based on y-coordinate
            lines: Dict[float, List[Dict]] = {}
            for word in words:
                y = round(word['top'], 1)
                if y not in lines:
                    lines[y] = []
                lines[y].append(word)
            
            # Process each line
            for y_pos, line_words in sorted(lines.items(), reverse=True):
                if not line_words:
                    continue
                
                # Combine words into line text
                line_text = ' '.join(w['text'] for w in sorted(line_words, key=lambda w: w['x0']))
                line_text = line_text.strip()
                
                if not line_text or len(line_text) < 2:
                    continue
                
                # Get font properties from first word
                first_word = line_words[0]
                font_size = first_word.get('size', 0)
                font_name = first_word.get('fontname', '')
                x0 = first_word.get('x0', 0)
                
                # Calculate heading score
                score = 0
                level = 1
                
                # Font size check (larger than mode = heading)
                if font_size > mode_size * 1.15:
                    score += 2
                    if font_size > mode_size * 1.5:
                        level = 1
                    elif font_size > mode_size * 1.3:
                        level = 2
                    else:
                        level = 3
                
                # Font weight/name check
                font_lower = font_name.lower()
                if 'bold' in font_lower or 'heavy' in font_lower or 'black' in font_lower:
                    score += 2
                
                # Length check (headings are typically shorter)
                if len(line_text) < 100:
                    score += 1
                if len(line_text) < 50:
                    score += 1
                
                # Position check (left-aligned, typically near top of page)
                page_height = page.height
                relative_y = y_pos / page_height if page_height > 0 else 0.5
                if relative_y < 0.3:  # Top 30% of page
                    score += 1
                
                # Text pattern checks
                # Numbering pattern (1., 1.1, etc.)
                if re.match(r'^\d+\.?\s+', line_text) or re.match(r'^\d+\.\d+\s+', line_text):
                    score += 2
                    # Determine level from numbering
                    if re.match(r'^\d+\.\d+\.\d+', line_text):
                        level = 3
                    elif re.match(r'^\d+\.\d+', line_text):
                        level = 2
                    else:
                        level = 1
                
                # Title case or ALL CAPS
                if line_text.isupper() and len(line_text) > 3:
                    score += 1
                elif line_text.istitle() or (line_text[0].isupper() and not line_text.endswith('.')):
                    score += 1
                
                # Ends without punctuation (common for headings)
                if not line_text.rstrip().endswith(('.', ',', ';', ':')):
                    score += 1
                
                # If score >= 3, consider it a heading
                if score >= 3:
                    candidate_headings.append({
                        'text': line_text,
                        'page_num': page_idx + 1,
                        'font_size': font_size,
                        'font_name': font_name,
                        'x0': x0,
                        'y': y_pos,
                        'level': level,
                        'score': score
                    })
    
    # Map headings to pypdf text positions using fuzzy alignment
    for heading in candidate_headings:
        heading_text = heading['text']
        page_num = heading['page_num']
        
        # Find approximate position in pypdf text
        # First, try to find in the corresponding page range
        page_start = 0
        page_end = len(pypdf_text)
        
        for pidx, ps, pe in pypdf_page_spans:
            if pidx + 1 == page_num:
                page_start = ps
                page_end = pe
                break
        
        # Search in the page range
        search_text = pypdf_text[page_start:page_end]
        pos = search_text.find(heading_text)
        
        if pos == -1:
            # Try normalized search (collapse whitespace)
            normalized_heading = re.sub(r'\s+', ' ', heading_text).strip()
            normalized_search = re.sub(r'\s+', ' ', search_text).strip()
            normalized_pos = normalized_search.find(normalized_heading)
            
            if normalized_pos != -1:
                # Map back to original text position
                # Count characters in normalized text up to match
                norm_chars_before = len(normalized_search[:normalized_pos])
                # Find corresponding position in original text
                orig_pos = 0
                norm_count = 0
                for i, char in enumerate(search_text):
                    if not char.isspace():
                        norm_count += 1
                    if norm_count >= norm_chars_before:
                        orig_pos = i
                        break
                pos = page_start + orig_pos
            else:
                # Try partial match (first 10 chars)
                heading_start = heading_text[:20].strip()
                if heading_start:
                    partial_pos = search_text.find(heading_start)
                    if partial_pos != -1:
                        pos = page_start + partial_pos
                    else:
                        # Fallback: use page start
                        pos = page_start
                else:
                    pos = page_start
        else:
            pos = page_start + pos
        
        # Estimate end position (start of next heading or end of page)
        char_offset_end = page_end
        
        sections.append(Section(
            title=heading_text,
            page_num=page_num,
            char_offset_start=pos,
            char_offset_end=char_offset_end,
            level=heading['level']
        ))
    
    # Update end positions based on next section
    for i in range(len(sections) - 1):
        sections[i].char_offset_end = sections[i + 1].char_offset_start
    
    return sections


# --------- PDF Metadata Extraction ---------
def extract_pdf_metadata(pdf_path: str) -> Dict[str, str]:
    """Extract metadata (author, title, etc.) from PDF using pypdf."""
    reader = PdfReader(pdf_path)
    metadata = reader.metadata or {}
    
    # pypdf metadata keys have '/' prefix
    return {
        'author': metadata.get('/Author', '') or '',
        'title': metadata.get('/Title', '') or '',
        'subject': metadata.get('/Subject', '') or '',
        'creator': metadata.get('/Creator', '') or '',
    }


# --------- Section Assignment ---------
def find_section_for_chunk(char_start: int, char_end: int, sections: List[Section]) -> Tuple[str, int]:
    """
    Find the section (heading) that a chunk belongs to.
    Returns (section_title, section_level).
    """
    if not sections:
        return ('', 0)
    
    # Find the nearest preceding heading
    # If chunk spans multiple sections, use the section where majority of text falls
    chunk_mid = (char_start + char_end) // 2
    
    best_section: Optional[Section] = None
    
    for section in sections:
        if section.char_offset_start <= chunk_mid:
            if best_section is None or section.char_offset_start > best_section.char_offset_start:
                best_section = section
    
    if best_section:
        return (best_section.title, best_section.level)
    
    # If no preceding section found, check if chunk starts before first section
    # In that case, return empty or first section
    if sections and char_start < sections[0].char_offset_start:
        return ('', 0)
    
    # Fallback: return first section
    if sections:
        return (sections[0].title, sections[0].level)
    
    return ('', 0)


# --------- Output schema ---------
@dataclass
class Section:
    """Represents a detected heading/section in the PDF."""
    title: str
    page_num: int
    char_offset_start: int
    char_offset_end: int
    level: int  # heading hierarchy level (1 for top-level, 2 for subsections, etc.)


@dataclass
class Chunk:
    chunk_id: int
    start_char: int
    end_char: int
    pages: List[int]
    text: str
    author: str
    title: str
    section_title: str
    section_level: int


def chunk_pdf(
    pdf_path: str,
    w: int = 20,
    k: int = 10,
    smoothing_width: int = 2,
) -> Dict[str, Any]:
    # Extract PDF metadata (author, title)
    metadata = extract_pdf_metadata(pdf_path)
    author = metadata.get('author', '')
    title = metadata.get('title', '')
    
    # Extract text using pypdf (clean, no layout noise for TextTiling)
    pages = extract_text_per_page(pdf_path)
    doc_text, page_spans = build_document_with_offsets(pages)
    
    # Detect headings using pdfplumber (preserves layout)
    sections = detect_headings_with_pdfplumber(pdf_path, doc_text, page_spans)
    
    # Perform TextTiling segmentation
    spans = texttiling_segments(doc_text, w=w, k=k, smoothing_width=smoothing_width)

    # Create chunks with metadata
    chunks: List[Chunk] = []
    for i, (s, e) in enumerate(spans, start=1):
        chunk_text = doc_text[s:e].strip()
        if not chunk_text:
            continue
        chunk_pages = pages_for_span(page_spans, (s, e))
        
        # Find section for this chunk
        section_title, section_level = find_section_for_chunk(s, e, sections)
        
        chunks.append(
            Chunk(
                chunk_id=i,
                start_char=s,
                end_char=e,
                pages=chunk_pages,
                text=chunk_text,
                author=author,
                title=title,
                section_title=section_title,
                section_level=section_level,
            )
        )

    return {
        "source_pdf": str(Path(pdf_path).resolve()),
        "params": {"w": w, "k": k, "smoothing_width": smoothing_width},
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "num_sections": len(sections),
        "chunks": [asdict(c) for c in chunks],
    }
