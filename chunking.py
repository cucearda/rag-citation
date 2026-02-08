#!/usr/bin/env python3
"""
PDF -> Text -> TextTiling chunks (topic-based segmentation)

Requirements:
  pip install pypdf nltk anthropic

  # one-time in Python:
  python -c "import nltk; nltk.download('punkt')"

Usage:
  from chunking import chunk_pdf
  result = chunk_pdf('input.pdf')

Features:
  - Uses Claude Haiku (PDF upload) for heading detection
  - Uses pypdf for clean text extraction (removes layout noise for TextTiling)
  - Adds metadata: author, title, page number, section title
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import anthropic
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import nltk
from nltk.tokenize.texttiling import TextTilingTokenizer
from pypdf import PdfReader

from models import Section, Chunk

# -------------------- NLTK check --------------------
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError as e:
    raise SystemExit(
        "NLTK 'punkt' tokenizer data not found.\n"
        "Run: python -c \"import nltk; nltk.download('punkt')\""
    ) from e


# -------------------- PDF extraction (PyPDF) --------------------
def extract_text_per_page(pdf_path: str) -> List[str]:
    """Extract text page-by-page using PyPDF."""
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


# -------------------- Cleaning / normalization --------------------
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_page_text(text: str) -> str:
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

        if idx != 0:
            sep = "\n\n"
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
    overlapped: List[int] = []
    for page_idx, ps, pe in page_spans:
        if e <= ps:
            break
        if s < pe and e > ps:
            overlapped.append(page_idx + 1)
    return overlapped


# -------------------- Line splitting + line-id mapping --------------------
_LINE_WS = re.compile(r"[ \t]+")


def split_doc_into_lines_with_offsets(
    doc_text: str,
    *,
    min_len: int = 1,
    strip: bool = True,
) -> List[Dict[str, Any]]:
    """
    Split doc_text into lines and keep deterministic char offsets.

    Returns list entries:
      {
        "line_id": "L000001",
        "text": "<original line text (optionally stripped)>",
        "start": int,  # start char offset in doc_text (inclusive)
        "end": int,    # end char offset in doc_text (exclusive, includes original line chars, not newline)
      }

    Note:
      - Offsets are with respect to doc_text (the exact concatenation produced by build_document_with_offsets()).
      - We treat '\n' as the line delimiter; returned spans exclude the delimiter.
    """
    lines: List[Dict[str, Any]] = []
    start = 0
    line_no = 0
    n = len(doc_text)

    while start <= n:
        nl = doc_text.find("\n", start)
        if nl == -1:
            end = n
            raw_line = doc_text[start:end]
            next_start = n + 1
        else:
            end = nl
            raw_line = doc_text[start:end]
            next_start = nl + 1

        text = raw_line
        if strip:
            # strip without changing offsets: we keep offsets for raw_line,
            # but store a trimmed version for matching.
            text = text.strip()

        if len(text) >= min_len:
            line_no += 1
            lines.append(
                {
                    "line_id": f"L{line_no:06d}",
                    "text": text,
                    "start": start,
                    "end": end,
                }
            )

        start = next_start
        if start > n:
            break

    return lines


def build_llm_heading_input_from_lines(
    lines: List[Dict[str, Any]],
    *,
    max_chars: int = 180_000,
) -> str:
    """
    Build the text we send to the LLM: each line prefixed by [L000123].
    We keep it under a character budget to avoid huge prompts (PDF is still uploaded separately).
    """
    out: List[str] = []
    total = 0
    for ln in lines:
        # Keep matching stable: collapse internal whitespace for the LLM view.
        # (We still match using offsets from the original line spans.)
        view_text = _LINE_WS.sub(" ", ln["text"]).strip()
        s = f"[{ln['line_id']}] {view_text}"
        if total + len(s) + 1 > max_chars:
            break
        out.append(s)
        total += len(s) + 1
    return "\n".join(out)


# -------------------- TextTiling chunking --------------------
def texttiling_segments(doc_text: str, w: int = 20, k: int = 10, smoothing_width: int = 2) -> List[Tuple[int, int]]:
    tokenizer = TextTilingTokenizer(w=w, k=k, smoothing_width=smoothing_width)
    segments = tokenizer.tokenize(doc_text)

    spans: List[Tuple[int, int]] = []
    cursor = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        pos = doc_text.find(seg, cursor)
        if pos == -1:
            def squash_ws(s: str) -> str:
                return re.sub(r"\s+", " ", s).strip()

            # If we can't align reliably, just take remaining text as the last chunk.
            _ = squash_ws(doc_text[cursor:])
            spans.append((cursor, len(doc_text)))
            cursor = len(doc_text)
            break

        start = pos
        end = pos + len(seg)
        spans.append((start, end))
        cursor = end

    # Merge tiny chunks (optional)
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if (e - s) < 200:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


# -------------------- PDF metadata --------------------
def extract_pdf_metadata(pdf_path: str) -> Dict[str, str]:
    reader = PdfReader(pdf_path)
    metadata = reader.metadata or {}
    return {
        "author": metadata.get("/Author", "") or "",
        "title": metadata.get("/Title", "") or "",
        "subject": metadata.get("/Subject", "") or "",
        "creator": metadata.get("/Creator", "") or "",
    }


DEFAULT_HEADING_SYSTEM_PROMPT = """You are a research assistant that extracts section headings from research paper PDFs.

Return ONLY valid JSON as a SINGLE LINE (minified). Do not use newlines or extra whitespace. Do not wrap in markdown. Do not include any commentary or additional keys.

Schema (single-line JSON):
{"headings":[{"text":"Heading title","level":1,"page_num":1,"line_id":"L000123"}]}

Rules:
- Preserve document order.
- Include only true section/subsection headings (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion, References, Appendix, and numbered/unnumbered headings).
- level=1 top-level, level=2 subsection, level=3 sub-subsection if clearly present.
- Prefer returning line_id when possible, matching the provided [Lxxxxxx] prefixes. If you cannot find a line_id, omit it.
- page_num is 1-based if you can infer it; otherwise omit page_num for that heading (do not set null).
- Exclude figure/table captions, running headers/footers, page numbers, affiliations, keywords, citations, and body text.
- Deduplicate repeated headings caused by headers/footers.
- Do not invent headings; if ambiguous, omit.
- Ensure all JSON string values use standard double quotes, and do not include unescaped newline characters in any string value.

"""


def extract_headings_with_claude(
    pdf_path: str,
    *,
    llm_line_text: Optional[str] = None,
    model: str = "claude-3-5-haiku-latest",
    system_prompt: str = DEFAULT_HEADING_SYSTEM_PROMPT,
    max_tokens: int = 1200,
    development: bool = False,
) -> List[Dict[str, Any]]:
    """
    Upload the PDF to Claude and extract headings as JSON.

    Returns:
      list of {"text": str, "level": int, "page_num": Optional[int], "line_id": Optional[str]}
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")

    client = anthropic.Anthropic(api_key=api_key)

    pdf_bytes = Path(pdf_path).read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    user_text = "Extract the section headings from this research paper PDF and return JSON only."
    if llm_line_text:
        user_text += (
            "\n\nBelow is the PDF text split into lines with stable IDs. "
            "When you identify a heading, include the matching line_id.\n\n"
            f"{llm_line_text}"
        )

    raw = ""
    if not development:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_b64,
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
        except Exception as e:
            logging.error(f"Error calling Anthropic API: {type(e).__name__}: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to call Anthropic API: {type(e).__name__}: {e}"
            ) from e


        raw_text_parts: List[str] = []
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                raw_text_parts.append(getattr(block, "text", "") or "")
        raw = "\n".join(raw_text_parts).strip()

    if development:
        raw = '```json\n{"headings":[{"text":"ABSTRACT","level":1,"line_id":"L000011"},{"text":"1. Affective (but not necessarily effective) entrepreneurship","level":1,"line_id":"L000024"},{"text":"2. Hope (theory) as an (ironically) wasted potential path in entrepreneurship research","level":1,"line_id":"L000071"},{"text":"3. Relegating affect to another cognitive tool in a hope-ful best case","level":1,"line_id":"L000146"},{"text":"Credit author statement","level":1,"line_id":"L000193"},{"text":"Funding","level":1,"line_id":"L000197"},{"text":"Declaration of competing interest","level":1,"line_id":"L000199"},{"text":"References","level":1,"line_id":"L000201"}]}\n```'

    # Parse JSON (simple + strict). If model returns stray text, try to salvage the JSON object.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Claude did not return valid JSON. Raw output:\n{raw[:2000]}")
        data = json.loads(m.group(0))

    headings = data.get("headings", [])
    if not isinstance(headings, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for h in headings:
        if not isinstance(h, dict):
            continue
        text = str(h.get("text", "")).strip()
        if not text:
            continue
        level = h.get("level", 1)
        try:
            level_i = int(level)
        except Exception:
            level_i = 1
        page_num = h.get("page_num", None)
        if page_num is not None:
            try:
                page_num = int(page_num)
            except Exception:
                page_num = None
        line_id = h.get("line_id", None)
        if line_id is not None:
            line_id = str(line_id).strip()
            if not re.fullmatch(r"L\d{6}", line_id):
                line_id = None

        cleaned.append(
            {
                "text": text,
                "level": max(1, min(level_i, 6)),
                "page_num": page_num,
                "line_id": line_id,
            }
        )

    # De-dupe exact repeats while preserving order
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for h in cleaned:
        key = (h["text"].lower(), h["level"], h.get("page_num"), h.get("line_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    return deduped


# -------------------- Heading anchoring --------------------
def _find_heading_in_range(
    heading_text: str,
    doc_text: str,
    start: int,
    end: int,
) -> Optional[Tuple[int, int]]:
    """
    Find heading_text in doc_text[start:end]. Returns absolute (s,e) if found.
    Exact match first; fallback to whitespace-squashed search.
    """
    h = heading_text.strip()
    if not h:
        return None

    hay = doc_text[start:end]
    pos = hay.find(h)
    if pos != -1:
        s = start + pos
        return (s, s + len(h))

    # Fallback: normalize whitespace.
    def squash(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    hay2 = squash(hay)
    h2 = squash(h)
    pos2 = hay2.find(h2)
    if pos2 == -1:
        return None

    # Mapping back exactly is hard; return None so caller can use another strategy.
    return None


def build_sections_from_headings(
    headings: List[Dict[str, Any]],
    doc_text: str,
    page_spans: List[Tuple[int, int, int]],
    lines: List[Dict[str, Any]],
) -> List[Section]:
    """
    Convert Claude headings into Section objects anchored in doc_text char offsets.

    Priority:
      1) If line_id present: use that line's offsets (deterministic).
      2) Else if page_num present: search within that page span for heading text.
      3) Else: global forward search for heading text (best-effort).
    """
    line_by_id: Dict[str, Dict[str, Any]] = {ln["line_id"]: ln for ln in lines}

    sections: List[Section] = []
    cursor = 0  # global forward search cursor to keep order stable

    for h in headings:
        title = h["text"]
        level = int(h.get("level", 1))
        page_num = int(h.get("page_num") or 0)
        line_id = h.get("line_id")

        # 1) Deterministic: line_id -> offsets
        if line_id and line_id in line_by_id:
            ln = line_by_id[line_id]
            s, e = int(ln["start"]), int(ln["end"])
            # Tighten to actual heading within the line if possible
            line_text = doc_text[s:e]
            pos = line_text.find(title)
            if pos != -1:
                s = s + pos
                e = s + len(title)
            sections.append(
                Section(
                    title=title,
                    page_num=page_num or (pages_for_span(page_spans, (s, e))[0] if pages_for_span(page_spans, (s, e)) else 0),
                    char_offset_start=s,
                    char_offset_end=e,
                    level=level,
                )
            )
            cursor = max(cursor, e)
            continue

        # 2) Page-bounded search
        found: Optional[Tuple[int, int]] = None
        if page_num and 1 <= page_num <= len(page_spans):
            page_idx = page_num - 1
            _, ps, pe = page_spans[page_idx]
            found = _find_heading_in_range(title, doc_text, ps, pe)

        # 3) Global forward search (best effort)
        if not found:
            found = _find_heading_in_range(title, doc_text, cursor, len(doc_text))

        # If still not found, skip (don't hallucinate offsets)
        if not found:
            continue

        s, e = found
        inferred_pages = pages_for_span(page_spans, (s, e))
        inferred_page_num = page_num or (inferred_pages[0] if inferred_pages else 0)

        sections.append(
            Section(
                title=title,
                page_num=inferred_page_num,
                char_offset_start=s,
                char_offset_end=e,
                level=level,
            )
        )
        cursor = max(cursor, e)

    # Ensure sorted by start offset
    sections.sort(key=lambda x: x.char_offset_start)
    return sections


def _section_for_chunk(sections: List[Section], chunk_start: int) -> Tuple[str, int]:
    """
    Given chunk_start, return (section_title, section_level) for the most recent section <= chunk_start.
    """
    if not sections:
        return ("", 0)
    current = None
    for s in sections:
        if s.char_offset_start <= chunk_start:
            current = s
        else:
            break
    if not current:
        return ("", 0)
    return (current.title, current.level)


# -------------------- Main API --------------------
def chunk_pdf(
    pdf_path: str,
    *,
    w: int = 20,
    k: int = 10,
    smoothing_width: int = 2,
    heading_model: str = "claude-haiku-4-5",
    heading_system_prompt: str = DEFAULT_HEADING_SYSTEM_PROMPT,
    heading_max_tokens: int = 1200,
    development: bool = False,
) -> List[Chunk]:
    """
    PDF -> text -> headings (Claude) -> TextTiling chunks -> enriched chunk metadata.

    Returns a list of Chunk objects, each containing:
      - chunk_id: sequential chunk number
      - start_char, end_char: character offsets in the document
      - pages: list of 1-based page numbers this chunk spans
      - text: the chunk text content
      - author, title: PDF metadata
      - section_title, section_level: detected section heading info
    """
    pdf_path = str(pdf_path)
    meta = extract_pdf_metadata(pdf_path)

    pages = extract_text_per_page(pdf_path)
    doc_text, page_spans = build_document_with_offsets(pages)

    # Split to lines and build LLM line-ID text (deterministic anchor support)
    lines = split_doc_into_lines_with_offsets(doc_text, min_len=1, strip=True)
    llm_line_text = build_llm_heading_input_from_lines(lines)

    # Headings via Claude (prefer line_id)
    headings = extract_headings_with_claude(
        pdf_path,
        llm_line_text=llm_line_text,
        model=heading_model,
        system_prompt=heading_system_prompt,
        max_tokens=heading_max_tokens,
        development=development,
    )

    # Anchor headings in doc_text -> Section objects
    sections = build_sections_from_headings(headings, doc_text, page_spans, lines)

    # TextTiling chunk spans
    spans = texttiling_segments(doc_text, w=w, k=k, smoothing_width=smoothing_width)

    # Build chunks with section metadata
    chunks: List[Chunk] = []
    for i, (s, e) in enumerate(spans, start=1):
        chunk_text = doc_text[s:e].strip()
        if not chunk_text:
            continue

        chunk_pages = pages_for_span(page_spans, (s, e))
        section_title, section_level = _section_for_chunk(sections, s)

        chunks.append(
            Chunk(
                chunk_id=i,
                start_char=s,
                end_char=e,
                pages=chunk_pages,
                text=chunk_text,
                author=meta.get("author", ""),
                title=meta.get("title", ""),
                section_title=section_title,
                section_level=section_level,
            )
        )

    return chunks
