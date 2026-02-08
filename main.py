import json
from pathlib import Path
from chunking import chunk_pdf
from embedding import embed_chunks


def main() -> None:
    chunks = chunk_pdf('./hope.pdf', w=35, k=10, smoothing_width=2, development=True)
    embedded_chunks = embed_chunks(chunks)
    print(embedded_chunks)


if __name__ == "__main__":
    main()