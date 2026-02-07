import json
from pathlib import Path
from chunking import chunk_pdf


def main() -> None:
    result = chunk_pdf('./hope.pdf', w=35, k=10, smoothing_width=2, development=True)
    print(result)


if __name__ == "__main__":
    main()