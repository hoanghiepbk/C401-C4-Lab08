import json
from pathlib import Path
from index import preprocess_document, chunk_document

DOCS_DIR = Path("data/docs")
for filepath in sorted(DOCS_DIR.glob("*.txt")):
    raw_text = filepath.read_text(encoding="utf-8")
    doc = preprocess_document(raw_text, str(filepath))
    chunks = chunk_document(doc)
    print(f"{filepath.name}: {len(chunks)} chunks")
