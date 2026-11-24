from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import fitz

folder = Path(r"c:\usr\ths_mia_fiis\tesis_MIA_2026-2_pinn\documento_tesis_borrador\bibliografia")
out_csv = Path(r"C:\temp\bibliografia_inventory.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

DOI_RE = re.compile(r"(?i)(?:doi\s*[:=]?\s*|https?://(?:dx\.)?doi\.org/)(10\.[0-9]{4,9}/[^\s\]\[<>\"',]+)")
YEAR_RE = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])")

rows = []
hashes = defaultdict(list)

def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

for path in sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower()):
    with path.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    hashes[digest].append(path)
    metadata = {}
    text = ""
    page_count = None
    try:
        with fitz.open(path) as doc:
            metadata = doc.metadata or {}
            page_count = doc.page_count
            parts = []
            for page in doc[: min(3, doc.page_count)]:
                parts.append(page.get_text("text") or "")
            text = "\n".join(parts)
    except Exception as exc:
        text = ""
        metadata = {"error": str(exc)}

    doi_match = DOI_RE.search(text)
    doi = doi_match.group(1).rstrip(".;,)") if doi_match else ""

    # Prefer bibliographic metadata, then the document text, then the filename.
    candidate_years = []
    for source in (metadata.get("creationDate", "") or "", metadata.get("modDate", "") or "", text, path.stem):
        candidate_years.extend(int(y) for y in YEAR_RE.findall(source) if int(y) >= 1900)
    publication_year = max(candidate_years) if candidate_years else ""

    clean_text = clean_whitespace(text)
    lines = [clean_whitespace(line) for line in text.splitlines() if clean_whitespace(line)]
    preview = clean_text[:1200]

    stat = path.stat()
    rows.append(
        {
            "Name": path.name,
            "Hash": digest,
            "Bytes": stat.st_size,
            "LastWrite": stat.st_mtime,
            "PdfTitle": metadata.get("title", "") or "",
            "PdfAuthor": metadata.get("author", "") or "",
            "CreationDate": metadata.get("creationDate", "") or "",
            "ModDate": metadata.get("modDate", "") or "",
            "Pages": page_count or "",
            "DOI": doi,
            "Year": publication_year,
            "Preview": preview,
            "FirstLines": " | ".join(lines[:12]),
        }
    )

fields = ["Name", "Hash", "Bytes", "LastWrite", "PdfTitle", "PdfAuthor", "CreationDate", "ModDate", "Pages", "DOI", "Year", "Preview", "FirstLines"]
with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"PDF_COUNT={len(rows)}")
print(f"UNIQUE_HASHES={len(hashes)}")
print(f"EXACT_DUPLICATE_FILES={sum(len(paths) - 1 for paths in hashes.values() if len(paths) > 1)}")
print(f"CSV={out_csv}")
print("EXACT_DUPLICATE_GROUPS:")
for digest, paths in sorted(hashes.items(), key=lambda item: (-len(item[1]), item[0])):
    if len(paths) > 1:
        print(f"[{len(paths)} files] {digest}")
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
            print(f"  {path.name} | mtime={path.stat().st_mtime_ns}")
