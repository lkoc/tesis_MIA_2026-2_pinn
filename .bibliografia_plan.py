from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

inventory = Path(r"C:\temp\bibliografia_inventory.csv")
report = Path(r"C:\temp\bibliografia_plan.txt")
winners_csv = Path(r"C:\temp\bibliografia_winners.csv")
losers_csv = Path(r"C:\temp\bibliografia_losers.csv")

rows = list(csv.DictReader(inventory.open("r", encoding="utf-8-sig")))
groups = defaultdict(list)
for row in rows:
    groups[row["Hash"]].append(row)

winners = [max(group, key=lambda row: float(row["LastWrite"])) for group in groups.values()]
losers = [row for group in groups.values() for row in group if row is not max(group, key=lambda item: float(item["LastWrite"]))]


def normalize_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(?i)\bzotero\b", " ", text)
    text = re.sub(r"\bet\s+al\.?\b", " ", text)
    text = re.sub(r"\b(?:19|20)[0-9]{2}\b", " ", text)
    text = re.sub(r"\b(?:physics informed|physics-informed)\b", " physics informed ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [word for word in text.split() if len(word) > 2]
    return " ".join(words)


def crossref_metadata(doi: str) -> dict:
    if not doi:
        return {}
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "tesis-bibliografia-cleanup/1.0 (mailto:local-user@example.com)"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            message = json.load(response)["message"]
        title = (message.get("title") or [""])[0]
        year = ""
        for key in ("published", "published-print", "published-online", "issued", "created"):
            date_parts = (message.get(key) or {}).get("date-parts") or []
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])
                break
        authors = message.get("author") or []
        surname = ""
        if authors:
            surname = authors[0].get("family") or ""
        return {"title": title, "year": year, "surname": surname, "authors": authors}
    except Exception as exc:
        return {"error": str(exc)}

crossref_cache = {}
for row in sorted(winners, key=lambda item: item["Name"].lower()):
    doi = row["DOI"]
    if doi:
        crossref_cache[doi] = crossref_metadata(doi)
        time.sleep(0.15)

lines = []
lines.append(f"PDF_COUNT={len(rows)}")
lines.append(f"UNIQUE_HASHES={len(groups)}")
lines.append(f"EXACT_DUPLICATE_FILES={len(losers)}")
lines.append(f"WINNER_COUNT={len(winners)}")
lines.append("")
lines.append("WINNERS:")
for index, row in enumerate(sorted(winners, key=lambda item: item["Name"].lower()), start=1):
    metadata = crossref_cache.get(row["DOI"], {})
    lines.append("\t".join([
        f"{index:03d}",
        row["DOI"] or "-",
        str(metadata.get("year") or row["Year"] or "-"),
        metadata.get("surname") or "-",
        metadata.get("title") or row["PdfTitle"] or "-",
        row["Name"],
        row["FirstLines"][:240].replace("\n", " "),
    ]))

# Exact DOI duplicates outside identical hashes.
doi_groups = defaultdict(list)
for row in winners:
    if row["DOI"]:
        doi_groups[row["DOI"].lower()].append(row)
lines.append("")
lines.append("EXACT_DOI_GROUPS_IN_WINNERS:")
for doi, group in sorted(doi_groups.items(), key=lambda item: item[0]):
    if len(group) > 1:
        lines.append(doi)
        for row in group:
            lines.append("  " + row["Name"])

# Conservative title similarity pairs for manual review.
lines.append("")
lines.append("TITLE_SIMILARITY_PAIRS_RATIO_GE_0.78:")
ordered = sorted(winners, key=lambda item: item["Name"].lower())
for i in range(len(ordered)):
    title_i = normalize_title(ordered[i]["FirstLines"][:500] or ordered[i]["Name"])
    for j in range(i + 1, len(ordered)):
        title_j = normalize_title(ordered[j]["FirstLines"][:500] or ordered[j]["Name"])
        ratio = SequenceMatcher(None, title_i, title_j).ratio()
        if ratio >= 0.78:
            lines.append(f"{ratio:.3f}\t{ordered[i]['Name']}\t{ordered[j]['Name']}")

report.write_text("\n".join(lines), encoding="utf-8")

fields = ["Name", "Hash", "Bytes", "LastWrite", "PdfTitle", "PdfAuthor", "CreationDate", "ModDate", "Pages", "DOI", "Year", "FirstLines"]
with winners_csv.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(winners)
with losers_csv.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(losers)

print(report)
print(f"WINNERS={len(winners)}")
print(f"LOSERS={len(losers)}")
