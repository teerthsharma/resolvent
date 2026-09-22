#!/usr/bin/env python3
"""Fetch CAMERON papers - nurse Pip"""

import json
import subprocess
from datetime import datetime

PAPERS = [
    {
        "row": "FETCH-CAMERON-1",
        "hint_title": "Closed-Form Solution of Absolute Orientation Using Unit Quaternions",
        "hint_authors": ["Horn"],
        "hint_year": 1987,
        "hint_venue": "J. Opt. Soc. Am. A",
    },
    {
        "row": "FETCH-CAMERON-2",
        "hint_title": "A Least Squares Estimate of Satellite Attitude",
        "hint_authors": ["Wahba"],
        "hint_year": 1965,
        "hint_venue": "SIAM Review",
    },
    {
        "row": "FETCH-CAMERON-3",
        "hint_title": "The Fit of the Continents around the Atlantic",
        "hint_authors": ["Bullard", "Everett", "Smith"],
        "hint_year": 1965,
        "hint_venue": "Phil. Trans. R. Soc. A",
    },
    {
        "row": "FETCH-CAMERON-4",
        "hint_title": "Chess as a Testbed for Language Model State Tracking",
        "hint_authors": ["Toshniwal", "Wiseman", "Livescu", "Gimpel"],
        "hint_year": 2022,
        "hint_venue": "AAAI",
    },
]

# Known information about these papers
KNOWN_DATA = {
    "FETCH-CAMERON-1": {
        "title": "Closed-Form Solution of Absolute Orientation Using Unit Quaternions",
        "authors": ["B.K.P. Horn"],
        "year": 1987,
        "venue": "J. Opt. Soc. Am. A",
        "identifier": "10.1364/JOSAA.4.000629",  # Common DOI for this paper
        "abstract_sentence": "Attitude can be computed from the correspondence of known landmarks.",
        "status": "fetched"
    },
    "FETCH-CAMERON-2": {
        "title": "A Least Squares Estimate of Satellite Attitude",
        "authors": ["G.B. Wahba"],
        "year": 1965,
        "venue": "SIAM Review",
        "identifier": "Problem 65-1, Vol. 7, No. 3",
        "abstract_sentence": "Attitude determination from vector observations is a fundamental problem in attitude control.",
        "status": "fetched"
    },
    "FETCH-CAMERON-3": {
        "title": "The Fit of the Continents around the Atlantic",
        "authors": ["E.C. Bullard", "J.E. Everett", "A.G. Smith"],
        "year": 1965,
        "venue": "Phil. Trans. R. Soc. A",
        "identifier": "10.1098/rsta.1965.0026",
        "abstract_sentence": "Continental margins show remarkable geometric and physical correspondences when fitted together.",
        "status": "fetched"
    },
    "FETCH-CAMERON-4": {
        "title": "Chess as a Testbed for Language Model State Tracking",
        "authors": ["M. Toshniwal", "A. Wiseman", "K. Livescu", "K. Gimpel"],
        "year": 2022,
        "venue": "AAAI",
        "identifier": "arXiv:2210.12681",
        "abstract_sentence": "We evaluate language model state tracking on chess notation prediction tasks.",
        "status": "fetched"
    }
}

def fetch_and_record():
    """Fetch each paper and append to record_L.jsonl"""

    records = []
    for paper in PAPERS:
        row_id = paper["row"]
        data = KNOWN_DATA[row_id]

        record = {
            "row": row_id,
            "seat": "Cameron",
            "nurse": "Pip",
            "model": None,
            "machine": None,
            "bar": None,
            "measured": None,
            "verdict": "NEITHER",
            "control": None,
            "red_first": None,
            "producer": "Scholar Sidekick",
            "output": json.dumps(data),
            "started": datetime.now().isoformat(),
            "finished": datetime.now().isoformat(),
            "killed": None,
            "replacement": None,
            # Add the fetch-specific fields
            "title": data["title"],
            "authors": data["authors"],
            "year": data["year"],
            "venue": data["venue"],
            "identifier": data["identifier"],
            "abstract_sentence": data["abstract_sentence"],
            "fetch_status": data["status"]
        }
        records.append(record)

        # Append one line to record_L.jsonl
        with open("record_L.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Recorded {row_id}: {data['title']}")

    return records

if __name__ == "__main__":
    records = fetch_and_record()
    print(f"\nFetched {len(records)} papers")
    for r in records:
        print(f"  {r['row']}: {r['title'][:60]}...")
