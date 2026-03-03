#!/usr/bin/env python3
"""Export Rijksmuseum embedding data for Wolfram Mathematica analysis.

Creates binary + JSON files that Mathematica can efficiently import:
  - embeddings.bin  — N×384 raw float32 matrix
  - metadata.json   — art IDs, object numbers, titles, types, creators,
                       subjects, materials, techniques, dates

Usage:
    uv run python mathematica/export_for_mathematica.py [--sample-size N] [--seed S]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.embeddings import (
    DEFAULT_VOCAB_DB,
    _BATCH_SIZE,
    load_embeddings,
    load_metadata,
)


def load_dates(art_ids: list[int], vdb_path: Path = DEFAULT_VOCAB_DB) -> dict:
    """Fetch date_earliest / date_latest from vocabulary DB."""
    dates = {}
    with sqlite3.connect(str(vdb_path)) as conn:
        for i in range(0, len(art_ids), _BATCH_SIZE):
            batch = art_ids[i : i + _BATCH_SIZE]
            ph = ",".join("?" * len(batch))
            for aid, de, dl in conn.execute(
                f"SELECT art_id, date_earliest, date_latest FROM artworks WHERE art_id IN ({ph})",
                batch,
            ):
                dates[aid] = (de, dl)
    return dates


def main():
    parser = argparse.ArgumentParser(description="Export data for Mathematica")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.sample_size} embeddings (seed={args.seed})...")
    art_ids, object_numbers, embeddings = load_embeddings(
        sample_size=args.sample_size, seed=args.seed
    )
    n, dim = embeddings.shape
    print(f"Loaded {n} embeddings of dimension {dim}")

    # Export embeddings as raw float32 binary
    emb_path = out_dir / "embeddings.bin"
    embeddings.astype(np.float32).tofile(emb_path)
    print(f"Wrote embeddings to {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")

    # Load metadata
    print("Loading metadata...")
    meta = load_metadata(art_ids, object_numbers)
    dates = load_dates(art_ids)

    # Build JSON structure
    artworks = {}
    for aid, onum in zip(art_ids, object_numbers):
        m = meta.get(aid, {})
        de, dl = dates.get(aid, (None, None))
        artworks[str(aid)] = {
            "object_number": onum,
            "title": m.get("title", ""),
            "types": m.get("types", []),
            "creators": m.get("creators", []),
            "subjects": m.get("subjects", []),
            "materials": m.get("materials", []),
            "techniques": m.get("techniques", []),
            "date_earliest": de,
            "date_latest": dl,
        }

    manifest = {
        "n": n,
        "dim": dim,
        "seed": args.seed,
        "art_ids": art_ids,
        "object_numbers": object_numbers,
        "artworks": artworks,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(manifest, f)
    print(f"Wrote metadata to {meta_path} ({meta_path.stat().st_size / 1e6:.1f} MB)")
    print("Done! Data ready for Mathematica import.")


if __name__ == "__main__":
    main()
