#!/usr/bin/env python3
"""Export Rijksmuseum embedding data for Wolfram Mathematica analysis.

Creates binary + JSON files that Mathematica can efficiently import:
  - embeddings.bin  — N×384 raw float32 matrix
  - metadata.json   — art IDs, object numbers, titles, creator_label,
                       date_display, date_earliest, date_latest, has_image,
                       iiif_id, importance, types, creators, subjects,
                       materials, techniques, plus a schema_version hint.

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
    load_embeddings,
    load_metadata,
)


def read_schema_version(vdb_path: Path = DEFAULT_VOCAB_DB) -> str | None:
    """Return the vocabulary.db build timestamp, or None if not available."""
    try:
        with sqlite3.connect(str(vdb_path)) as conn:
            row = conn.execute(
                "SELECT value FROM version_info WHERE key='built_at'"
            ).fetchone()
            return row[0] if row else None
    except sqlite3.Error:
        return None


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

    # Load metadata (now includes date + denormalized display fields natively)
    print("Loading metadata...")
    meta = load_metadata(art_ids, object_numbers)

    # Build JSON structure. All existing keys are preserved for the 5 Mathematica
    # notebooks; new keys are additive so older .nb files still work unchanged.
    artworks = {}
    for aid, onum in zip(art_ids, object_numbers):
        m = meta.get(aid, {})
        artworks[str(aid)] = {
            "object_number": onum,
            "title": m.get("title", ""),
            "creator_label": m.get("creator_label", ""),
            "date_display": m.get("date_display", ""),
            "date_earliest": m.get("date_earliest"),
            "date_latest": m.get("date_latest"),
            "has_image": m.get("has_image", False),
            "iiif_id": m.get("iiif_id", ""),
            "importance": m.get("importance", 0),
            "types": m.get("types", []),
            "creators": m.get("creators", []),
            "subjects": m.get("subjects", []),
            "materials": m.get("materials", []),
            "techniques": m.get("techniques", []),
        }

    manifest = {
        "n": n,
        "dim": dim,
        "seed": args.seed,
        "schema_version": read_schema_version(),
        "art_ids": art_ids,
        "object_numbers": object_numbers,
        "artworks": artworks,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(manifest, f)
    print(f"Wrote metadata to {meta_path} ({meta_path.stat().st_size / 1e6:.1f} MB)")
    print(f"schema_version: {manifest['schema_version']}")
    print("Done! Data ready for Mathematica import.")


if __name__ == "__main__":
    main()
