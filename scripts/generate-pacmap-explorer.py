#!/usr/bin/env python3
"""Generate an interactive PaCMAP-MLX cluster explorer for the Rijksmuseum collection.

Pipeline: load embeddings → PaCMAP-MLX (Metal GPU) → HDBSCAN → interactive HTML.

Usage:
    uv run python scripts/generate-pacmap-explorer.py              # full 831K
    uv run python scripts/generate-pacmap-explorer.py --sample 20000  # 20K sample (~2 min on M4)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.embeddings import PROJECT_ROOT, load_embeddings, load_metadata, build_hover_text
from scripts._html_template import build_cluster_traces, generate_explorer_html

OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description="Generate PaCMAP-MLX cluster explorer")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (default: all 831K)")
    parser.add_argument("--n-neighbors", type=int, default=10, help="PaCMAP n_neighbors (default: 10)")
    parser.add_argument("--mn-ratio", type=float, default=0.5, help="PaCMAP MN_ratio (default: 0.5)")
    parser.add_argument("--fp-ratio", type=float, default=2.0, help="PaCMAP FP_ratio (default: 2.0)")
    parser.add_argument("--min-cluster-size", type=int, default=100, help="HDBSCAN min_cluster_size (default: 100)")
    parser.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output HTML filename (default: pacmap-explorer.html)")
    args = parser.parse_args()

    # 1. Load embeddings
    print(f"Loading embeddings{f' (sample={args.sample:,})' if args.sample else ' (all)'}...")
    t0 = time.time()
    art_ids, object_numbers, embeddings = load_embeddings(sample_size=args.sample, seed=args.seed)
    n = len(art_ids)
    print(f"  Loaded {n:,} embeddings in {time.time() - t0:.1f}s")

    # 2. PaCMAP-MLX dimensionality reduction
    print(f"Running PaCMAP-MLX (n_neighbors={args.n_neighbors}, MN_ratio={args.mn_ratio}, FP_ratio={args.fp_ratio})...")
    from pacmap_mlx import PaCMAP

    t0 = time.time()
    reducer = PaCMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        MN_ratio=args.mn_ratio,
        FP_ratio=args.fp_ratio,
        random_state=args.seed,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    coords = np.asarray(coords, dtype=np.float32)
    print(f"  PaCMAP done in {time.time() - t0:.1f}s")

    # Free high-dim embeddings and reducer — only 2D coords needed from here
    del embeddings, reducer

    # 3. HDBSCAN clustering
    print(f"Running HDBSCAN (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})...")
    import hdbscan

    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    labels = clusterer.fit_predict(coords)
    n_noise = int((labels == -1).sum())
    n_clust = len(set(labels)) - (1 if n_noise > 0 else 0)
    print(f"  {n_clust} clusters, {n_noise:,} noise points ({n_noise/n*100:.1f}%) in {time.time() - t0:.1f}s")
    del clusterer

    # 4. Load metadata
    print("Loading metadata...")
    t0 = time.time()
    meta = load_metadata(art_ids, object_numbers)
    hover_texts = build_hover_text(meta, art_ids)
    print(f"  Metadata loaded in {time.time() - t0:.1f}s")

    # 5. Build traces and generate HTML
    print("Building HTML...")
    result = build_cluster_traces(
        labels=labels,
        coords=coords,
        art_ids=art_ids,
        object_numbers=object_numbers,
        meta=meta,
        hover_texts=hover_texts,
    )

    subtitle = (
        f"{n:,} artworks \u00b7 {result['n_clusters']} clusters \u00b7 "
        f"PaCMAP-MLX n_neighbors={args.n_neighbors} MN={args.mn_ratio} FP={args.fp_ratio}"
    )
    html = generate_explorer_html(
        title="PaCMAP-MLX Embedding Clusters",
        subtitle=subtitle,
        axis_label="PaCMAP",
        **result,
    )

    # 6. Write output
    OUTPUT_DIR.mkdir(exist_ok=True)
    html_path = OUTPUT_DIR / (args.output or "pacmap-explorer.html")
    html_path.write_text(html)
    print(f"\nSaved: {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")

    # Save coordinates for reuse
    npz_path = OUTPUT_DIR / "pacmap-coords.npz"
    np.savez_compressed(
        npz_path,
        coords=coords,
        labels=labels,
        art_ids=np.array(art_ids, dtype=np.int64),
        object_numbers=np.array(object_numbers, dtype=str),
    )
    print(f"Saved: {npz_path} ({npz_path.stat().st_size / 1024:.0f} KB)")
    print(f"\nOpen with: open '{html_path}'")


if __name__ == "__main__":
    main()
