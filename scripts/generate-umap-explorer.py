#!/usr/bin/env python3
"""Generate an interactive UMAP-MLX cluster explorer for the Rijksmuseum collection.

Pipeline: load embeddings -> UMAP-MLX (Metal GPU) -> HDBSCAN -> interactive HTML.

Usage:
    uv run python scripts/generate-umap-explorer.py              # full 831K
    uv run python scripts/generate-umap-explorer.py --sample 20000  # 20K sample (~2 min on M4)
    uv run python scripts/generate-umap-explorer.py --type painting  # paintings only
    uv run python scripts/generate-umap-explorer.py --type painting --creator "Rijn, Rembrandt van"
    uv run python scripts/generate-umap-explorer.py --subject dog    # all artworks depicting dogs
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.embeddings import (
    PROJECT_ROOT,
    load_embeddings,
    load_metadata,
    build_hover_text,
)
from scripts._filters import collect_filters, apply_filters, filter_suffix, default_output, autoscale_hdbscan
from scripts._html_template import build_cluster_traces, generate_explorer_html

OUTPUT_DIR = PROJECT_ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP-MLX cluster explorer")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (default: all 831K)")
    parser.add_argument("--type", type=str, default=None, help="Filter by object type (e.g. 'painting')")
    parser.add_argument("--creator", type=str, default=None, help="Filter by creator (e.g. 'Rijn, Rembrandt van')")
    parser.add_argument("--subject", type=str, default=None, help="Filter by subject (e.g. 'dog')")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--min-cluster-size", type=int, default=None, help="HDBSCAN min_cluster_size (auto-scaled if omitted)")
    parser.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min_samples (auto-scaled if omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output HTML filename (auto-generated if omitted)")
    args = parser.parse_args()

    filters = collect_filters(args)
    allowed_ids = apply_filters(filters)

    # 1. Load embeddings
    if allowed_ids is not None:
        if not allowed_ids:
            print("  No matching artworks — exiting.")
            return

        if args.sample and args.sample > len(allowed_ids):
            print(f"  Note: only {len(allowed_ids):,} available, using all")
        print("Loading embeddings...")
        t0 = time.time()
        art_ids, object_numbers, embeddings = load_embeddings(seed=args.seed)
        mask = np.array([aid in allowed_ids for aid in art_ids])
        art_ids = [aid for aid, m in zip(art_ids, mask) if m]
        object_numbers = [obj for obj, m in zip(object_numbers, mask) if m]
        embeddings = embeddings[mask]
        if args.sample and args.sample < len(art_ids):
            random.seed(args.seed)
            indices = sorted(random.sample(range(len(art_ids)), args.sample))
            art_ids = [art_ids[i] for i in indices]
            object_numbers = [object_numbers[i] for i in indices]
            embeddings = embeddings[indices]
    else:
        desc = f" (sample={args.sample:,})" if args.sample else " (all)"
        print(f"Loading embeddings{desc}...")
        t0 = time.time()
        art_ids, object_numbers, embeddings = load_embeddings(sample_size=args.sample, seed=args.seed)

    n = len(art_ids)
    print(f"  Loaded {n:,} embeddings in {time.time() - t0:.1f}s")

    min_cluster_size, min_samples = autoscale_hdbscan(n, args.min_cluster_size, args.min_samples)

    # 2. UMAP-MLX dimensionality reduction
    print(f"Running UMAP-MLX (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
    from mlx_vis import UMAP

    t0 = time.time()
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    coords = np.asarray(coords, dtype=np.float32)
    print(f"  UMAP done in {time.time() - t0:.1f}s")

    # Free high-dim embeddings and reducer — only 2D coords needed from here
    del embeddings, reducer

    # 3. HDBSCAN clustering
    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    import hdbscan

    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
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

    suffix = filter_suffix(filters)
    subtitle = (
        f"{n:,} artworks{suffix} \u00b7 {result['n_clusters']} clusters \u00b7 "
        f"UMAP-MLX n_neighbors={args.n_neighbors} min_dist={args.min_dist}"
    )
    title = f"UMAP-MLX Embedding Clusters{suffix}"
    html = generate_explorer_html(
        title=title,
        subtitle=subtitle,
        axis_label="UMAP",
        **result,
    )

    # 6. Write output
    OUTPUT_DIR.mkdir(exist_ok=True)
    html_path = OUTPUT_DIR / (args.output or default_output("umap", filters))
    html_path.write_text(html)
    print(f"\nSaved: {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")

    # Save coordinates for reuse
    npz_path = OUTPUT_DIR / "umap-coords.npz"
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
