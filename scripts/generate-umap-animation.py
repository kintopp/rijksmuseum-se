#!/usr/bin/env python3
"""Generate a UMAP-MLX epoch-by-epoch animation of Rijksmuseum artwork embeddings.

Captures a snapshot at every epoch, then renders an MP4 showing embeddings
evolving from spectral initialization to final layout. Points are colored
by object type (top 10 categories + "other").

Inspired by github.com/hanxiao/umap-mlx/blob/main/fashion_mnist_anim.py

Usage:
    uv run python scripts/generate-umap-animation.py                    # 20K sample, 200 epochs
    uv run python scripts/generate-umap-animation.py --sample 50000     # 50K sample
    uv run python scripts/generate-umap-animation.py --n-epochs 500     # more epochs = smoother
    uv run python scripts/generate-umap-animation.py --fps 30           # lower fps = smaller file
    uv run python scripts/generate-umap-animation.py --snapshot-every 3  # capture every 3rd epoch (less memory)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.embeddings import PROJECT_ROOT, load_embeddings, load_metadata

OUTPUT_DIR = PROJECT_ROOT / "output"

# Top object types to highlight (the rest become "other")
TOP_N_TYPES = 10

# Memory budget for epoch snapshots (each snapshot is N × 2 × float32 bytes).
# Auto snap_every is computed to keep total snapshot memory under this limit.
SNAPSHOT_MEMORY_BUDGET = 1_500_000_000  # 1.5 GB


def get_type_colors(art_ids, meta, top_n=TOP_N_TYPES):
    """Assign colors by primary object type, grouping rare types as 'other'.

    Returns (colors, legend_entries) where colors is an (N, 4) RGBA array
    and legend_entries is a list of (label, color) for the legend.
    """
    from collections import Counter

    primary_types = []
    for aid in art_ids:
        types = meta[aid]["types"]
        primary_types.append(types[0] if types else "unknown")

    # Find top N types by frequency
    counts = Counter(primary_types)
    top_types = [t for t, _ in counts.most_common(top_n)]
    type_to_idx = {t: i for i, t in enumerate(top_types)}

    # Assign color indices: top types get 0..N-1, everything else gets N ("other")
    n_categories = top_n + 1
    cmap = plt.cm.tab10 if n_categories <= 11 else plt.cm.tab20
    color_values = [cmap(i / max(n_categories - 1, 1)) for i in range(n_categories)]

    indices = np.array([type_to_idx.get(t, top_n) for t in primary_types])
    colors = np.array([color_values[i] for i in indices])

    # Build legend entries
    legend_entries = []
    for i, t in enumerate(top_types):
        legend_entries.append((f"{t} ({counts[t]:,})", color_values[i]))
    other_count = sum(c for t, c in counts.items() if t not in type_to_idx)
    if other_count > 0:
        legend_entries.append((f"other ({other_count:,})", color_values[top_n]))

    return colors, legend_entries


def get_square_lims(emb, margin=0.1):
    """Compute square axis limits centered on the embedding."""
    cx = (emb[:, 0].min() + emb[:, 0].max()) / 2
    cy = (emb[:, 1].min() + emb[:, 1].max()) / 2
    span = max(emb[:, 0].max() - emb[:, 0].min(), emb[:, 1].max() - emb[:, 1].min())
    hs = span / 2 * (1 + margin)
    return (cx - hs, cx + hs), (cy - hs, cy + hs)


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP-MLX animation")
    parser.add_argument("--sample", type=int, default=20_000,
                        help="Number of embeddings to sample (default: 20000)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="UMAP epochs (default: auto — 500 for ≤10K, 200 for larger)")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help="Capture every Nth epoch (default: auto, targets ≤1.5 GB snapshot memory)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Animation FPS (default: 60)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: output/umap-animation.mp4)")
    args = parser.parse_args()

    # 1. Load embeddings
    print(f"Loading embeddings (sample={args.sample:,})...")
    t0 = time.time()
    art_ids, object_numbers, embeddings = load_embeddings(
        sample_size=args.sample, seed=args.seed
    )
    n = len(art_ids)
    print(f"  Loaded {n:,} embeddings ({embeddings.shape[1]}d) in {time.time() - t0:.1f}s")

    # 2. Load metadata for coloring
    print("Loading metadata...")
    t0 = time.time()
    meta = load_metadata(art_ids, object_numbers)
    colors, legend_entries = get_type_colors(art_ids, meta)
    print(f"  Metadata loaded in {time.time() - t0:.1f}s")
    print(f"  Color categories: {len(legend_entries)}")
    for label, _ in legend_entries:
        print(f"    {label}")

    # Free metadata — only colors and legend_entries are needed from here
    del meta, art_ids, object_numbers

    # 3. Auto-compute snapshot interval to bound memory usage
    snap_every = args.snapshot_every
    if snap_every is None:
        snap_bytes = n * 2 * 4  # bytes per snapshot (N × 2 × float32)
        expected_epochs = args.n_epochs or (500 if n <= 10_000 else 200)
        snap_every = max(1, math.ceil(snap_bytes * (expected_epochs + 1) / SNAPSHOT_MEMORY_BUDGET))

    est_snaps = ((args.n_epochs or 200) // snap_every) + 1
    est_mem_mb = est_snaps * n * 2 * 4 / 1e6
    print(f"  Snapshot interval: every {snap_every} epoch(s) (~{est_snaps} snapshots, ~{est_mem_mb:.0f} MB)")

    # 4. Run UMAP-MLX with epoch callback to capture snapshots
    snaps = []
    snap_epochs = []
    snap_times = []
    t_global = time.time()
    last_epoch = 0

    def on_epoch(epoch, Y_np):
        nonlocal last_epoch
        last_epoch = epoch
        if epoch % snap_every == 0:
            snaps.append(Y_np.copy())
            snap_epochs.append(epoch)
            snap_times.append(time.time() - t_global)

    from umap_mlx import UMAP

    umap_kwargs = dict(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        verbose=True,
    )
    if args.n_epochs is not None:
        umap_kwargs["n_epochs"] = args.n_epochs

    print(f"\nRunning UMAP-MLX on {n:,} × {embeddings.shape[1]} ...")
    UMAP(**umap_kwargs).fit_transform(embeddings, epoch_callback=on_epoch)
    t_total = time.time() - t_global
    n_epochs = last_epoch
    print(f"  Done: {t_total:.1f}s, {len(snaps)} snapshots ({n_epochs} epochs, every {snap_every})")

    # Free embeddings — only snaps and colors are needed for rendering
    del embeddings

    # 5. Build animation
    xlim, ylim = get_square_lims(snaps[-1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    scatter = ax.scatter([], [], s=1.0, alpha=0.5)

    title = ax.set_title("", color="white", fontsize=12, pad=10, fontfamily="monospace")

    # Legend (top-right, small)
    legend_handles = []
    for label, color in legend_entries:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="black", markerfacecolor=color,
                       markersize=5, label=label, linestyle="None")
        )
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=7,
        facecolor="black", edgecolor="#333", labelcolor="white",
        framealpha=0.8, handletextpad=0.3, borderpad=0.4,
    )

    # Frame plan: hold init, play epochs, hold final
    init_frames = 30
    hold_frames = 120
    n_snap = len(snaps)
    total_frames = init_frames + n_snap + hold_frames

    print(f"\nRendering {total_frames} frames at {args.fps} fps...")

    def update(frame):
        if frame < init_frames:
            idx = 0
            t = snap_times[0]
            label = (f"umap-mlx  Rijksmuseum  {n:,} × 384  "
                     f"init  t={t:.2f}s")
        elif frame < init_frames + n_snap:
            idx = frame - init_frames
            t = snap_times[idx]
            epoch = snap_epochs[idx]
            label = (f"umap-mlx  Rijksmuseum  {n:,} × 384  "
                     f"epoch {epoch}/{n_epochs}  t={t:.2f}s")
        else:
            idx = n_snap - 1
            label = (f"umap-mlx  Rijksmuseum  {n:,} × 384  "
                     f"done in {t_total:.1f}s")

        scatter.set_offsets(snaps[idx])
        scatter.set_color(colors)
        title.set_text(label)
        return scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, blit=True, interval=1000 // args.fps
    )

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    outpath = args.output or str(OUTPUT_DIR / "umap-animation.mp4")

    anim.save(
        outpath,
        writer=animation.FFMpegWriter(
            fps=args.fps,
            bitrate=8000,
            extra_args=["-pix_fmt", "yuv420p"],
        ),
    )
    plt.close()

    size_mb = os.path.getsize(outpath) / 1024 / 1024
    duration = total_frames / args.fps
    print(f"\nSaved: {outpath} ({size_mb:.1f} MB, {duration:.1f}s at {args.fps} fps)")
    print(f"Open with: open '{outpath}'")


if __name__ == "__main__":
    main()
