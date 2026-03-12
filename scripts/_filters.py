"""Shared filter helpers for explorer generator scripts."""

import re

from lib.embeddings import filter_by_field


def collect_filters(args) -> dict[str, str]:
    """Return {field_name: value} for all active CLI filters."""
    filters = {}
    if args.type:
        filters["type"] = args.type
    if args.creator:
        filters["creator"] = args.creator
    if args.subject:
        filters["subject"] = args.subject
    return filters


def apply_filters(filters: dict[str, str]) -> set[int] | None:
    """Intersect filter results. Returns None if no filters are active."""
    if not filters:
        return None
    allowed = None
    for field, value in filters.items():
        print(f"Filtering by {field}={value!r}...")
        ids = filter_by_field(field, value)
        print(f"  {len(ids):,} artworks match {field}={value!r}")
        allowed = ids if allowed is None else allowed & ids
    if len(filters) > 1:
        print(f"  {len(allowed):,} artworks match all filters")
    return allowed


def filter_suffix(filters: dict[str, str]) -> str:
    """Build a human-readable suffix like ' (paintings, subject: dog)'."""
    if not filters:
        return ""
    parts = []
    for field, value in filters.items():
        if field == "type":
            parts.append(f"{value}s")
        else:
            parts.append(f"{field}: {value}")
    return f" ({', '.join(parts)})"


def _slugify(text: str) -> str:
    """Turn a free-text value into a filename-safe slug."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def default_output(method: str, filters: dict[str, str]) -> str:
    """Build a descriptive default filename from the method name and active filters.

    Examples:
        umap, {}                                -> umap-explorer.html
        umap, {type: painting}                  -> umap-paintings.html
        pacmap, {type: painting, creator: ...}  -> pacmap-rijn-rembrandt-van-paintings.html
        tsne, {subject: dog}                    -> tsne-dog.html
    """
    if not filters:
        return f"{method}-explorer.html"
    parts = []
    # creator and subject before type so "rembrandt-paintings" reads naturally
    for field in ("creator", "subject", "type"):
        if field not in filters:
            continue
        value = filters[field]
        if field == "type":
            parts.append(_slugify(value) + "s")
        else:
            parts.append(_slugify(value))
    return f"{method}-{'-'.join(parts)}.html"


def autoscale_hdbscan(n: int, min_cluster_size: int | None, min_samples: int | None) -> tuple[int, int]:
    """Auto-scale HDBSCAN parameters to dataset size when not explicitly set.

    Returns (min_cluster_size, min_samples). Prints a message if auto-scaling was applied.
    """
    mcs = min_cluster_size or max(5, min(100, n // 50))
    ms = min_samples or max(2, min(10, n // 100))
    if min_cluster_size is None or min_samples is None:
        print(f"  Auto-scaled HDBSCAN: min_cluster_size={mcs}, min_samples={ms}")
    return mcs, ms
