"""Shared filter helpers for explorer generator scripts."""

import re

from lib.embeddings import filter_by_artwork_column, filter_by_field

# Vocabulary-mapped filters resolve through field_lookup/mappings/vocabulary.
_VOCAB_FIELDS = ("type", "creator", "subject", "production_place")

# Artwork-column filters run as direct WHERE clauses on the artworks table.
_COLUMN_FILTERS = ("date_from", "date_to", "with_image_only", "min_importance")

# Bucket key for artwork-column predicates inside a filter dict.
_COLUMNS_KEY = "__columns__"


def collect_filters(args) -> dict:
    """Return a filter dict with both vocab-mapped and artwork-column entries.

    Vocab-mapped entries are {field: value} strings (keys in _VOCAB_FIELDS).
    Artwork-column entries are stored under _COLUMNS_KEY as a predicate dict
    passed straight to filter_by_artwork_column().
    """
    filters: dict = {}
    # Vocabulary-mapped
    for field in _VOCAB_FIELDS:
        value = getattr(args, field, None)
        if value:
            filters[field] = value
    # Artwork-column predicates — only include if set
    columns: dict = {}
    for key in _COLUMN_FILTERS:
        value = getattr(args, key, None)
        if value:  # truthy covers store_true flags and non-None ints
            columns[key] = value
    if columns:
        filters[_COLUMNS_KEY] = columns
    return filters


def apply_filters(filters: dict) -> set[int] | None:
    """Intersect filter results. Returns None if no filters are active."""
    if not filters:
        return None
    allowed: set[int] | None = None
    # Vocab-mapped first
    for field, value in filters.items():
        if field == _COLUMNS_KEY:
            continue
        print(f"Filtering by {field}={value!r}...")
        ids = filter_by_field(field, value)
        print(f"  {len(ids):,} artworks match {field}={value!r}")
        allowed = ids if allowed is None else allowed & ids
    # Then artwork-column predicates in one query
    if _COLUMNS_KEY in filters:
        cols = filters[_COLUMNS_KEY]
        pretty = ", ".join(f"{k}={v!r}" for k, v in cols.items())
        print(f"Filtering by artwork columns: {pretty}")
        ids = filter_by_artwork_column(cols)
        print(f"  {len(ids):,} artworks match {pretty}")
        allowed = ids if allowed is None else allowed & ids
    if len(filters) > 1:
        print(f"  {len(allowed):,} artworks match all filters")
    return allowed


def filter_suffix(filters: dict) -> str:
    """Build a human-readable suffix like ' (paintings, subject: dog, 1600–1700)'."""
    if not filters:
        return ""
    parts = []
    for field, value in filters.items():
        if field == _COLUMNS_KEY:
            cols = value
            if cols.get("date_from") or cols.get("date_to"):
                lo = cols.get("date_from", "")
                hi = cols.get("date_to", "")
                parts.append(f"{lo}–{hi}")
            if cols.get("with_image_only"):
                parts.append("with image")
            if cols.get("min_importance"):
                parts.append(f"importance≥{cols['min_importance']}")
        elif field == "type":
            parts.append(f"{value}s")
        else:
            parts.append(f"{field.replace('_', ' ')}: {value}")
    return f" ({', '.join(parts)})"


def _slugify(text: str) -> str:
    """Turn a free-text value into a filename-safe slug."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def default_output(method: str, filters: dict) -> str:
    """Build a descriptive default filename from the method name and active filters.

    Examples:
        umap, {}                                         -> umap-explorer.html
        umap, {type: painting}                           -> umap-paintings.html
        pacmap, {type: painting, creator: 'Rembrandt'}   -> pacmap-rembrandt-paintings.html
        tsne, {subject: dog}                             -> tsne-dog.html
        umap, {production_place: Amsterdam}              -> umap-amsterdam.html
        umap, {__columns__: {date_from: 1600, date_to: 1700}} -> umap-1600-1700.html
        umap, {__columns__: {with_image_only: True}}     -> umap-with-image.html
    """
    if not filters:
        return f"{method}-explorer.html"
    parts = []
    # creator and subject before type so "rembrandt-paintings" reads naturally
    for field in ("creator", "subject", "production_place", "type"):
        if field not in filters:
            continue
        value = filters[field]
        if field == "type":
            parts.append(_slugify(value) + "s")
        else:
            parts.append(_slugify(value))
    cols = filters.get(_COLUMNS_KEY) or {}
    if cols.get("date_from") or cols.get("date_to"):
        lo = cols.get("date_from") or ""
        hi = cols.get("date_to") or ""
        parts.append(f"{lo}-{hi}".strip("-"))
    if cols.get("with_image_only"):
        parts.append("with-image")
    if cols.get("min_importance"):
        parts.append(f"importance-{cols['min_importance']}")
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
