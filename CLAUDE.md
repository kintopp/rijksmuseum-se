# Rijksmuseum Semantic Explorer

Exploratory project for visualizing the Rijksmuseum collection's 831K artwork embeddings
using MLX-accelerated dimensionality reduction on Apple Silicon.

## Project Structure

```
lib/embeddings.py    — Shared API: load_embeddings, load_metadata, filter_by_field, build_hover_text
notebooks/           — Jupyter notebooks (UMAP, PaCMAP) for exploratory work
scripts/             — CLI generators for UMAP/t-SNE/PaCMAP HTML explorers (primary output path)
mathematica/         — 5 Wolfram notebooks + export_for_mathematica.py data bridge
patches/             — Local patches for vendored deps (re-apply after every uv sync)
data/                — Symlinks to ../rijksmuseum-mcp-plus/data/{embeddings,vocabulary}.db
output/              — Generated HTML, coords (.gitignore'd)
```

## Data

- **embeddings.db**: ~833K artwork embeddings (multilingual-e5-small, 384 dims, int8 quantized). Also carries a `desc_embeddings` table (511K subset, clips/e5-small-trm-nl description model) and sqlite-vec HNSW virtual tables (`vec_artworks`, `vec_desc_artworks`) — currently unused by this project.
- **vocabulary.db**: Structured metadata. 14 `field_lookup` names: `type`, `creator`, `subject`, `material`, `technique`, `production_place`, `spatial`, `profession`, `collection_set`, `attribution_qualifier`, `birth_place`, `death_place`, `production_role`, `source_type` (first 5 surfaced through `load_metadata`; all 14 usable via `filter_by_field`).
- Both are symlinked from `../rijksmuseum-mcp-plus/data/`
- Both DBs carry a `version_info` table. `load_metadata` warns if it's missing (indicates a pre-v0.24 build) and hard-errors only if `field_lookup` itself is absent.

### New metadata surface (current upstream build: 2026-04-19+)

- `artworks` now has denormalized display columns: `creator_label`, `date_display`, `date_earliest`, `date_latest`, `has_image` (~88% of rows), `iiif_id`, `importance`, plus `description_text`, `credit_line`, `inscription_text`, `provenance_text`, dimensions, `current_location`. `load_metadata` surfaces the first seven; the rest are available via direct SQL.
- `vocabulary` now has per-term enrichments: `lat`/`lon` (geocoded places), `notation` (Iconclass), `wikidata_id`, `broader_id` (hierarchy), `birth_year`/`death_year`/`gender`/`bio` (persons). None are consumed yet — good material for future viz work.
- New tables not yet consumed: `vocabulary_external_ids` (Wikidata/VIAF/RKD/ULAN/TGN/AAT/Iconclass/GeoNames cross-refs), `person_names`, `exhibitions`, `related_objects`, `title_variants`, `rights_lookup`.

## Key Patterns

- Embeddings stored as int8 BLOBs in SQLite → decode with `np.frombuffer` (zero-copy, ~10× faster than `struct.unpack`) → float32 → L2-normalize
- Vocabulary DB uses integer-encoded schema (field_lookup + mappings + vocabulary tables)
- Batch queries in chunks of ~990 to stay within `SQLITE_LIMIT_VARIABLE_NUMBER`
- Reservoir-sample art_ids first, then batch-fetch BLOBs — avoid `ORDER BY RANDOM()` which materializes all 831K BLOBs into a temp sort buffer

## Commands

```bash
uv sync && bash patches/apply-patches.sh   # Install dependencies + apply local fixes
uv run jupyter lab                          # Launch notebooks

# Generate standalone HTML explorers (output/*.html)
uv run python scripts/generate-umap-explorer.py --sample 20000
uv run python scripts/generate-tsne-explorer.py --sample 20000
uv run python scripts/generate-pacmap-explorer.py --sample 20000

# Filters (combine with AND, auto-name output file)
#   Vocabulary:     --type, --creator, --subject, --production-place
#   Artwork column: --date-from YEAR, --date-to YEAR, --with-image-only, --min-importance N
#   HDBSCAN:        --min-cluster-size, --min-samples

# Mathematica: export data once, then open mathematica/*.nb in Wolfram 14+
uv run python mathematica/export_for_mathematica.py
```

## Patches

Local fixes to vendored dependencies live in `patches/`. Run `bash patches/apply-patches.sh`
after every `uv sync` to re-apply them. The script is idempotent (skips already-applied patches).

- **mlx-vis-pacmap-memory-fix.patch** — Fixes O(n) Metal memory accumulation in `_brute_knn` that
  caused 20+ GB RAM / 7 GB swap at 200K points. Switches to per-chunk numpy copy (umap pattern)
  and removes redundant numpy↔MLX round-trips in `fit_transform`.
