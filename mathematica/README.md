# Mathematica Notebooks

Five Wolfram Mathematica notebooks analysing 831K Rijksmuseum artwork embeddings
(multilingual-e5-small, 384 dimensions) from complementary angles: geometry,
graphs, artists, time, and metadata.

## Setup

**1. Export data** (one-time, ~30s):

```bash
uv run python mathematica/export_for_mathematica.py
```

This creates `mathematica/data/` with:
- `embeddings.bin` — 20K×384 float32 matrix (~30 MB)
- `metadata.json` — art IDs, object numbers, titles, types, creators, subjects, materials, techniques, dates (~8 MB)

**2. Open** any `.nb` file in Wolfram Mathematica (14.0+) and evaluate all cells top-to-bottom.

## Notebooks

### 1. Embedding Space Geometry (`01-embedding-geometry.nb`)

Probes the intrinsic mathematical structure of the 384-dimensional space: PCA
eigenspectrum and cumulative explained variance, effective dimensionality via
participation ratio and Shannon entropy, pairwise cosine/euclidean distance
distributions, k-NN density analysis, interactive 3D PCA scatter coloured by
object type, and isotropy measurement.

### 2. Semantic Similarity Networks (`02-semantic-networks.nb`)

Builds a k-nearest-neighbor graph (k=8) from cosine similarities and analyses
its structure using `FindGraphCommunities` and `CommunityGraphPlot`. Includes
degree distribution, betweenness centrality to identify bridge artworks linking
distinct communities, per-community metadata composition (types and subjects),
and an inter-community similarity heatmap with a centroid network graph.

### 3. Creator Style Space (`03-creator-stylespace.nb`)

Constructs a derived embedding space where each point is an artist rather than
an artwork. Computes centroid embeddings for the top 100 creators, visualises
their pairwise similarity as a heatmap and labelled zoom, builds a `Dendrogram`
of artistic styles using cosine distance, measures intra-artist style diversity
vs. productivity, ranks the most/least similar creator pairs, and projects the
creator landscape onto a 2D PCA map (size = works, colour = diversity).

### 4. Temporal-Semantic Journeys (`04-temporal-journeys.nb`)

Combines embedding coordinates with date metadata to trace how artistic style
evolves through time. Computes century and 50-year-period centroids, measures
inter-century style drift, builds a cross-century similarity matrix, draws a
colour-coded trajectory through PCA space, detects anachronistic artworks
(semantically resembling a different era), and estimates style velocity at
25-year resolution.

### 5. Metadata Landscapes (`05-metadata-landscapes.nb`)

Maps relationships between metadata categories through the embedding lens.
Type and material frequency charts, type×material and material×technique
co-occurrence heatmaps (log-scaled), category centroid similarity matrices,
a separability analysis measuring how distinctly each object type clusters in
embedding space, per-type subject word clouds, and a type similarity network
graph with size/colour encoding.

## Options

Adjust sample size (default 20K, max ~831K):

```bash
uv run python mathematica/export_for_mathematica.py --sample-size 50000
```

Larger samples give more robust statistics but increase memory usage in
Mathematica. The network notebook (02) internally subsamples to 2K for graph
analysis regardless of export size.

## Data pipeline

```
embeddings.db ──┐
                ├── export_for_mathematica.py ──► data/embeddings.bin
vocabulary.db ──┘                               data/metadata.json
                                                       │
                                    ┌──────────────────┼──────────────────┐
                                    ▼                  ▼                  ▼
                              01-embedding-    02-semantic-    03–05 ...
                              geometry.nb      networks.nb
```

The Python export script reads int8-quantized BLOBs from `embeddings.db`,
decodes them to float32, L2-normalises, and writes raw binary. Metadata is
pulled from the integer-encoded vocabulary schema (v0.13+) and serialised
as JSON. Mathematica loads these via `BinaryReadList` and `Import["RawJSON"]`.
