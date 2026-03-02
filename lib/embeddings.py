"""Shared utilities for loading Rijksmuseum artwork embeddings and metadata.

Embeddings are stored as int8-quantized BLOBs (384 dims) in SQLite.
Metadata lives in a separate vocabulary DB with integer-encoded schema.
"""

import html
import random
import sqlite3
from pathlib import Path

import numpy as np

# Default paths (symlinked from rijksmuseum-mcp-plus)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_EMBEDDINGS_DB = DATA_DIR / "embeddings.db"
DEFAULT_VOCAB_DB = DATA_DIR / "vocabulary.db"

# SQLite max host parameters — keep batches under this
_BATCH_SIZE = 990
_EMBEDDING_DIM = 384


def _decode_int8_blob(blob: bytes) -> np.ndarray:
    """Decode a raw int8 BLOB into a numpy array (zero-copy)."""
    if len(blob) != _EMBEDDING_DIM:
        raise ValueError(f"Expected {_EMBEDDING_DIM}-byte BLOB, got {len(blob)}")
    return np.frombuffer(blob, dtype=np.int8)


def load_embeddings(
    db_path: Path = DEFAULT_EMBEDDINGS_DB,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[list[int], list[str], np.ndarray]:
    """Load embeddings from SQLite, decode int8 → float32, L2-normalize.

    Args:
        db_path: Path to embeddings.db
        sample_size: If set, reservoir-sample this many embeddings. None = load all.
        seed: Random seed for sampling.

    Returns:
        (art_ids, object_numbers, embeddings) where embeddings is N×384 float32, L2-normalized.
    """
    with sqlite3.connect(str(db_path)) as db:
        # Get total count
        total = db.execute("SELECT COUNT(*) FROM artwork_embeddings").fetchone()[0]

        # Reservoir sampling: fetch IDs first, then retrieve selected BLOBs
        # (ORDER BY RANDOM() would materialize all 831K BLOBs into a temp sort buffer)
        if sample_size is not None and sample_size < total:
            random.seed(seed)
            all_ids = db.execute("SELECT art_id FROM artwork_embeddings").fetchall()
            sampled = random.sample(all_ids, sample_size)
            del all_ids
            sampled_ids = [r[0] for r in sampled]
        else:
            sampled_ids = None  # fetch all

        # Batch-fetch rows
        rows = []
        if sampled_ids is not None:
            for i in range(0, len(sampled_ids), _BATCH_SIZE):
                batch = sampled_ids[i : i + _BATCH_SIZE]
                ph = ",".join("?" * len(batch))
                rows.extend(
                    db.execute(
                        f"SELECT art_id, object_number, embedding FROM artwork_embeddings WHERE art_id IN ({ph})",
                        batch,
                    ).fetchall()
                )
        else:
            rows = db.execute(
                "SELECT art_id, object_number, embedding FROM artwork_embeddings"
            ).fetchall()

    art_ids = [r[0] for r in rows]
    object_numbers = [r[1] for r in rows]

    # Decode int8 BLOBs → float32 matrix (np.frombuffer is ~10x faster than struct.unpack)
    n = len(rows)
    embeddings = np.empty((n, _EMBEDDING_DIM), dtype=np.float32)
    for i, r in enumerate(rows):
        embeddings[i] = _decode_int8_blob(r[2])
    del rows

    # L2-normalize (each row to unit length)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings /= norms

    return art_ids, object_numbers, embeddings


def load_metadata(
    art_ids: list[int],
    object_numbers: list[str],
    vdb_path: Path = DEFAULT_VOCAB_DB,
) -> dict[int, dict]:
    """Fetch titles, creators, types, subjects, materials, techniques from vocabulary DB.

    Args:
        art_ids: List of artwork integer IDs.
        object_numbers: Corresponding object number strings.
        vdb_path: Path to vocabulary.db

    Returns:
        Dict keyed by art_id with keys: object_number, title, types, creators,
        subjects, materials, techniques (all lists of strings).
    """
    with sqlite3.connect(str(vdb_path)) as vdb:
        # Require integer-encoded schema (v0.13+)
        has_int = vdb.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='field_lookup'"
        ).fetchone()[0] > 0
        if not has_int:
            raise RuntimeError(
                "vocabulary.db uses old text schema; requires v0.13+ integer-encoded DB"
            )

        field_map = dict(vdb.execute("SELECT name, id FROM field_lookup").fetchall())

        type_fid = field_map.get("type")
        creator_fid = field_map.get("creator")
        subject_fid = field_map.get("subject")
        material_fid = field_map.get("material")
        technique_fid = field_map.get("technique")

        field_ids = [f for f in [type_fid, creator_fid, subject_fid, material_fid, technique_fid] if f is not None]

        # Initialize metadata dict
        meta = {}
        for aid, obj in zip(art_ids, object_numbers):
            meta[aid] = {
                "object_number": obj,
                "title": "",
                "types": [],
                "creators": [],
                "subjects": [],
                "materials": [],
                "techniques": [],
            }

        # Fetch titles
        for i in range(0, len(art_ids), _BATCH_SIZE):
            batch = art_ids[i : i + _BATCH_SIZE]
            ph = ",".join("?" * len(batch))
            for aid, title in vdb.execute(
                f"SELECT art_id, title FROM artworks WHERE art_id IN ({ph})", batch
            ):
                if aid in meta:
                    meta[aid]["title"] = title or ""

        # Fetch vocabulary metadata
        if field_ids:
            field_ph = ",".join("?" * len(field_ids))
            mbatch = _BATCH_SIZE - len(field_ids)
            assert mbatch > 0, f"Too many field_ids ({len(field_ids)}) for batch size {_BATCH_SIZE}"
            for i in range(0, len(art_ids), mbatch):
                batch = art_ids[i : i + mbatch]
                ph = ",".join("?" * len(batch))
                query = f"""
                    SELECT m.artwork_id, m.field_id, COALESCE(v.label_en, v.label_nl)
                    FROM mappings m
                    JOIN vocabulary v ON v.vocab_int_id = m.vocab_rowid
                    WHERE m.artwork_id IN ({ph})
                      AND m.field_id IN ({field_ph})
                """
                for aid, fid, label in vdb.execute(query, batch + field_ids):
                    if aid not in meta or label is None:
                        continue
                    if fid == type_fid:
                        meta[aid]["types"].append(label)
                    elif fid == creator_fid:
                        meta[aid]["creators"].append(label)
                    elif fid == subject_fid:
                        meta[aid]["subjects"].append(label)
                    elif fid == material_fid:
                        meta[aid]["materials"].append(label)
                    elif fid == technique_fid:
                        meta[aid]["techniques"].append(label)

    return meta


def build_hover_text(meta: dict[int, dict], art_ids: list[int]) -> list[str]:
    """Build HTML hover text strings for Plotly from metadata.

    Returns a list aligned with art_ids. Output contains raw HTML tags
    (<b>, <br>) for formatting; all user content is html.escape()'d.
    Safe for use in Plotly hovertemplate %{text} fields.
    """
    esc = html.escape
    texts = []
    for aid in art_ids:
        m = meta[aid]
        obj = esc(m["object_number"])
        title = esc(m["title"])
        lines = [f"<b>{title or obj}</b>"]
        if title:
            lines.append(f"Object: {obj}")
        if m["creators"]:
            lines.append(f"Creator: {esc(', '.join(m['creators'][:2]))}")
        if m["types"]:
            lines.append(f"Type: {esc(', '.join(m['types'][:2]))}")
        if m["subjects"]:
            lines.append(f"Subjects: {esc(', '.join(m['subjects'][:3]))}")
        if m["materials"]:
            lines.append(f"Material: {esc(', '.join(m['materials'][:2]))}")
        if m["techniques"]:
            lines.append(f"Technique: {esc(', '.join(m['techniques'][:2]))}")
        texts.append("<br>".join(lines))
    return texts
