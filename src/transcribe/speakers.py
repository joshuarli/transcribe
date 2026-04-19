"""Speaker embedding storage and identification across episodes."""

import json
from pathlib import Path
from typing import cast

EMBEDDINGS_PATH = Path("cache/speakers.json")
SIMILARITY_THRESHOLD = 0.75


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    return cast("dict[str, list[float]]", json.loads(path.read_text()))


def save_embeddings(new: dict[str, list[float]], path: Path = EMBEDDINGS_PATH) -> None:
    """Merge new embeddings into the store (overwrites same-name entries)."""
    path.parent.mkdir(exist_ok=True)
    existing = load_embeddings(path)
    existing.update(new)
    path.write_text(json.dumps(existing))


def match_speakers(
    cluster_embeddings: dict[str, list[float]],
    known: dict[str, list[float]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """Map SPEAKER_XX cluster labels to known names by cosine similarity.

    Clusters whose best match falls below threshold keep their original label.
    """
    mapping = {}
    for cluster, emb in cluster_embeddings.items():
        best_name, best_sim = None, 0.0
        for name, known_emb in known.items():
            sim = _cosine(emb, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        mapping[cluster] = best_name if (best_sim >= threshold and best_name is not None) else cluster
    return mapping


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
