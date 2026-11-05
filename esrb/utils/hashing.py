import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def hash_artifact(path: Path) -> str:
    """SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_object(obj: Any) -> str:
    """SHA-256 of an in-memory Python object via pickle."""
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def hash_dict(d: dict) -> str:
    """SHA-256 of a JSON-serialisable dict (sorted keys for determinism)."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
