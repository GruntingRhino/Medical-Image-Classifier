"""Experiment ledger: append-only JSON records with provenance.

Each record stores: experiment name, config dict + hash, code revision (git
sha if available), seed(s), dataset checksums, metrics, and wall-time. Results
are written to ``results/<name>/<record>.json`` and indexed in
``results/ledger.jsonl`` for aggregation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results"


def code_revision() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return sha or None
    except Exception:
        return None


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_checksum(path: str | Path, max_bytes: int = 2**31) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def save_record(name: str, cfg: dict, metrics: dict, seed=None, artifacts: dict | None = None) -> Path:
    """Save one experiment record; returns the record path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "experiment": name,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "code_revision": code_revision(),
        "seed": seed,
        "artifacts": artifacts or {},
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{name}-{rec['config_hash']}{f'-s{seed}' if seed is not None else ''}.json"
    path = out_dir / fname
    path.write_text(json.dumps(rec, indent=2, default=str))
    with open(RESULTS_DIR / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"experiment": name, "path": str(path), "config_hash": rec["config_hash"], "seed": seed}) + "\n")
    return path


def load_records(name: str) -> list[dict]:
    out_dir = RESULTS_DIR / name
    if not out_dir.exists():
        return []
    recs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    return recs
