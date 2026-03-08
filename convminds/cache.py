from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def convminds_home() -> Path:
    home = os.environ.get("CONVMINDS_HOME", "~/.convminds")
    return Path(home).expanduser()


def ensure_cache_dir(namespace: str) -> Path:
    path = convminds_home() / namespace
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        fallback = Path.cwd() / ".convminds" / namespace
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_hash(payload: dict[str, Any]) -> str:
    normalized = _normalize(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_normalize(payload), indent=2, sort_keys=True))


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def cache_paths(namespace: str, config: dict[str, Any], suffix: str = ".pkl") -> tuple[Path, Path]:
    namespace_dir = ensure_cache_dir(namespace)
    cache_id = config_hash(config)
    payload_path = namespace_dir / f"{cache_id}{suffix}"
    meta_path = namespace_dir / f"{cache_id}.json"
    return payload_path, meta_path


def load_cache(namespace: str, config: dict[str, Any], suffix: str = ".pkl") -> Any | None:
    payload_path, _ = cache_paths(namespace, config=config, suffix=suffix)
    if not payload_path.exists():
        return None
    return read_pickle(payload_path)


def save_cache(namespace: str, config: dict[str, Any], payload: Any, suffix: str = ".pkl") -> Path:
    payload_path, meta_path = cache_paths(namespace, config=config, suffix=suffix)
    write_pickle(payload_path, payload)
    write_json(meta_path, {"config": _normalize(config), "payload_path": str(payload_path)})
    return payload_path


def save_score_summary(config: dict[str, Any], summary: dict[str, Any], diagnostics: dict[str, Any] | None = None) -> Path:
    score_dir = ensure_cache_dir("scores")
    cache_id = config_hash(config)
    summary_path = score_dir / f"{cache_id}.json"
    write_json(summary_path, {"config": _normalize(config), "summary": _normalize(summary)})
    if diagnostics is not None:
        diagnostics_path = score_dir / f"{cache_id}.pkl"
        write_pickle(diagnostics_path, diagnostics)
    return summary_path


def _iter_score_manifests() -> Iterable[dict[str, Any]]:
    score_dir = ensure_cache_dir("scores")
    for path in sorted(score_dir.glob("*.json")):
        yield read_json(path)


def display_score(**filters: Any) -> dict[str, Any] | None:
    filters = {key: value for key, value in filters.items() if value is not None}
    matches = []
    for manifest in _iter_score_manifests():
        config = manifest.get("config", {})
        if all(config.get(key) == value for key, value in filters.items()):
            matches.append(manifest)

    if not matches:
        print("No cached score matches the provided filters.")
        return None

    if len(matches) > 1:
        print(f"Found {len(matches)} cached scores. Showing the most recent manifest order match.")

    selected = matches[-1]
    summary = selected.get("summary", {})
    config = selected.get("config", {})
    print("Cached score")
    for key in sorted(config):
        print(f"  {key}: {config[key]}")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    return selected
