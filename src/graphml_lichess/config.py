from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out  # lists are replaced, not concatenated


def _load_layered(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    return deep_merge(_load_layered(path.parent / base), cfg) if base else cfg


def load_config(path: Path, overrides: list[str] | None = None) -> dict:
    """Load a config with `base:` inheritance, then apply 'a.b.c=value' overrides."""
    cfg = _load_layered(path)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got: {item}")

        set_path(cfg, key, yaml.safe_load(raw))

    return cfg


def set_path(cfg: dict, dotted_key: str, value) -> None:
    *parents, leaf = dotted_key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value
