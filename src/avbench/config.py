"""YAML run-config support.

Scripts accept `--config path.yaml`. Precedence is CLI flag > config file >
built-in default, implemented by giving config-backed argparse options a default
of None and filling the gaps from the YAML afterwards.
"""

from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml  # lazy: only needed when --config is used

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config {} must be a YAML mapping, got {}".format(path, type(data).__name__))
    return data


def apply_config(args, cfg: Dict[str, Any], defaults: Dict[str, Any]):
    """Fill args attributes left as None from cfg, then from defaults.

    Mutates and returns `args`. A flag the user actually passed is non-None and
    therefore wins over the config; a key absent from both falls to `defaults`.
    """
    for name, default in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, cfg.get(name, default))
    return args
