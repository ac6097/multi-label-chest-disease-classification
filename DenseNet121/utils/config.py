from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve_config_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    return candidate if candidate.exists() else PROJECT_DIR / path


def read_yaml(path):
    path = resolve_config_path(path)
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def resolve_data_paths(data_config, config_path):
    base = resolve_config_path(config_path).resolve().parent.parent
    output = dict(data_config)
    for key in ("image_root", "train_csv", "val_csv", "test_csv"):
        value = output.get(key)
        if value:
            path = Path(value)
            output[key] = path if path.is_absolute() else (base / path).resolve()
    return output
