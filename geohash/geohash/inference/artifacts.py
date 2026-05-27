"""Run artifact loading and model reconstruction."""

import json
from pathlib import Path
from typing import Any

import torch

from geohash.model import NextMagnitudeLSTM


def save_preprocess_artifact(
    run_dir: Path,
    stoi: dict[str, int],
    numeric_mean: list[float],
    numeric_std: list[float],
    numeric_cols: list[str],
    geohash_precision: int,
    encoding: str,
    input_mode: str,
    char_stoi: dict[str, int] | None = None,
) -> Path:
    """Save preprocessing artifact to run directory."""
    payload = {
        "stoi": stoi,
        "char_stoi": char_stoi,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "numeric_cols": numeric_cols,
        "geohash_precision": geohash_precision,
        "encoding": encoding,
        "input_mode": input_mode,
    }
    path = run_dir / "preprocess.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def save_model_config_artifact(run_dir: Path, model_config: dict[str, Any]) -> Path:
    """Save model architecture config."""
    path = run_dir / "model_config.json"
    with open(path, "w") as f:
        json.dump(model_config, f, indent=2)
    return path


def load_run_bundle(run_dir: Path) -> dict[str, Any]:
    """Load full run bundle from directory."""
    run_dir = Path(run_dir)
    bundle: dict[str, Any] = {"run_dir": run_dir}

    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            bundle["config"] = json.load(f)

    preprocess_path = run_dir / "preprocess.json"
    if preprocess_path.exists():
        with open(preprocess_path) as f:
            bundle["preprocess"] = json.load(f)

    model_config_path = run_dir / "model_config.json"
    if model_config_path.exists():
        with open(model_config_path) as f:
            bundle["model_config"] = json.load(f)

    model_path = run_dir / "model_final.pt"
    if model_path.exists():
        bundle["model_state_dict"] = torch.load(model_path, map_location="cpu", weights_only=True)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            bundle["metrics"] = json.load(f)

    return bundle


def build_model_from_bundle(bundle: dict[str, Any]) -> NextMagnitudeLSTM:
    """Reconstruct model from bundle model_config and preprocess."""
    mc = bundle["model_config"]
    pp = bundle.get("preprocess", {})
    encoding = mc.get("encoding", pp.get("encoding", "flat"))
    vocab_size = mc["vocab_size"]
    if encoding == "hierarchical":
        char_stoi = pp.get("char_stoi") or {}
        vocab_size = len(char_stoi) if char_stoi else 34

    char_stoi = pp.get("char_stoi") or {}
    char_vocab_size = len(char_stoi) if char_stoi else 34

    return NextMagnitudeLSTM(
        vocab_size=vocab_size,
        embedding_dim=mc["embedding_dim"],
        num_numeric=mc["num_numeric"],
        hidden_size=mc["hidden_size"],
        num_layers=mc.get("num_layers", 1),
        dropout=mc.get("dropout", 0.0),
        input_mode=mc.get("input_mode", "full"),
        encoding=encoding,
        geohash_precision=mc.get("geohash_precision", pp.get("geohash_precision", 4)),
        char_vocab_size=char_vocab_size,
    )
