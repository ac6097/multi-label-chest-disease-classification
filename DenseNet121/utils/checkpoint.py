import json
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, epoch=0,
                    best_metric=None, config=None, labels=None, stale_epochs=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch, "best_metric": best_metric,
             "config": config, "labels": labels, "stale_epochs": stale_epochs}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    metadata = {"epoch": epoch, "best_macro_auroc": best_metric, "labels": labels,
                "config": config, "checkpoint": path.name}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state
