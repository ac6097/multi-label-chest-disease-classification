import argparse

import torch
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from engine.losses import AsymmetricLoss, MaskedBCEWithLogitsLoss, estimate_pos_weight
from engine.validator import validate
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate DenseNet-121 at Study level")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    args = parser.parse_args()
    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    model_path = resolve_config_path(PROJECT_DIR / "configs/model.yaml")
    train_path = resolve_config_path(PROJECT_DIR / "configs/train.yaml")
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    model_cfg, train_cfg = read_yaml(model_path), read_yaml(train_path)
    checkpoint_path = resolve_config_path(args.checkpoint)
    if data_cfg.get("label_semantics") != "closed_world":
        raise RuntimeError("Validation requires the confirmed closed-world Predict_class semantics in configs/data.yaml")
    csv_file = args.csv or data_cfg.get("val_csv")
    if not csv_file:
        raise ValueError("Set val_csv in configs/data.yaml or pass --csv")
    dataset = StudyDataset(csv_file, data_cfg["image_root"], build_transform(data_cfg["image_size"]),
        data_cfg["label_column"], data_cfg["label_semantics"],
        data_cfg["image_id_column"], data_cfg["subject_column"], data_cfg["study_column"])
    _, known = dataset.label_tensors()
    if not bool(known.all()):
        raise ValueError("Validation contains a Study without a complete 10-class label")
    loader = DataLoader(dataset,
        batch_sampler=StudyBatchSampler(dataset, train_cfg["image_budget"], shuffle=False),
        num_workers=data_cfg["num_workers"], pin_memory=torch.cuda.is_available(),
        collate_fn=collate_studies)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_model_cfg = dict(state.get("config", {}).get("model_config", model_cfg))
    saved_model_cfg["pretrained"] = False  # Checkpoint already contains its backbone weights.
    model = ChestXrayClassifier(**{key: value for key, value in saved_model_cfg.items()
                                   if key != "name"}).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    loss_name = state.get("config", {}).get("loss", "bce")
    if loss_name == "weighted_bce":
        weight_cfg = state.get("config", {}).get("weight_counts") or {}
        pos_weight = torch.tensor(weight_cfg.get("pos_weight", [1.0] * 10), dtype=torch.float32)
        criterion = MaskedBCEWithLogitsLoss(pos_weight)
    elif loss_name == "asl":
        saved_cfg = state.get("config", {})
        criterion = AsymmetricLoss(saved_cfg.get("gamma_pos", 0.0), saved_cfg.get("gamma_neg", 4.0),
                                   saved_cfg.get("asl_clip", 0.05))
    else:
        criterion = MaskedBCEWithLogitsLoss()
    result = validate(model, loader, criterion.to(device), device)
    print({"loss": result["loss"], "macro_auroc": result["macro_auroc"],
           "macro_f1": result["macro_f1"], "per_label": result["per_label"]})


if __name__ == "__main__":
    main()
