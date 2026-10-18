import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from engine.losses import AsymmetricLoss, MaskedBCEWithLogitsLoss, estimate_pos_weight
from engine.trainer import fit
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths
from utils.logger import create_logger
from utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train the DenseNet-121 Study-level route")
    parser.add_argument("--data-config", default=str(PROJECT_DIR / "configs/data.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_DIR / "configs/train.yaml"))
    parser.add_argument("--model-config", default=str(PROJECT_DIR / "configs/model.yaml"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None,
                        help="Load model weights only, then start a new optimizer/scheduler run")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    data_path, train_path, model_path = map(resolve_config_path,
        (args.data_config, args.train_config, args.model_config))
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    train_cfg, model_cfg = read_yaml(train_path), read_yaml(model_path)
    output_dir = Path(train_cfg.get("output_dir", "."))
    train_cfg["output_dir"] = str(output_dir if output_dir.is_absolute()
                                  else (PROJECT_DIR / output_dir).resolve())

    semantics = data_cfg.get("label_semantics", "unresolved")
    if semantics != "closed_world":
        raise RuntimeError(
            "Training requires label_semantics: closed_world because the confirmed train/validation "
            "rule is that classes omitted from Predict_class are negative. Update configs/data.yaml."
        )
    for key in ("image_root", "train_csv", "val_csv"):
        if not data_cfg.get(key) or not Path(data_cfg[key]).exists():
            raise FileNotFoundError(f"Configured {key} does not exist: {data_cfg.get(key)}")
    class_ids = data_cfg.get("class_ids", list(range(1, 11)))
    class_names = data_cfg.get("class_names", [])
    if class_ids != list(range(1, 11)) or len(class_names) != 10:
        raise ValueError("configs/data.yaml must define the official ordered 10 class IDs and names")

    seed_everything(int(train_cfg.get("seed", 42)))
    dataset_columns = dict(image_id_column=data_cfg["image_id_column"],
                          subject_column=data_cfg["subject_column"],
                          study_column=data_cfg["study_column"])
    train_data = StudyDataset(data_cfg["train_csv"], data_cfg["image_root"],
        transform=build_transform(data_cfg["image_size"], train=True),
        label_column=data_cfg["label_column"], label_semantics=semantics, **dataset_columns)
    val_data = StudyDataset(data_cfg["val_csv"], data_cfg["image_root"],
        transform=build_transform(data_cfg["image_size"]),
        label_column=data_cfg["label_column"], label_semantics=semantics, **dataset_columns)
    targets, known = train_data.label_tensors()
    _, val_known = val_data.label_tensors()
    for split_name, known_mask in (("train", known), ("validation", val_known)):
        incomplete = int((~known_mask.all(dim=1)).sum())
        if incomplete:
            raise ValueError(
                f"{split_name} contains {incomplete} Study(s) without a complete 10-class label; "
                "closed-world mode only resolves omitted IDs in a non-empty Predict_class value."
            )
    train_subjects = {subject_id for subject_id, _ in train_data.study_keys}
    val_subjects = {subject_id for subject_id, _ in val_data.study_keys}
    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(f"Subject leakage across train/validation splits: {len(overlap)} Subjects")
    train_sampler = StudyBatchSampler(train_data, train_cfg["image_budget"], shuffle=True,
                                       seed=train_cfg.get("seed", 42))
    val_sampler = StudyBatchSampler(val_data, train_cfg["image_budget"], shuffle=False)
    loader_args = dict(num_workers=int(data_cfg["num_workers"]),
                       pin_memory=torch.cuda.is_available(), collate_fn=collate_studies)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_data, batch_sampler=val_sampler, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_checkpoint = (resolve_config_path(args.init_checkpoint)
                       if args.init_checkpoint else None)
    model_kwargs = {key: value for key, value in model_cfg.items() if key != "name"}
    if init_checkpoint:
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_checkpoint}")
        # The checkpoint already contains the complete backbone, so do not fetch ImageNet
        # weights before loading it. A strict load below rejects architectural mismatches.
        model_kwargs["pretrained"] = False
    model = ChestXrayClassifier(**model_kwargs).to(device)
    init_state = None
    if init_checkpoint:
        init_state = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(init_state, dict) or "model" not in init_state:
            raise ValueError(f"Expected a project checkpoint with a 'model' state: {init_checkpoint}")
        source_config = init_state.get("config") or {}
        source_model_cfg = source_config.get("model_config")
        if not isinstance(source_model_cfg, dict):
            raise ValueError("Initialization checkpoint must include config.model_config so the "
                             "architecture and Study fusion can be verified")
        model_fields = ("name", "num_labels", "head", "cbam", "study_fusion",
                        "label_d_model", "label_heads", "label_dropout")
        model_mismatches = {
            key: (source_model_cfg.get(key), model_cfg.get(key))
            for key in model_fields
            if source_model_cfg.get(key) != model_cfg.get(key)
        }
        source_data_cfg = source_config.get("data") or {}
        if source_data_cfg.get("image_size") not in (None, data_cfg["image_size"]):
            model_mismatches["image_size"] = (source_data_cfg.get("image_size"),
                                               data_cfg["image_size"])
        if source_config.get("label_semantics") not in (None, semantics):
            model_mismatches["label_semantics"] = (source_config.get("label_semantics"), semantics)
        if model_mismatches:
            raise ValueError(f"Initialization checkpoint/config mismatch: {model_mismatches}")
        model.load_state_dict(init_state["model"], strict=True)
        del init_state
    groups = [
        {"params": model.backbone.parameters(), "lr": train_cfg["learning_rate_backbone"]},
        {"params": (model.classifier.parameters() if model.classifier is not None
                    else model.vector_classifier.parameters()), "lr": train_cfg["learning_rate_head"]},
    ]
    if model.study_attention is not None:
        groups.append({"params": model.study_attention.parameters(),
                       "lr": train_cfg["learning_rate_head"]})
    if model.cbam.__class__.__name__ == "CBAM":
        groups.append({"params": model.cbam.parameters(), "lr": train_cfg["learning_rate_head"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])

    def schedule(epoch):
        warmup = int(train_cfg["warmup_epochs"])
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(int(train_cfg["epochs"]) - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loss_name = train_cfg.get("loss", "bce").lower()
    weight_counts = None
    if loss_name == "bce":
        criterion = MaskedBCEWithLogitsLoss()
    elif loss_name == "weighted_bce":
        strategy = train_cfg.get("weighting", "sqrt")
        weights = estimate_pos_weight(targets, known, strategy=strategy,
                                      cap=train_cfg.get("pos_weight_cap", 5.0))
        weight_counts = {
            "positive": ((targets == 1) & known).sum(0).tolist(),
            "negative": ((targets == 0) & known).sum(0).tolist(),
            "pos_weight": weights.tolist(),
            "strategy": strategy,
        }
        criterion = MaskedBCEWithLogitsLoss(weights)
    elif loss_name == "asl":
        criterion = AsymmetricLoss(train_cfg.get("gamma_pos", 0.0),
                                   train_cfg.get("gamma_neg", 4.0),
                                   train_cfg.get("asl_clip", 0.05))
    else:
        raise ValueError("loss must be bce, weighted_bce, or asl")
    criterion = criterion.to(device)
    logger = create_logger(Path(train_cfg["output_dir"]) / "outputs/logs/train.log")
    logger.info("device=%s imagenet_pretrained_init=%s fusion=%s loss=%s", device,
                bool(model_cfg.get("pretrained", False) and not init_checkpoint),
                model_cfg.get("study_fusion"), loss_name)
    if init_checkpoint:
        logger.info("initialized all model parameters strictly from %s; optimizer/scheduler start fresh",
                    init_checkpoint)
    if weight_counts is not None:
        logger.info("training Study class counts and pos_weight=%s", weight_counts)
    if args.resume:
        train_cfg["resume"] = str(resolve_config_path(args.resume))
    saved_model_cfg = dict(model_cfg)
    if init_checkpoint:
        saved_model_cfg["pretrained"] = False
    run_config = {
        **train_cfg,
        "data": {**data_cfg, "train_csv": str(data_cfg["train_csv"]),
                 "val_csv": str(data_cfg["val_csv"]), "image_root": str(data_cfg["image_root"])},
        "model_config": saved_model_cfg,
        "class_ids": class_ids,
        "class_names": class_names,
        "label_semantics": semantics,
        "weight_counts": weight_counts,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "pretrained_weight_source": (
            "initialized from checkpoint" if init_checkpoint else
            ("torchvision ImageNet weights" if model_cfg.get("pretrained") else None)
        ),
        "study_fusion": model_cfg["study_fusion"],
    }
    fit(model, train_loader, val_loader, criterion, optimizer, scheduler, run_config,
        device, logger, class_names=class_names)


if __name__ == "__main__":
    main()
