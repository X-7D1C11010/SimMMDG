from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models

from dataset_tada_rgb_ir import (
    TADARgbIrDataset,
    build_numeric_class_mapping,
    build_tada_transforms,
    collect_domain_summaries,
)
from losses import SupConLoss


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


class Encoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feat):
        return self.net(feat)


class EncoderTrans(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feat):
        return self.net(feat)


class ProjectHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feat):
        return F.normalize(self.head(feat), dim=1)


class FeatureHead(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embedding_dim * 2),
        )

    def forward(self, feat):
        return self.net(feat)


def build_resnet_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if not hasattr(models, name):
        raise ValueError("Unknown torchvision model '{}'. Use a ResNet backbone.".format(name))
    constructor = getattr(models, name)
    try:
        backbone = constructor(pretrained=pretrained)
    except TypeError:
        backbone = constructor(weights=None)

    if not hasattr(backbone, "fc"):
        raise ValueError("Backbone '{}' is not supported because it has no fc layer.".format(name))
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class SimMMDGImageModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        trans_hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rgb_backbone, rgb_dim = build_resnet_backbone(backbone_name, pretrained)
        self.ir_backbone, ir_dim = build_resnet_backbone(backbone_name, pretrained)

        self.rgb_head = FeatureHead(rgb_dim, embedding_dim, hidden_dim, dropout)
        self.ir_head = FeatureHead(ir_dim, embedding_dim, hidden_dim, dropout)
        fused_dim = embedding_dim * 4
        self.classifier = Encoder(fused_dim, num_classes, hidden=hidden_dim, dropout=dropout)

        modality_dim = embedding_dim * 2
        self.rgb_to_ir = EncoderTrans(modality_dim, modality_dim, hidden=trans_hidden_dim, dropout=dropout)
        self.ir_to_rgb = EncoderTrans(modality_dim, modality_dim, hidden=trans_hidden_dim, dropout=dropout)

        self.rgb_proj = ProjectHead(embedding_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.ir_proj = ProjectHead(embedding_dim, hidden_dim=hidden_dim, out_dim=projection_dim)

    def forward(self, rgb, ir):
        rgb_feat = self.rgb_backbone(rgb)
        ir_feat = self.ir_backbone(ir)
        rgb_emd = self.rgb_head(rgb_feat)
        ir_emd = self.ir_head(ir_feat)
        logits = self.classifier(torch.cat([rgb_emd, ir_emd], dim=1))
        return {
            "logits": logits,
            "rgb_emd": rgb_emd,
            "ir_emd": ir_emd,
            "rgb_shared": rgb_emd[:, : self.embedding_dim],
            "rgb_private": rgb_emd[:, self.embedding_dim :],
            "ir_shared": ir_emd[:, : self.embedding_dim],
            "ir_private": ir_emd[:, self.embedding_dim :],
        }


def simmmdg_loss(model, outputs, labels, ce_loss, contrast_loss, args):
    cls_loss = ce_loss(outputs["logits"], labels)

    rgb_emd = outputs["rgb_emd"]
    ir_emd = outputs["ir_emd"]
    rgb_to_ir = F.normalize(model.rgb_to_ir(rgb_emd), dim=1)
    ir_to_rgb = F.normalize(model.ir_to_rgb(ir_emd), dim=1)
    trans_loss = 0.5 * (
        torch.mean(torch.norm(rgb_to_ir - F.normalize(ir_emd, dim=1), dim=1))
        + torch.mean(torch.norm(ir_to_rgb - F.normalize(rgb_emd, dim=1), dim=1))
    )

    rgb_proj = model.rgb_proj(outputs["rgb_shared"])
    ir_proj = model.ir_proj(outputs["ir_shared"])
    contrast_features = torch.stack([rgb_proj, ir_proj], dim=1)
    con_loss = contrast_loss(contrast_features, labels)

    if args.split_loss == "negative_mse":
        split_loss = -0.5 * (
            F.mse_loss(outputs["rgb_shared"], outputs["rgb_private"])
            + F.mse_loss(outputs["ir_shared"], outputs["ir_private"])
        )
    else:
        rgb_shared = F.normalize(outputs["rgb_shared"], dim=1)
        rgb_private = F.normalize(outputs["rgb_private"], dim=1)
        ir_shared = F.normalize(outputs["ir_shared"], dim=1)
        ir_private = F.normalize(outputs["ir_private"], dim=1)
        split_loss = 0.5 * (
            torch.sum(rgb_shared * rgb_private, dim=1).pow(2).mean()
            + torch.sum(ir_shared * ir_private, dim=1).pow(2).mean()
        )

    total = (
        cls_loss
        + args.alpha_trans * trans_loss
        + args.alpha_contrast * con_loss
        + args.explore_loss_coeff * split_loss
    )
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_cls": float(cls_loss.detach().cpu()),
        "loss_trans": float(trans_loss.detach().cpu()),
        "loss_contrast": float(con_loss.detach().cpu()),
        "loss_split": float(split_loss.detach().cpu()),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            confusion[target, pred] += 1

    acc = float(np.trace(confusion) / max(1, confusion.sum()))
    active_labels = [
        idx
        for idx in range(num_classes)
        if confusion[idx, :].sum() > 0 or confusion[:, idx].sum() > 0
    ]
    if not active_labels:
        return {"acc": acc, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precisions = []
    recalls = []
    f1s = []
    for idx in active_labels:
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "acc": acc,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


def train_one_epoch(model, loader, optimizer, ce_loss, contrast_loss, device, args):
    model.train()
    running_loss = 0.0
    total = 0
    component_sums: Dict[str, float] = {}
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(rgb, ir)
        loss, components = simmmdg_loss(model, outputs, labels, ce_loss, contrast_loss, args)
        loss.backward()
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
        total += batch_size
        preds = outputs["logits"].argmax(dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, args.num_classes)
    metrics["loss"] = running_loss / max(1, total)
    for key, value in component_sums.items():
        metrics[key] = value / max(1, total)
    return metrics


@torch.no_grad()
def evaluate(model, loader, ce_loss, device, args):
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(rgb, ir)
        loss = ce_loss(outputs["logits"], labels)
        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        y_pred.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, args.num_classes)
    metrics["loss"] = running_loss / max(1, total)
    return metrics


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def setup_logger(log_path: Path):
    logger = logging.getLogger("tada_simmmdg")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def metric_row(target_domain, repeat_idx, seed, epoch, split, metrics):
    return {
        "target_domain": target_domain,
        "repeat": repeat_idx,
        "seed": seed,
        "epoch": epoch,
        "split": split,
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "loss_cls": metrics.get("loss_cls", ""),
        "loss_trans": metrics.get("loss_trans", ""),
        "loss_contrast": metrics.get("loss_contrast", ""),
        "loss_split": metrics.get("loss_split", ""),
    }


def update_best_metrics(best_metrics: Dict[str, Dict], epoch: int, metrics: Dict[str, float]):
    for key in ("acc", "precision", "recall", "f1"):
        if metrics[key] > best_metrics[key]["value"]:
            best_metrics[key] = {"value": metrics[key], "epoch": epoch}


def run_repeat(args, target_domain: str, repeat_idx: int, seed: int, exp_dir: Path):
    set_seed(seed)
    run_dir = exp_dir / target_domain / "repeat_{:02d}_seed_{}".format(repeat_idx, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")

    train_tf = build_tada_transforms(args.image_size, train=True)
    eval_tf = build_tada_transforms(args.image_size, train=False)
    class_to_idx = build_numeric_class_mapping(args.num_classes)

    source_train = TADARgbIrDataset(
        args.data_root,
        args.source_domain,
        args.train_split,
        transform_rgb=train_tf,
        transform_ir=train_tf,
        pairing=args.pairing,
        class_to_idx=class_to_idx,
    )
    source_val = TADARgbIrDataset(
        args.data_root,
        args.source_domain,
        args.val_split,
        transform_rgb=eval_tf,
        transform_ir=eval_tf,
        pairing=args.pairing,
        class_to_idx=class_to_idx,
    )
    target_test = TADARgbIrDataset(
        args.data_root,
        target_domain,
        args.test_split,
        transform_rgb=eval_tf,
        transform_ir=eval_tf,
        pairing=args.pairing,
        class_to_idx=class_to_idx,
    )

    train_datasets = [source_train]
    dataset_summary = {
        "source_train": source_train.summary,
        "source_val": source_val.summary,
        "target_test": target_test.summary,
    }

    if args.include_target_train:
        target_train = TADARgbIrDataset(
            args.data_root,
            target_domain,
            args.train_split,
            transform_rgb=train_tf,
            transform_ir=train_tf,
            pairing=args.pairing,
            class_to_idx=class_to_idx,
        )
        train_datasets.append(target_train)
        dataset_summary["target_train"] = target_train.summary

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    write_json(run_dir / "dataset_summary.json", dataset_summary)

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=len(train_dataset) >= args.batch_size,
    )
    val_loader = make_loader(source_val, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader = make_loader(target_test, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SimMMDGImageModel(
        num_classes=args.num_classes,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        trans_hidden_dim=args.trans_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    contrast_loss = SupConLoss(temperature=args.temp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = {
        "acc": {"value": -1.0, "epoch": -1},
        "precision": {"value": -1.0, "epoch": -1},
        "recall": {"value": -1.0, "epoch": -1},
        "f1": {"value": -1.0, "epoch": -1},
    }
    best_monitor = {"value": -1.0, "epoch": -1, "metrics": None}

    epoch_csv = run_dir / "epoch_metrics.csv"
    with epoch_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_domain",
                "repeat",
                "seed",
                "epoch",
                "split",
                "loss",
                "acc",
                "precision",
                "recall",
                "f1",
                "loss_cls",
                "loss_trans",
                "loss_contrast",
                "loss_split",
            ],
        )
        writer.writeheader()

        logger.info(
            "Start target=%s repeat=%s seed=%s train_samples=%s target_test_samples=%s",
            target_domain,
            repeat_idx,
            seed,
            len(train_dataset),
            len(target_test),
        )

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, ce_loss, contrast_loss, device, args)
            val_metrics = evaluate(model, val_loader, ce_loss, device, args)
            test_metrics = evaluate(model, test_loader, ce_loss, device, args)

            for split, metrics in (
                ("train", train_metrics),
                ("source_val", val_metrics),
                ("target_test", test_metrics),
            ):
                writer.writerow(metric_row(target_domain, repeat_idx, seed, epoch, split, metrics))
            f.flush()

            update_best_metrics(best_metrics, epoch, test_metrics)
            if test_metrics[args.best_metric] > best_monitor["value"]:
                best_monitor = {
                    "value": test_metrics[args.best_metric],
                    "epoch": epoch,
                    "metrics": dict(test_metrics),
                }
                if args.save_best:
                    torch.save(
                        {
                            "epoch": epoch,
                            "target_domain": target_domain,
                            "repeat": repeat_idx,
                            "seed": seed,
                            "args": vars(args),
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "test_metrics": test_metrics,
                        },
                        run_dir / "best_model.pt",
                    )

            logger.info(
                "epoch=%03d train_acc=%.4f source_val_acc=%.4f target_acc=%.4f "
                "target_p=%.4f target_r=%.4f target_f1=%.4f best_%s=%.4f@%s",
                epoch,
                train_metrics["acc"],
                val_metrics["acc"],
                test_metrics["acc"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
                args.best_metric,
                best_monitor["value"],
                best_monitor["epoch"],
            )

    summary = {
        "target_domain": target_domain,
        "repeat": repeat_idx,
        "seed": seed,
        "best_epoch_by_{}".format(args.best_metric): best_monitor["epoch"],
        "best_{}_value".format(args.best_metric): best_monitor["value"],
        "best_epoch_metrics": best_monitor["metrics"],
    }
    for metric, payload in best_metrics.items():
        summary["best_{}".format(metric)] = payload["value"]
        summary["best_{}_epoch".format(metric)] = payload["epoch"]

    write_json(run_dir / "repeat_summary.json", summary)
    logger.info("Finished target=%s repeat=%s summary=%s", target_domain, repeat_idx, summary)
    return summary


def write_summary_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict], target_domains: Iterable[str]) -> List[Dict]:
    aggregate_rows = []
    for target in target_domains:
        target_rows = [r for r in rows if r["target_domain"] == target]
        if not target_rows:
            continue
        out = {"target_domain": target, "repeats": len(target_rows)}
        for metric in ("acc", "precision", "recall", "f1"):
            values = [float(r["best_{}".format(metric)]) for r in target_rows]
            out["{}_mean".format(metric)] = statistics.mean(values)
            out["{}_std".format(metric)] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(out)
    return aggregate_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run SimMMDG-style RGB-IR experiments on TADA weather domains.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/lixiang/lx/Data",
        help="Root directory of the TADA dataset. It can also be set by TADA_DATA_ROOT.",
    )
    parser.add_argument("--source_domain", type=str, default="晴天")
    parser.add_argument("--target_domains", nargs="+", default=["黑天", "逆光", "雾天", "雨天"])
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="val")
    parser.add_argument("--include_target_train", action="store_true")

    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pairing", choices=["cycle", "intersection", "strict"], default="cycle")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--trans_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_trans", type=float, default=0.1)
    parser.add_argument("--alpha_contrast", type=float, default=3.0)
    parser.add_argument("--explore_loss_coeff", type=float, default=0.7)
    parser.add_argument(
        "--split_loss",
        choices=["orthogonal", "negative_mse"],
        default="orthogonal",
        help="Feature split regularizer. orthogonal is bounded and stable; negative_mse is the original unbounded form.",
    )
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--best_metric", choices=["acc", "precision", "recall", "f1"], default="f1")
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/tada_simmmdg",
        help="Directory for logs and results. Relative paths are resolved under the SimMMDG repo root.",
    )
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.environ.get("TADA_DATA_ROOT")
    if not args.data_root:
        parser.error("Please set --data_root or the TADA_DATA_ROOT environment variable.")

    args.data_root = str(resolve_repo_path(args.data_root))
    return args


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or "weather_rgb_ir_{}".format(timestamp)
    exp_dir = resolve_repo_path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(exp_dir / "config.json", vars(args))

    all_domains = [args.source_domain] + list(args.target_domains)
    summaries = collect_domain_summaries(
        args.data_root,
        domains=all_domains,
        splits=[args.train_split, args.val_split],
        pairing=args.pairing,
        num_classes=args.num_classes,
    )
    write_json(exp_dir / "dataset_preflight_summary.json", summaries)

    if args.dry_run:
        print("Dry run complete. Dataset summary written to {}".format(exp_dir / "dataset_preflight_summary.json"))
        return

    if args.seeds is not None and len(args.seeds) > 0:
        if len(args.seeds) < args.repeats:
            raise ValueError("--seeds must contain at least --repeats values.")
        seeds = args.seeds[: args.repeats]
    else:
        seeds = [args.seed + i for i in range(args.repeats)]

    repeat_rows: List[Dict] = []
    for target_domain in args.target_domains:
        for repeat_idx, seed in enumerate(seeds, start=1):
            repeat_rows.append(run_repeat(args, target_domain, repeat_idx, seed, exp_dir))

    write_summary_csv(exp_dir / "all_repeats_summary.csv", repeat_rows)
    aggregate_rows = aggregate(repeat_rows, args.target_domains)
    write_summary_csv(exp_dir / "aggregate_summary.csv", aggregate_rows)
    print("Experiment finished. Results written to {}".format(exp_dir))


if __name__ == "__main__":
    main()
