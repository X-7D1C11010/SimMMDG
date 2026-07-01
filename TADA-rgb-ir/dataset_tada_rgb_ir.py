from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairRecord:
    rgb_path: str
    ir_path: str
    label: int
    class_name: str
    domain: str
    split: str


@dataclass
class ClassPairStats:
    rgb_files: int
    ir_files: int
    paired_samples: int
    exact_pairs: int
    cycled_pairs: int


def natural_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def list_image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_key(p.stem),
    )


def infer_layout(split_dir: Path, rgb_dir_name: str, ir_dir_name: str) -> str:
    if (split_dir / rgb_dir_name).is_dir() and (split_dir / ir_dir_name).is_dir():
        return "modality_first"

    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()] if split_dir.exists() else []
    if any((p / rgb_dir_name).is_dir() and (p / ir_dir_name).is_dir() for p in class_dirs):
        return "class_first"

    raise FileNotFoundError(
        "Cannot infer RGB-IR layout under '{}'. Expected either "
        "'split/{rgb,class}/...' or 'split/{class,rgb}/...' style directories.".format(split_dir)
    )


def class_dirs_for_split(split_dir: Path, layout: str, rgb_dir_name: str, ir_dir_name: str) -> List[str]:
    if layout == "modality_first":
        names = set()
        for modality in (rgb_dir_name, ir_dir_name):
            modality_dir = split_dir / modality
            if modality_dir.exists():
                names.update(p.name for p in modality_dir.iterdir() if p.is_dir())
        return sorted(names, key=natural_key)

    names = [
        p.name
        for p in split_dir.iterdir()
        if p.is_dir() and ((p / rgb_dir_name).is_dir() or (p / ir_dir_name).is_dir())
    ]
    return sorted(names, key=natural_key)


def modality_paths(
    split_dir: Path,
    layout: str,
    class_name: str,
    rgb_dir_name: str,
    ir_dir_name: str,
) -> Tuple[Path, Path]:
    if layout == "modality_first":
        return split_dir / rgb_dir_name / class_name, split_dir / ir_dir_name / class_name
    return split_dir / class_name / rgb_dir_name, split_dir / class_name / ir_dir_name


def label_for_class(class_name: str, class_to_idx: Optional[Dict[str, int]]) -> int:
    if class_to_idx is not None:
        if class_name not in class_to_idx:
            raise KeyError("Class '{}' is not present in class_to_idx.".format(class_name))
        return class_to_idx[class_name]
    if class_name.isdigit():
        return int(class_name) - 1
    raise ValueError(
        "Non-numeric class '{}' requires an explicit class_to_idx mapping.".format(class_name)
    )


def pair_files(
    rgb_files: Sequence[Path],
    ir_files: Sequence[Path],
    pairing: str,
) -> Tuple[List[Tuple[Path, Path]], ClassPairStats]:
    if not rgb_files or not ir_files:
        stats = ClassPairStats(
            rgb_files=len(rgb_files),
            ir_files=len(ir_files),
            paired_samples=0,
            exact_pairs=0,
            cycled_pairs=0,
        )
        return [], stats

    rgb_by_stem = {p.stem: p for p in rgb_files}
    ir_by_stem = {p.stem: p for p in ir_files}
    common_stems = sorted(set(rgb_by_stem) & set(ir_by_stem), key=natural_key)

    if pairing == "strict":
        rgb_stems = sorted(rgb_by_stem, key=natural_key)
        ir_stems = sorted(ir_by_stem, key=natural_key)
        if rgb_stems != ir_stems:
            raise ValueError(
                "Strict pairing failed: {} RGB stems, {} IR stems, {} common stems.".format(
                    len(rgb_stems), len(ir_stems), len(common_stems)
                )
            )

    if pairing in {"strict", "intersection"}:
        pairs = [(rgb_by_stem[s], ir_by_stem[s]) for s in common_stems]
        stats = ClassPairStats(
            rgb_files=len(rgb_files),
            ir_files=len(ir_files),
            paired_samples=len(pairs),
            exact_pairs=len(pairs),
            cycled_pairs=0,
        )
        return pairs, stats

    if pairing != "cycle":
        raise ValueError("Unknown pairing mode '{}'. Use strict, intersection, or cycle.".format(pairing))

    pairs = [(rgb_by_stem[s], ir_by_stem[s]) for s in common_stems]
    used_rgb = {p for p, _ in pairs}
    used_ir = {p for _, p in pairs}
    remaining_rgb = [p for p in rgb_files if p not in used_rgb]
    remaining_ir = [p for p in ir_files if p not in used_ir]

    missing_pairs = max(len(remaining_rgb), len(remaining_ir))
    if missing_pairs:
        rgb_fallback = cycle(rgb_files)
        ir_fallback = cycle(ir_files)
        for i in range(missing_pairs):
            rgb = remaining_rgb[i] if i < len(remaining_rgb) else next(rgb_fallback)
            ir = remaining_ir[i] if i < len(remaining_ir) else next(ir_fallback)
            pairs.append((rgb, ir))

    stats = ClassPairStats(
        rgb_files=len(rgb_files),
        ir_files=len(ir_files),
        paired_samples=len(pairs),
        exact_pairs=len(common_stems),
        cycled_pairs=max(0, len(pairs) - len(common_stems)),
    )
    return pairs, stats


def scan_tada_split(
    data_root: Path,
    domain: str,
    split: str,
    pairing: str = "cycle",
    rgb_dir_name: str = "可见光",
    ir_dir_name: str = "红外",
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[List[PairRecord], Dict]:
    split_dir = Path(data_root) / domain / split
    if not split_dir.exists():
        raise FileNotFoundError("Split directory does not exist: {}".format(split_dir))

    layout = infer_layout(split_dir, rgb_dir_name, ir_dir_name)
    records: List[PairRecord] = []
    class_stats: Dict[str, Dict] = {}

    for class_name in class_dirs_for_split(split_dir, layout, rgb_dir_name, ir_dir_name):
        rgb_dir, ir_dir = modality_paths(split_dir, layout, class_name, rgb_dir_name, ir_dir_name)
        rgb_files = list_image_files(rgb_dir)
        ir_files = list_image_files(ir_dir)
        pairs, stats = pair_files(rgb_files, ir_files, pairing)
        label = label_for_class(class_name, class_to_idx)

        class_stats[class_name] = asdict(stats)
        for rgb_path, ir_path in pairs:
            records.append(
                PairRecord(
                    rgb_path=str(rgb_path),
                    ir_path=str(ir_path),
                    label=label,
                    class_name=class_name,
                    domain=domain,
                    split=split,
                )
            )

    summary = {
        "data_root": str(data_root),
        "domain": domain,
        "split": split,
        "layout": layout,
        "pairing": pairing,
        "rgb_dir_name": rgb_dir_name,
        "ir_dir_name": ir_dir_name,
        "num_samples": len(records),
        "num_classes_with_pairs": sum(1 for s in class_stats.values() if s["paired_samples"] > 0),
        "classes": class_stats,
    }
    return records, summary


class TADARgbIrDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        domain: str,
        split: str,
        transform_rgb=None,
        transform_ir=None,
        pairing: str = "cycle",
        rgb_dir_name: str = "可见光",
        ir_dir_name: str = "红外",
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.records, self.summary = scan_tada_split(
            data_root=Path(data_root),
            domain=domain,
            split=split,
            pairing=pairing,
            rgb_dir_name=rgb_dir_name,
            ir_dir_name=ir_dir_name,
            class_to_idx=class_to_idx,
        )
        if not self.records:
            raise ValueError("No paired RGB-IR samples found for domain={} split={}.".format(domain, split))

        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir if transform_ir is not None else transform_rgb

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.rgb_path) as rgb_img:
            rgb_img = rgb_img.convert("RGB")
        with Image.open(record.ir_path) as ir_img:
            ir_img = ir_img.convert("RGB")

        if self.transform_rgb is not None:
            rgb_img = self.transform_rgb(rgb_img)
        if self.transform_ir is not None:
            ir_img = self.transform_ir(ir_img)

        return {
            "rgb": rgb_img,
            "ir": ir_img,
            "label": torch.tensor(record.label, dtype=torch.long),
            "class_name": record.class_name,
            "domain": record.domain,
            "rgb_path": record.rgb_path,
            "ir_path": record.ir_path,
        }


def build_tada_transforms(image_size: int = 224, train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_numeric_class_mapping(num_classes: int) -> Dict[str, int]:
    return {str(i): i - 1 for i in range(1, num_classes + 1)}


def collect_domain_summaries(
    data_root: str,
    domains: Iterable[str],
    splits: Iterable[str],
    pairing: str,
    num_classes: int,
    rgb_dir_name: str = "可见光",
    ir_dir_name: str = "红外",
) -> Dict[str, Dict[str, Dict]]:
    class_to_idx = build_numeric_class_mapping(num_classes)
    summaries: Dict[str, Dict[str, Dict]] = {}
    for domain in domains:
        summaries[domain] = {}
        for split in splits:
            _, summary = scan_tada_split(
                Path(data_root),
                domain=domain,
                split=split,
                pairing=pairing,
                rgb_dir_name=rgb_dir_name,
                ir_dir_name=ir_dir_name,
                class_to_idx=class_to_idx,
            )
            summaries[domain][split] = summary
    return summaries
