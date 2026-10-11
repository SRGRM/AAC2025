"""Train the detection-track model: is a face image clean or adversarial?

All settings are in CONFIG below. Edit them and run from the repository root:

    python train_detection_track.py

Each run gets its own folder under ``output_root`` containing config.json, train.log,
training_report.txt, history.csv, the checkpoint and validation predictions of every
epoch (epoch_XXX.pt, epoch_XXX_val.json), best.pt and last_state.pt (for ``resume``).
When ``submission_dir`` is set, the scores for its detection_track.txt are also
written after every epoch (epoch_XXX_detection_track.json, same format as the API).

With the default settings this trains the model of the challenge entry: an
ImageNet-initialised ResNet-18 on 224 x 224 inputs, binary cross-entropy, Adam (lr 1e-4)
with cosine annealing for 100 epochs and a decision threshold of 0.5.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from aac.checkpoint import write_json
from aac.corruptions import CORRUPTIONS
from aac.data import (
    CLEAN_CLASS,
    SYNTHETIC_CLASS,
    DataError,
    DetectionDataset,
    ImageListDataset,
    ImageSample,
    natural_key,
    scan_detection_data,
    split_items,
    verify_images,
)
from aac.losses import DetectionLoss
from aac.metrics import detection_metrics, select_threshold
from aac.models import NORMALIZATIONS, STREAMS, build_detection_model, scores_from_logits
from aac.protocols import DETECTION_PROTOCOL, read_detection_protocol
from aac.training import (
    BackboneFreezer,
    EpochLR,
    MetricTracker,
    ModelEMA,
    TrainingReport,
    autocast,
    build_optimizer,
    create_run_dir,
    load_train_state,
    make_grad_scaler,
    make_loader,
    maybe_data_parallel,
    param_groups,
    save_epoch_checkpoints,
    save_train_state,
    seed_everything,
    setup_devices,
    setup_logging,
    unwrap_model,
    warn_config_changes,
)
from aac.transforms import detection_train_transform, eval_transform

# fmt: off
CONFIG: Dict[str, Any] = {
    # Hardware and reproducibility
    "gpu_ids": [0],                  # e.g. [0] or [0, 1, 2, 3] (DataParallel); [] = CPU
    "seed": 42,                      # any integer
    "deterministic": False,          # True = reproducible cuDNN kernels (slower)
    "num_workers": 4,                # DataLoader workers; 0 = load in the main process
    "amp": True,                     # mixed precision on GPU; False = full fp32 (as in the challenge)

    # Data
    "data_root": "./AdvCelebA",      # folder with image_dir and the attack lists
    "image_dir": "images",           # sub-folder holding clean and attacked images
    "attack_list_glob": "final_attack_attackid_*.txt",  # one list per attack method
    "attack_list_header": True,      # True = first line of every list is a header
    "attack_list_column": 0,         # column with the attacked image name
    # Column with the clean source image of each attacked image. When set, an image and
    # its attacked copies always land on the same side of the train/validation split.
    "source_column": None,           # None | column index, e.g. 1
    "recursive": False,              # True = also scan sub-folders of image_dir
    "strict_data": True,             # True = stop on missing/unreadable files; False = skip them
    "verify_images": True,           # True = decode every image once before training
    "val_fraction": 0.1,             # held-out share for validation; 0.0 = none (as in the challenge)
    "balanced_sampling": False,      # True = sample clean/adversarial equally often

    # Labels
    "label_mode": "binary",          # "binary" | "attack_type" (one class per attack list)
    "synthetic_attack_prob": 0.0,    # 0.0-1.0: chance to corrupt a clean image and relabel it
    "synthetic_attacks": ["grid", "msb", "noise", "patch"],  # also "blur", "jpeg"

    # Input and augmentation
    "image_size": 224,               # input side in pixels, e.g. 128-384
    "streams": ["rgb"],              # any of "rgb", "srm" (noise residuals), "fft" (spectrum)
    "normalization": "none",         # "none" | "imagenet" | "face"
    "hflip": False,                  # random horizontal flips
    "crop_padding": 0,               # 0 = off; e.g. 8 = reflect-pad, then random crop
    "color_jitter": 0.0,             # 0.0 = off; e.g. 0.1
    "tta_flip": False,               # True = average scores of image and mirror image

    # Model
    "backbone": "resnet18",          # any timm name: "resnet50", "efficientnet_b0", "convnext_tiny", ...
    "pretrained": True,              # True = ImageNet weights from timm; False = random init
    "drop_rate": 0.0,                # dropout before the classifier, e.g. 0.2

    # Loss
    "loss": "ce",                    # "ce" (BCE when binary) | "focal"
    "pos_weight": None,              # binary only: None | "auto" (n_clean / n_adv) | a float
    "focal_gamma": 2.0,              # focal loss focusing parameter
    "focal_alpha": None,             # binary focal class weight: None | e.g. 0.25
    "label_smoothing": 0.0,          # 0.0 = off; e.g. 0.1

    # Optimisation
    "epochs": 100,                   # number of epochs, e.g. 30-200
    "batch_size": 64,                # images per batch
    "optimizer": "adam",             # "adam" | "adamw" | "sgd"
    "lr": 1e-4,                      # initial learning rate, e.g. 1e-3 with "sgd"
    "weight_decay": 0.0,             # e.g. 0.05 with "adamw"
    "momentum": 0.9,                 # "sgd" only
    "backbone_lr_mult": 1.0,         # backbone lr = lr * mult, e.g. 0.1
    "freeze_backbone_epochs": 0,     # train only the classifier for the first N epochs
    "scheduler": "cosine",           # "cosine" | "cosine_warmup" | "step" | "plateau" | "constant"
    "warmup_epochs": 0,              # "cosine_warmup" only
    "min_lr_ratio": 0.0,             # cosine floor as a fraction of lr
    "step_size": 30,                 # "step" only: epochs between decays
    "gamma": 0.1,                    # decay factor for "step" and "plateau"
    "plateau_patience": 5,           # "plateau" only
    "grad_clip": 0.0,                # max gradient norm; 0.0 = off
    "ema_decay": 0.0,                # 0.0 = off; e.g. 0.999 (EMA weights are evaluated and saved)
    "early_stopping_patience": 0,    # epochs without improvement before stopping; 0 = off

    # Model selection and decision threshold (both use the validation split)
    "selection_metric": "auc",       # "auc" | "f1" | "eer" | "acer" | "loss"
    "threshold_mode": "fixed",       # "fixed" | "eer" | "youden" | "f1" | "bpcer"
    "threshold": 0.5,                # used when threshold_mode = "fixed"
    "threshold_target": 0.01,        # highest allowed BPCER for threshold_mode = "bpcer"

    # Output
    "output_root": "./runs/detection",  # parent folder of the run folders
    "run_name": None,                # None = timestamp
    "save_every_epoch": True,        # True = keep epoch_XXX.pt for every epoch
    "submission_dir": None,          # None | folder with detection_track.txt and its images
    "log_interval": 50,              # batches between progress lines
    "resume": None,                  # None | path to last_state.pt of an interrupted run
}
# fmt: on

LOGGER = logging.getLogger("aac")

SELECTION_MODES = {"auc": "max", "f1": "max", "eer": "min", "acer": "min", "loss": "min"}
THRESHOLD_MODES = ("fixed", "eer", "youden", "f1", "bpcer")


def check_config(cfg: Dict[str, Any]) -> None:
    """Reject unknown option values before any work is done."""
    choices = {
        "label_mode": ("binary", "attack_type"),
        "normalization": tuple(NORMALIZATIONS),
        "loss": ("ce", "focal"),
        "optimizer": ("adam", "adamw", "sgd"),
        "scheduler": ("cosine", "cosine_warmup", "step", "plateau", "constant"),
        "selection_metric": tuple(SELECTION_MODES),
        "threshold_mode": THRESHOLD_MODES,
    }
    for key, allowed in choices.items():
        if cfg[key] not in allowed:
            raise ValueError(f"CONFIG[{key!r}] = {cfg[key]!r}; expected one of {allowed}")
    if not cfg["streams"] or set(cfg["streams"]) - set(STREAMS):
        raise ValueError(f"CONFIG['streams'] must be a non-empty subset of {STREAMS}")
    if set(cfg["synthetic_attacks"]) - set(CORRUPTIONS):
        raise ValueError(f"CONFIG['synthetic_attacks'] must be a subset of {sorted(CORRUPTIONS)}")
    if not 0.0 <= cfg["val_fraction"] < 1.0:
        raise ValueError("CONFIG['val_fraction'] must be in [0, 1)")
    if not 0.0 <= cfg["synthetic_attack_prob"] <= 1.0:
        raise ValueError("CONFIG['synthetic_attack_prob'] must be in [0, 1]")
    if cfg["pos_weight"] not in (None, "auto") and not isinstance(cfg["pos_weight"], (int, float)):
        raise ValueError("CONFIG['pos_weight'] must be None, 'auto' or a number")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["image_size"] < 16:
        raise ValueError("epochs and batch_size must be >= 1 and image_size >= 16")
    if cfg["scheduler"] == "plateau" and cfg["val_fraction"] == 0:
        raise ValueError("The plateau scheduler needs a validation split (val_fraction > 0)")


def prepare_data(cfg: Dict[str, Any]) -> Tuple[List[ImageSample], List[ImageSample], List[str]]:
    """Scan, check and split the data; returns train samples, validation samples and class names."""
    LOGGER.info("Reading detection data from %s", os.path.abspath(cfg["data_root"]))
    samples = scan_detection_data(
        cfg["data_root"],
        image_dir=cfg["image_dir"],
        list_glob=cfg["attack_list_glob"],
        has_header=cfg["attack_list_header"],
        image_column=cfg["attack_list_column"],
        source_column=cfg["source_column"],
        recursive=cfg["recursive"],
        strict=cfg["strict_data"],
    )
    if cfg["verify_images"]:
        LOGGER.info("Decoding every image once to check it")
        samples = verify_images(samples, lambda sample: [sample.path], strict=cfg["strict_data"])

    attacks = sorted({sample.attack for sample in samples} - {CLEAN_CLASS}, key=natural_key)
    if cfg["label_mode"] == "binary":
        class_names = [CLEAN_CLASS, "adversarial"]
    else:
        class_names = [CLEAN_CLASS] + attacks
        if cfg["synthetic_attack_prob"] > 0:
            class_names.append(SYNTHETIC_CLASS)

    group_by = (lambda sample: sample.group) if cfg["source_column"] is not None else None
    train, val = split_items(
        samples,
        cfg["val_fraction"],
        cfg["seed"],
        stratify_by=lambda sample: sample.attack,
        group_by=group_by,
    )
    for name, part in (("train", train), ("validation", val)):
        clean = sum(sample.attack == CLEAN_CLASS for sample in part)
        LOGGER.info(
            "%s: %d images (%d clean, %d adversarial)", name, len(part), clean, len(part) - clean
        )
    if val and len({sample.attack == CLEAN_CLASS for sample in val}) < 2:
        LOGGER.warning("The validation split holds a single class; AUC and EER are undefined")
    LOGGER.info("Classes: %s", class_names)
    return train, val, class_names


def load_submission_paths(cfg: Dict[str, Any]) -> List[str]:
    """Images listed in ``submission_dir/detection_track.txt`` (empty when not configured)."""
    if not cfg["submission_dir"]:
        return []
    protocol = os.path.join(cfg["submission_dir"], DETECTION_PROTOCOL)
    if not os.path.isfile(protocol):
        raise DataError(f"Protocol file not found: {protocol}")
    paths = [
        os.path.join(cfg["submission_dir"], name) for name in read_detection_protocol(protocol)
    ]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        message = f"{len(missing)} images listed in {protocol} do not exist, e.g. {missing[:3]}"
        if cfg["strict_data"]:
            raise DataError(message)
        LOGGER.warning(message)
        paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        raise DataError(f"No usable images in {protocol}")
    LOGGER.info("Submission protocol: %d images from %s", len(paths), protocol)
    return paths


def resolve_pos_weight(cfg: Dict[str, Any], samples: Sequence[ImageSample]) -> Optional[float]:
    """Weight of the adversarial class in the binary loss (None when not used)."""
    if cfg["pos_weight"] is None:
        return None
    if cfg["label_mode"] != "binary":
        LOGGER.warning("pos_weight only applies to label_mode = 'binary'; ignored")
        return None
    if cfg["pos_weight"] == "auto":
        clean = sum(sample.attack == CLEAN_CLASS for sample in samples)
        weight = clean / max(1, len(samples) - clean)
        LOGGER.info("pos_weight set to %.3f (clean / adversarial)", weight)
        return weight
    return float(cfg["pos_weight"])


def head_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    return ((name, p) for name, p in model.named_parameters() if not name.startswith("backbone."))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    ema: Optional[ModelEMA],
) -> float:
    """One pass over the training set; returns the mean training loss."""
    model.train()
    total_loss, seen = 0.0, 0
    started = time.time()
    for step, (images, targets, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, cfg["amp"]):
            logits = model(images)
        loss = criterion(logits, targets)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(
                f"Loss is {loss.item()} at epoch {epoch}, step {step}; "
                "try a lower lr or amp = False"
            )
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            nn.utils.clip_grad_norm_(parameters, cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        if step % cfg["log_interval"] == 0 or step == len(loader):
            LOGGER.info(
                "epoch %d | step %d/%d | loss %.4f | %.0f img/s",
                epoch,
                step,
                len(loader),
                total_loss / seen,
                seen / max(time.time() - started, 1e-9),
            )
    return total_loss / max(seen, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_mode: str,
    amp: bool,
    tta: bool,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Adversarial scores for every item of ``loader`` (plus targets and loss when labelled)."""
    model.eval()
    scores, targets, indices = [], [], []
    loss_sum, seen = 0.0, 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(images).float()
            mirrored = model(torch.flip(images, dims=[3])).float() if tta else None
        batch_scores = scores_from_logits(logits, label_mode)
        if mirrored is not None:
            batch_scores = 0.5 * (batch_scores + scores_from_logits(mirrored, label_mode))
        if len(batch) == 3:
            if criterion is not None:
                loss_sum += criterion(logits, batch[1].to(device)).item() * images.size(0)
                seen += images.size(0)
            targets.append(batch[1])
        scores.append(batch_scores.cpu())
        indices.append(batch[-1])
    result: Dict[str, Any] = {
        "scores": torch.cat(scores).numpy(),
        "indices": torch.cat(indices).numpy(),
    }
    if targets:
        result["targets"] = torch.cat(targets).numpy()
        result["loss"] = loss_sum / seen if seen else float("nan")
    return result


def detection_rate_per_attack(
    samples: Sequence[ImageSample], indices: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Share of each attack's validation images flagged at ``threshold``."""
    flagged: Dict[str, List[bool]] = defaultdict(list)
    for index, score in zip(indices, scores):
        attack = samples[int(index)].attack
        if attack != CLEAN_CLASS:
            flagged[attack].append(bool(score >= threshold))
    return {name: float(np.mean(flagged[name])) for name in sorted(flagged, key=natural_key)}


def main(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train with ``cfg`` (default: CONFIG); returns the run folder and the best epoch."""
    cfg = dict(CONFIG if cfg is None else cfg)
    check_config(cfg)
    state = load_train_state(cfg["resume"]) if cfg["resume"] else None
    if state is not None:
        run_dir = os.path.dirname(os.path.abspath(cfg["resume"]))
    else:
        run_dir = create_run_dir(cfg["output_root"], cfg["run_name"])
    setup_logging(run_dir)
    LOGGER.info("Detection track, run folder %s", os.path.abspath(run_dir))
    start_epoch = 1 if state is None else int(state["epoch"]) + 1
    if state is not None:
        LOGGER.info("Resuming after epoch %d", start_epoch - 1)
        warn_config_changes(run_dir, cfg)
    device, n_gpus = setup_devices(cfg["gpu_ids"])
    LOGGER.info("Device: %s (%d GPU%s)", device, n_gpus, "" if n_gpus == 1 else "s")
    seed_everything(cfg["seed"] + start_epoch - 1, cfg["deterministic"])

    # Data
    train_samples, val_samples, class_names = prepare_data(cfg)
    submission_paths = load_submission_paths(cfg)
    label_mode = cfg["label_mode"]
    size = cfg["image_size"]
    train_set = DetectionDataset(
        train_samples,
        class_names,
        label_mode,
        detection_train_transform(size, cfg["hflip"], cfg["crop_padding"], cfg["color_jitter"]),
        synthetic_prob=cfg["synthetic_attack_prob"],
        synthetic_kinds=cfg["synthetic_attacks"],
    )
    loader_seed = cfg["seed"] + start_epoch - 1  # a resumed run does not replay epoch 1's order
    sampler = None
    if cfg["balanced_sampling"]:
        counts = Counter(sample.attack == CLEAN_CLASS for sample in train_samples)
        weights = [1.0 / counts[sample.attack == CLEAN_CLASS] for sample in train_samples]
        generator = torch.Generator()
        generator.manual_seed(loader_seed)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
    loader_args = (cfg["batch_size"], cfg["num_workers"], device)
    train_loader = make_loader(
        train_set,
        *loader_args,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=len(train_set) >= cfg["batch_size"],
        seed=loader_seed,
    )
    val_loader = None
    if val_samples:
        val_set = DetectionDataset(val_samples, class_names, label_mode, eval_transform(size))
        val_loader = make_loader(val_set, *loader_args)
    submission_loader = None
    if submission_paths:
        submission_loader = make_loader(
            ImageListDataset(submission_paths, eval_transform(size)), *loader_args
        )
    LOGGER.info("%d training batches per epoch", len(train_loader))

    # Model, loss and optimisation
    model_config = {
        "task": "detection",
        "backbone": cfg["backbone"],
        "streams": list(cfg["streams"]),
        "normalization": cfg["normalization"],
        "label_mode": label_mode,
        "class_names": class_names,
        "num_outputs": 1 if label_mode == "binary" else len(class_names),
        "image_size": size,
        "drop_rate": cfg["drop_rate"],
        "tta_flip": cfg["tta_flip"],
    }
    model = build_detection_model(model_config, pretrained=cfg["pretrained"] and state is None)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    LOGGER.info(
        "Model: %s, streams %s, %.1fM parameters", cfg["backbone"], cfg["streams"], n_params
    )
    freezer = BackboneFreezer(model.backbone)
    criterion = DetectionLoss(
        label_mode,
        cfg["loss"],
        pos_weight=resolve_pos_weight(cfg, train_samples),
        focal_gamma=cfg["focal_gamma"],
        focal_alpha=cfg["focal_alpha"],
        label_smoothing=cfg["label_smoothing"],
    ).to(device)
    groups = param_groups(
        [
            (model.backbone.named_parameters(), cfg["lr"] * cfg["backbone_lr_mult"]),
            (head_parameters(model), cfg["lr"]),
        ],
        cfg["weight_decay"],
    )
    optimizer = build_optimizer(groups, cfg["optimizer"], cfg["momentum"])
    selection_mode = SELECTION_MODES[cfg["selection_metric"]]
    scheduler = EpochLR(
        optimizer,
        cfg["scheduler"],
        cfg["epochs"],
        warmup_epochs=cfg["warmup_epochs"],
        min_lr_ratio=cfg["min_lr_ratio"],
        step_size=cfg["step_size"],
        gamma=cfg["gamma"],
        plateau_mode=selection_mode,
        plateau_patience=cfg["plateau_patience"],
    )
    scaler = make_grad_scaler(device, cfg["amp"])
    tracker = MetricTracker(selection_mode)

    history = None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        tracker.load_state_dict(state["tracker"])
        history = state["history"]
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    if ema is not None and state is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    model = maybe_data_parallel(model, n_gpus)

    report = TrainingReport(run_dir, "Training report: detection track", cfg, history)
    if history is None:
        report.section(
            "Data",
            [
                f"train images: {len(train_samples)}",
                f"validation images: {len(val_samples)}",
                f"submission images: {len(submission_paths)}",
                f"classes: {class_names}",
            ],
        )

    # Training loop
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        started = time.time()
        scheduler.start_epoch(epoch - 1)
        freezer.set_frozen(epoch <= cfg["freeze_backbone_epochs"])
        lr = max(group["lr"] for group in optimizer.param_groups)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch, ema
        )
        eval_model = ema.module if ema is not None else model
        row: Dict[str, Any] = {"epoch": epoch, "lr": lr, "train_loss": train_loss}
        notes: List[str] = []
        metrics: Dict[str, float] = {}
        threshold = float(cfg["threshold"])
        tag = f"epoch_{epoch:03d}"

        if val_loader is not None:
            output = predict(
                eval_model, val_loader, device, label_mode, cfg["amp"], cfg["tta_flip"], criterion
            )
            is_attack = (output["targets"] != 0).astype(int)
            threshold = select_threshold(
                is_attack,
                output["scores"],
                cfg["threshold_mode"],
                fixed=cfg["threshold"],
                target=cfg["threshold_target"],
            )
            metrics = detection_metrics(is_attack, output["scores"], threshold)
            metrics["loss"] = output["loss"]
            row["val_loss"] = metrics["loss"]
            row.update(
                {key: metrics[key] for key in ("auc", "apcer", "bpcer", "f1", "eer", "acer")}
            )
            rates = detection_rate_per_attack(
                val_samples, output["indices"], output["scores"], threshold
            )
            notes.append(
                "detection rate per attack: "
                + ", ".join(f"{name} {rate:.3f}" for name, rate in rates.items())
            )
            write_json(
                [
                    {
                        "image_path": val_samples[int(index)].path,
                        "attack": val_samples[int(index)].attack,
                        "label": int(label),
                        "score": float(score),
                        "decision": bool(score >= threshold),
                    }
                    for index, label, score in zip(output["indices"], is_attack, output["scores"])
                ],
                os.path.join(run_dir, f"{tag}_val.json"),
            )
        row["threshold"] = threshold

        selection_value = metrics.get(cfg["selection_metric"]) if val_loader is not None else None
        if val_loader is not None and (
            selection_value is None or not math.isfinite(selection_value)
        ):
            LOGGER.warning("%s is undefined this epoch", cfg["selection_metric"])
        is_best = tracker.update(selection_value, epoch)
        save_epoch_checkpoints(
            run_dir,
            epoch,
            unwrap_model(eval_model).state_dict(),
            model_config,
            threshold,
            metrics,
            keep_epoch_file=cfg["save_every_epoch"],
            is_best=is_best,
        )

        if submission_loader is not None:
            output = predict(
                eval_model, submission_loader, device, label_mode, cfg["amp"], cfg["tta_flip"]
            )
            write_json(
                [
                    {
                        "image_path": submission_paths[int(index)],
                        "score": float(score),
                        "decision": bool(score >= threshold),
                        "comment": "OK",
                    }
                    for index, score in zip(output["indices"], output["scores"])
                ],
                os.path.join(run_dir, f"{tag}_detection_track.json"),
            )

        row["seconds"] = round(time.time() - started, 1)
        row["best"] = is_best
        report.add_epoch(row, notes)
        LOGGER.info(
            "epoch %d/%d done | %s%s",
            epoch,
            cfg["epochs"],
            " | ".join(
                f"{key} {value:.4f}"
                for key, value in row.items()
                if key not in ("epoch", "lr", "seconds", "best") and isinstance(value, float)
            ),
            " | best so far" if is_best else "",
        )
        for note in notes:
            LOGGER.info(note)
        scheduler.end_epoch(selection_value)
        save_train_state(
            os.path.join(run_dir, "last_state.pt"),
            epoch=epoch,
            model=unwrap_model(model).state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            ema=None if ema is None else ema.state_dict(),
            scheduler=scheduler.state_dict(),
            tracker=tracker.state_dict(),
            history=report.history,
        )
        patience = cfg["early_stopping_patience"]
        if patience and val_loader is not None and tracker.stale_epochs >= patience:
            LOGGER.info("No improvement for %d epochs, stopping early", patience)
            break

    summary = [
        f"best epoch: {tracker.best_epoch}",
        f"best checkpoint: {os.path.join(run_dir, 'best.pt')}",
    ]
    if tracker.best is not None:
        summary.insert(1, f"best {cfg['selection_metric']}: {tracker.best:.4f}")
    report.section("Summary", summary)
    for line in summary:
        LOGGER.info(line)
    return {"run_dir": run_dir, "best_epoch": tracker.best_epoch, "best_value": tracker.best}


if __name__ == "__main__":
    main()
