"""Train the resilience-track model (Siamese network with PGD adversarial training).

All settings are in CONFIG below. Edit them and run from the repository root:

    python train_resilience_track.py

With the default settings this trains the model of the challenge entry: a Siamese
network whose two branches share an ImageNet-initialised ResNet-18 with a 512-d linear
head and L2-normalised output, trained on 500 x 500 inputs with a cosine contrastive loss
(margin 0.5) and PGD adversarial training. Pairs are compared by cosine similarity.

Each run gets its own folder under ``output_root`` containing config.json, train.log,
training_report.txt, history.csv, the checkpoint and validation similarities of every
epoch (epoch_XXX.pt, epoch_XXX_val.json), best.pt and last_state.pt (for ``resume``).
When ``submission_dir`` is set, the similarities for its resilience_track.txt are also
written after every epoch (epoch_XXX_resilience_track.json, same format as the API).
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from aac.attacks import TARGETS, pgd_pair_attack
from aac.checkpoint import load_checkpoint, write_json
from aac.corruptions import CORRUPTIONS, DEFENSES
from aac.data import (
    DataError,
    ImageListDataset,
    PairDataset,
    PairSample,
    scan_pair_data,
    split_items,
    verify_images,
)
from aac.iresnet import IRESNET_LAYERS
from aac.losses import PairLoss
from aac.metrics import verification_metrics
from aac.models import HEADS, NORMALIZATIONS, build_embedding_model
from aac.protocols import RESILIENCE_PROTOCOL, read_resilience_protocol
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
from aac.transforms import eval_transform, resilience_train_transform

# fmt: off
CONFIG: Dict[str, Any] = {
    # Hardware and reproducibility
    "gpu_ids": [0],                  # e.g. [0] or [0, 1, 2, 3] (DataParallel); [] = CPU
    "seed": 42,                      # any integer
    "deterministic": False,          # True = reproducible cuDNN kernels (slower)
    "num_workers": 4,                # DataLoader workers; 0 = load in the main process
    "amp": True,                     # mixed precision on GPU; False = full fp32 (as in the challenge)

    # Data
    "data_root": "./resilience_pairs",  # folder with pair_<n>_label_<y>/ sub-folders
    "strict_data": True,             # True = stop on incomplete/unreadable pairs; False = skip them
    "verify_images": True,           # True = decode every image once before training
    "val_fraction": 0.1,             # held-out share of pairs; 0.0 = no validation (as in the challenge)

    # Input, augmentation and input purification
    "image_size": 500,               # 500 in the challenge entry; 112 for IResNet backbones
    "normalization": "none",         # "none" | "imagenet" | "face" ((x - 0.5) / 0.5)
    "hflip": True,                   # random horizontal flips
    "vflip": False,                  # random vertical flips
    "rotation": 0.0,                 # max random rotation in degrees; 0.0 = off, e.g. 15
    "color_jitter": 0.0,             # 0.0 = off; e.g. 0.1
    "corruption_prob": 0.0,          # chance to corrupt one image of a training pair
    "corruptions": ["grid", "msb", "noise", "patch", "blur", "jpeg"],  # any subset
    # Purification applied to every image in training and at inference (it is stored in
    # the checkpoint, so the API applies it too).
    "input_defense": "none",         # "none" | "gaussian_blur" | "median" | "jpeg" | "bit_depth"
    "defense_param": None,           # None = default: sigma 1.0 | size 3 | quality 75 | 5 bits
    "tta_flip": False,               # True = average embeddings of image and mirror image

    # Model
    # Any timm model name, or an IResNet face backbone:
    # "iresnet18" | "iresnet34" | "iresnet50" | "iresnet100" | "iresnet200"
    "backbone": "resnet18",          # e.g. "resnet50", "convnext_tiny", "iresnet50"
    "pretrained": True,              # timm backbones: True = ImageNet weights; False = random init
    "backbone_weights": None,        # None | path to backbone weights (e.g. an InsightFace .pth)
    "head": "linear",                # "linear" | "bn" | "mlp" | "identity"
    "embedding_dim": 512,            # embedding size, e.g. 256 or 512 (not used by "identity")
    "drop_rate": 0.0,                # dropout before the head, e.g. 0.2

    # Loss
    "loss": "contrastive",           # "contrastive" | "double_margin" | "logistic"
    "margin": 0.5,                   # contrastive: impostor cosine above this is penalised
    "pos_margin": 0.8,               # double_margin: genuine cosine below this is penalised
    "neg_margin": 0.3,               # double_margin: impostor cosine above this is penalised
    "logistic_scale": 10.0,          # logistic: initial scale (learnt)
    "logistic_bias": -5.0,           # logistic: initial bias (learnt)
    "hard_pair_ratio": 1.0,          # 1.0 = all pairs; e.g. 0.5 = hardest half of each batch

    # Adversarial training (perturbations are computed on the network input)
    "adv_training": True,            # True = PGD adversarial training; False = clean pairs only
    "adv_method": "pgd",             # "pgd" | "fgsm" (single step, random start, alpha = 1.25 eps)
    "adv_eps": 8 / 255,              # L-inf budget in [0, 1] pixel units, e.g. 4 / 255 or 16 / 255
    "adv_alpha": 2 / 255,            # PGD step size (also used by the validation attack)
    "adv_steps": 7,                  # PGD iterations, e.g. 3 for faster epochs
    "adv_random_start": True,        # start PGD from a random point in the eps-ball
    "adv_target": "random",          # perturbed image: "first" | "second" | "both" | "random"
    "adv_weight": 0.5,               # loss = (1 - w) * clean + w * adversarial; 1.0 = adversarial only
    "adv_start_epoch": 1,            # first epoch (1-based) that uses adversarial pairs
    "eval_attack_steps": 0,          # 0 = off; e.g. 10 = also report EER under white-box PGD (adv_eer)

    # Optimisation
    "epochs": 100,                   # number of epochs, e.g. 50-500
    "batch_size": 32,                # pairs per batch
    "optimizer": "adam",             # "adam" | "adamw" | "sgd"
    "lr": 1e-4,                      # initial learning rate, e.g. 1e-3 with "sgd"
    "weight_decay": 0.0,             # e.g. 0.05 with "adamw"
    "momentum": 0.9,                 # "sgd" only
    "backbone_lr_mult": 1.0,         # backbone lr = lr * mult, e.g. 0.1
    "freeze_backbone_epochs": 0,     # train only the head for the first N epochs
    "scheduler": "cosine",           # "cosine" | "cosine_warmup" | "step" | "plateau" | "constant"
    "warmup_epochs": 0,              # "cosine_warmup" only
    "min_lr_ratio": 0.0,             # cosine floor as a fraction of lr
    "step_size": 30,                 # "step" only: epochs between decays
    "gamma": 0.1,                    # decay factor for "step" and "plateau"
    "plateau_patience": 5,           # "plateau" only
    "grad_clip": 0.0,                # max gradient norm; 0.0 = off
    "ema_decay": 0.0,                # 0.0 = off; e.g. 0.999 (EMA weights are evaluated and saved)
    "early_stopping_patience": 0,    # epochs without improvement before stopping; 0 = off

    # Model selection (validation split)
    "selection_metric": "eer",       # "eer" | "auc" | "tar_far_1e-2" | "tar_far_1e-3" | "adv_eer" | "loss"

    # Output
    "output_root": "./runs/resilience",  # parent folder of the run folders
    "run_name": None,                # None = timestamp
    "save_every_epoch": True,        # True = keep epoch_XXX.pt for every epoch
    "submission_dir": None,          # None | folder with resilience_track.txt and its images
    "umap_plot": False,              # True = save a UMAP plot of validation embeddings
    "log_interval": 20,              # batches between progress lines
    "resume": None,                  # None | path to last_state.pt of an interrupted run
}
# fmt: on

LOGGER = logging.getLogger("aac")

SELECTION_MODES = {
    "eer": "min",
    "auc": "max",
    "tar_far_1e-2": "max",
    "tar_far_1e-3": "max",
    "adv_eer": "min",
    "loss": "min",
}

# Colours of the UMAP plot: blue and orange stay distinct for colour-blind readers.
PLOT_BACKGROUND, PLOT_TEXT, PLOT_LEGEND = "#fcfcfb", "#0b0b0b", "#52514e"
PLOT_IMPOSTOR, PLOT_GENUINE = "#2a78d6", "#eb6834"


def check_config(cfg: Dict[str, Any]) -> None:
    """Reject unknown option values before any work is done."""
    choices = {
        "normalization": tuple(NORMALIZATIONS),
        "input_defense": ("none",) + tuple(DEFENSES),
        "head": HEADS,
        "loss": ("contrastive", "double_margin", "logistic"),
        "adv_method": ("pgd", "fgsm"),
        "adv_target": TARGETS,
        "optimizer": ("adam", "adamw", "sgd"),
        "scheduler": ("cosine", "cosine_warmup", "step", "plateau", "constant"),
        "selection_metric": tuple(SELECTION_MODES),
    }
    for key, allowed in choices.items():
        if cfg[key] not in allowed:
            raise ValueError(f"CONFIG[{key!r}] = {cfg[key]!r}; expected one of {allowed}")
    if set(cfg["corruptions"]) - set(CORRUPTIONS):
        raise ValueError(f"CONFIG['corruptions'] must be a subset of {sorted(CORRUPTIONS)}")
    if not 0.0 <= cfg["val_fraction"] < 1.0:
        raise ValueError("CONFIG['val_fraction'] must be in [0, 1)")
    if not 0.0 <= cfg["adv_weight"] <= 1.0 or not 0.0 <= cfg["corruption_prob"] <= 1.0:
        raise ValueError("adv_weight and corruption_prob must be in [0, 1]")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["image_size"] < 16:
        raise ValueError("epochs and batch_size must be >= 1 and image_size >= 16")
    if cfg["backbone"] in IRESNET_LAYERS and cfg["image_size"] % 16:
        raise ValueError(
            "IResNet backbones need an image_size divisible by 16 (112 for InsightFace)"
        )
    if cfg["selection_metric"] == "adv_eer" and cfg["eval_attack_steps"] < 1:
        raise ValueError("selection_metric 'adv_eer' needs eval_attack_steps >= 1")
    if (
        cfg["freeze_backbone_epochs"] > 0
        and cfg["head"] == "identity"
        and cfg["loss"] != "logistic"
    ):
        raise ValueError(
            "With head 'identity' every trainable weight is in the backbone, so "
            "freeze_backbone_epochs would leave nothing to train"
        )
    if cfg["scheduler"] == "plateau" and cfg["val_fraction"] == 0:
        raise ValueError("The plateau scheduler needs a validation split (val_fraction > 0)")
    if cfg["umap_plot"]:
        try:
            import matplotlib  # noqa: F401
            import umap  # noqa: F401
        except ImportError as exc:
            raise ImportError("umap_plot = True needs: pip install umap-learn matplotlib") from exc


def attack_settings(cfg: Dict[str, Any]) -> Tuple[float, float, int, bool]:
    """(eps, alpha, steps, random_start) of the training attack."""
    eps = float(cfg["adv_eps"])
    if cfg["adv_method"] == "fgsm":
        return eps, 1.25 * eps, 1, True
    return eps, float(cfg["adv_alpha"]), int(cfg["adv_steps"]), bool(cfg["adv_random_start"])


def prepare_data(cfg: Dict[str, Any]) -> Tuple[List[PairSample], List[PairSample]]:
    """Scan, check and split the pair folders; returns train and validation pairs."""
    LOGGER.info("Reading pairs from %s", os.path.abspath(cfg["data_root"]))
    pairs = scan_pair_data(cfg["data_root"], strict=cfg["strict_data"])
    if cfg["verify_images"]:
        LOGGER.info("Decoding every image once to check it")
        pairs = verify_images(
            pairs, lambda pair: [pair.path1, pair.path2], strict=cfg["strict_data"]
        )
    train, val = split_items(pairs, cfg["val_fraction"], cfg["seed"], stratify_by=lambda p: p.label)
    for name, part in (("train", train), ("validation", val)):
        genuine = sum(pair.label for pair in part)
        LOGGER.info(
            "%s: %d pairs (%d genuine, %d impostor)", name, len(part), genuine, len(part) - genuine
        )
    return train, val


def load_submission_pairs(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Pairs listed in ``submission_dir/resilience_track.txt`` (empty when not configured)."""
    if not cfg["submission_dir"]:
        return []
    folder = cfg["submission_dir"]
    protocol = os.path.join(folder, RESILIENCE_PROTOCOL)
    if not os.path.isfile(protocol):
        raise DataError(f"Protocol file not found: {protocol}")
    pairs = [
        (os.path.join(folder, first), os.path.join(folder, second))
        for first, second in read_resilience_protocol(protocol)
    ]
    missing = sorted({path for pair in pairs for path in pair if not os.path.isfile(path)})
    if missing:
        message = f"{len(missing)} images listed in {protocol} do not exist, e.g. {missing[:3]}"
        if cfg["strict_data"]:
            raise DataError(message)
        LOGGER.warning(message)
        pairs = [pair for pair in pairs if os.path.isfile(pair[0]) and os.path.isfile(pair[1])]
    if not pairs:
        raise DataError(f"No usable pairs in {protocol}")
    LOGGER.info("Submission protocol: %d pairs from %s", len(pairs), protocol)
    return pairs


def head_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    return ((name, p) for name, p in model.named_parameters() if not name.startswith("backbone."))


def embed(
    model: nn.Module, images: torch.Tensor, device: torch.device, amp: bool, tta: bool
) -> torch.Tensor:
    """L2-normalised embeddings, optionally averaged with those of the mirrored images."""
    with autocast(device, amp):
        embeddings = model(images).float()
        if tta:
            mirrored = model(torch.flip(images, dims=[3])).float()
            embeddings = F.normalize(embeddings + mirrored, dim=1)
    return embeddings


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: PairLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    ema: Optional[ModelEMA],
) -> Dict[str, float]:
    """One pass over the training pairs (plus their adversarial versions); returns mean losses."""
    model.train()
    adversarial = cfg["adv_training"] and epoch >= cfg["adv_start_epoch"]
    eps, alpha, steps, random_start = attack_settings(cfg)
    weight = cfg["adv_weight"] if adversarial else 0.0

    totals = {"loss": 0.0, "clean_loss": 0.0, "adv_loss": 0.0}
    seen = 0
    started = time.time()
    for step, (images1, images2, labels, _) in enumerate(loader, 1):
        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if adversarial:
            adv1, adv2 = pgd_pair_attack(
                model,
                images1,
                images2,
                labels,
                eps,
                alpha,
                steps,
                random_start=random_start,
                target=cfg["adv_target"],
            )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, cfg["amp"]):
            emb1, emb2 = model(torch.cat([images1, images2])).chunk(2)
            if adversarial:
                adv_emb1, adv_emb2 = model(torch.cat([adv1, adv2])).chunk(2)
        clean_loss = criterion(emb1, emb2, labels)
        adv_loss = criterion(adv_emb1, adv_emb2, labels) if adversarial else clean_loss.detach()
        loss = (1.0 - weight) * clean_loss + weight * adv_loss
        if not math.isfinite(loss.item()):
            raise FloatingPointError(
                f"Loss is {loss.item()} at epoch {epoch}, step {step}; try a lower lr or amp = False"
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

        batch = labels.size(0)
        seen += batch
        totals["loss"] += loss.item() * batch
        totals["clean_loss"] += clean_loss.item() * batch
        totals["adv_loss"] += adv_loss.item() * batch
        if step % cfg["log_interval"] == 0 or step == len(loader):
            LOGGER.info(
                "epoch %d | step %d/%d | loss %.4f (clean %.4f, adversarial %s) | %.1f pairs/s",
                epoch,
                step,
                len(loader),
                totals["loss"] / seen,
                totals["clean_loss"] / seen,
                f"{totals['adv_loss'] / seen:.4f}" if adversarial else "off",
                seen / max(time.time() - started, 1e-9),
            )
    result = {key: value / max(seen, 1) for key, value in totals.items()}
    if not adversarial:
        result["adv_loss"] = float("nan")
    return result


@torch.no_grad()
def evaluate_pairs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    criterion: PairLoss,
) -> Dict[str, Any]:
    """Cosine similarity, label and mean loss for every validation pair."""
    model.eval()
    similarities, labels, indices = [], [], []
    loss_sum, seen = 0.0, 0
    for images1, images2, target, index in loader:
        images = torch.cat([images1, images2]).to(device, non_blocking=True)
        emb1, emb2 = embed(model, images, device, cfg["amp"], cfg["tta_flip"]).chunk(2)
        loss_sum += criterion.per_pair(emb1, emb2, target.to(device)).sum().item()
        seen += target.numel()
        similarities.append((emb1 * emb2).sum(dim=1).cpu())
        labels.append(target)
        indices.append(index)
    return {
        "similarities": torch.cat(similarities).numpy(),
        "labels": torch.cat(labels).numpy().astype(int),
        "indices": torch.cat(indices).numpy(),
        "loss": loss_sum / max(seen, 1),
    }


def evaluate_under_attack(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict[str, Any]
) -> float:
    """EER on the validation pairs after a white-box PGD attack on the evaluated model.

    The attack works on the network input (after resizing and any input purification),
    so the result measures the robustness of the network itself.
    """
    model.eval()
    similarities, labels = [], []
    for images1, images2, target, _ in loader:
        images1, images2 = images1.to(device), images2.to(device)
        adv1, adv2 = pgd_pair_attack(
            model,
            images1,
            images2,
            target.to(device),
            float(cfg["adv_eps"]),
            float(cfg["adv_alpha"]),
            int(cfg["eval_attack_steps"]),
            random_start=True,
            target=cfg["adv_target"],
        )
        with torch.no_grad():
            images = torch.cat([adv1, adv2])
            emb1, emb2 = embed(model, images, device, cfg["amp"], cfg["tta_flip"]).chunk(2)
        similarities.append((emb1 * emb2).sum(dim=1).cpu())
        labels.append(target)
    return verification_metrics(torch.cat(labels).numpy(), torch.cat(similarities).numpy())["eer"]


@torch.no_grad()
def embed_images(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict[str, Any]
) -> np.ndarray:
    """Embeddings of every image of ``loader``, in dataset order."""
    model.eval()
    chunks = [
        embed(
            model, images.to(device, non_blocking=True), device, cfg["amp"], cfg["tta_flip"]
        ).cpu()
        for images, _ in loader
    ]
    return torch.cat(chunks).numpy()


def save_umap_plot(
    model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict[str, Any], path: str
) -> None:
    """2-D UMAP of the validation embeddings, one colour and marker per pair label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for images1, images2, target, _ in loader:
            images = torch.cat([images1, images2]).to(device)
            embeddings.append(
                embed(model, images, device, cfg["amp"], cfg["tta_flip"]).cpu().numpy()
            )
            labels.extend(target.int().tolist() * 2)  # first images, then second images
    points = umap.UMAP(min_dist=0.1, random_state=cfg["seed"]).fit_transform(
        np.concatenate(embeddings)
    )
    labels_array = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PLOT_BACKGROUND)
    ax.set_facecolor(PLOT_BACKGROUND)
    series = (
        (0, PLOT_IMPOSTOR, "o", "impostor pairs (impersonation)"),
        (1, PLOT_GENUINE, "^", "genuine pairs (evasion)"),
    )
    for value, colour, marker, name in series:
        selected = labels_array == value
        ax.scatter(
            points[selected, 0],
            points[selected, 1],
            s=16,
            c=colour,
            marker=marker,
            edgecolors=PLOT_BACKGROUND,
            linewidths=0.4,
            alpha=0.85,
            label=f"{name}: {selected.sum()} images",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("UMAP of validation embeddings", color=PLOT_TEXT, fontsize=12, loc="left")
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(PLOT_LEGEND)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=PLOT_BACKGROUND)
    plt.close(fig)


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
    LOGGER.info("Resilience track, run folder %s", os.path.abspath(run_dir))
    start_epoch = 1 if state is None else int(state["epoch"]) + 1
    if state is not None:
        LOGGER.info("Resuming after epoch %d", start_epoch - 1)
        warn_config_changes(run_dir, cfg)
    device, n_gpus = setup_devices(cfg["gpu_ids"])
    LOGGER.info("Device: %s (%d GPU%s)", device, n_gpus, "" if n_gpus == 1 else "s")
    seed_everything(cfg["seed"] + start_epoch - 1, cfg["deterministic"])

    # Data
    train_pairs, val_pairs = prepare_data(cfg)
    submission_pairs = load_submission_pairs(cfg)
    size = cfg["image_size"]
    defense, defense_param = cfg["input_defense"], cfg["defense_param"]
    train_set = PairDataset(
        train_pairs,
        resilience_train_transform(
            size, cfg["hflip"], cfg["vflip"], cfg["rotation"], cfg["color_jitter"]
        ),
        defense=defense,
        defense_param=defense_param,
        corruption_prob=cfg["corruption_prob"],
        corruption_kinds=cfg["corruptions"],
    )
    loader_args = (cfg["batch_size"], cfg["num_workers"], device)
    train_loader = make_loader(
        train_set,
        *loader_args,
        shuffle=True,
        drop_last=len(train_set) >= cfg["batch_size"],
        seed=cfg["seed"] + start_epoch - 1,  # a resumed run does not replay epoch 1's order
    )
    val_loader = None
    if val_pairs:
        val_set = PairDataset(
            val_pairs, eval_transform(size), defense=defense, defense_param=defense_param
        )
        val_loader = make_loader(val_set, *loader_args)
    submission_loader, submission_paths = None, []
    if submission_pairs:
        submission_paths = sorted({path for pair in submission_pairs for path in pair})
        submission_set = ImageListDataset(
            submission_paths, eval_transform(size), defense, defense_param
        )
        submission_loader = make_loader(submission_set, *loader_args)
    LOGGER.info("%d training batches per epoch", len(train_loader))
    if cfg["adv_training"]:
        eps, alpha, steps, _ = attack_settings(cfg)
        LOGGER.info(
            "Adversarial training: %s, eps %.4f, step %.4f, %d step(s), target %s, weight %.2f",
            cfg["adv_method"],
            eps,
            alpha,
            steps,
            cfg["adv_target"],
            cfg["adv_weight"],
        )

    # Model, loss and optimisation
    model_config = {
        "task": "resilience",
        "backbone": cfg["backbone"],
        "head": cfg["head"],
        "embedding_dim": cfg["embedding_dim"],
        "normalization": cfg["normalization"],
        "image_size": size,
        "drop_rate": cfg["drop_rate"],
        "input_defense": defense,
        "defense_param": defense_param,
        "tta_flip": cfg["tta_flip"],
    }
    model = build_embedding_model(
        model_config,
        pretrained=cfg["pretrained"] and state is None,
        backbone_weights=cfg["backbone_weights"] if state is None else None,
    )
    model.to(device)
    model_config["embedding_dim"] = model.embedding_dim
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    LOGGER.info(
        "Model: %s, head %s, %d-d embeddings, %.1fM parameters",
        cfg["backbone"],
        cfg["head"],
        model.embedding_dim,
        n_params,
    )
    freezer = BackboneFreezer(model.backbone)
    criterion = PairLoss(
        cfg["loss"],
        margin=cfg["margin"],
        pos_margin=cfg["pos_margin"],
        neg_margin=cfg["neg_margin"],
        scale=cfg["logistic_scale"],
        bias=cfg["logistic_bias"],
        hard_ratio=cfg["hard_pair_ratio"],
    ).to(device)
    groups = param_groups(
        [
            (model.backbone.named_parameters(), cfg["lr"] * cfg["backbone_lr_mult"]),
            (head_parameters(model), cfg["lr"]),
            (criterion.named_parameters(), cfg["lr"]),
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
        criterion.load_state_dict(state["criterion"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        tracker.load_state_dict(state["tracker"])
        history = state["history"]
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None
    if ema is not None and state is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    model = maybe_data_parallel(model, n_gpus)

    report = TrainingReport(run_dir, "Training report: resilience track", cfg, history)
    if history is None:
        report.section(
            "Data",
            [
                f"train pairs: {len(train_pairs)}",
                f"validation pairs: {len(val_pairs)}",
                f"submission pairs: {len(submission_pairs)}",
            ],
        )

    # Training loop
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        started = time.time()
        scheduler.start_epoch(epoch - 1)
        freezer.set_frozen(epoch <= cfg["freeze_backbone_epochs"])
        lr = max(group["lr"] for group in optimizer.param_groups)
        losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch, ema
        )
        eval_model = ema.module if ema is not None else model
        row: Dict[str, Any] = {"epoch": epoch, "lr": lr}
        row.update({f"train_{key}": value for key, value in losses.items()})
        metrics: Dict[str, float] = {}
        threshold: Optional[float] = None
        tag = f"epoch_{epoch:03d}"

        if val_loader is not None:
            output = evaluate_pairs(eval_model, val_loader, device, cfg, criterion)
            metrics = verification_metrics(output["labels"], output["similarities"])
            metrics["loss"] = output["loss"]
            if cfg["eval_attack_steps"] > 0:
                metrics["adv_eer"] = evaluate_under_attack(eval_model, val_loader, device, cfg)
            threshold = metrics["threshold"]
            row["val_loss"] = metrics["loss"]
            for key in ("auc", "eer", "tar_far_1e-2", "tar_far_1e-3", "adv_eer", "threshold"):
                if key in metrics:
                    row[key] = metrics[key]
            write_json(
                [
                    {
                        "image_path_1": val_pairs[int(index)].path1,
                        "image_path_2": val_pairs[int(index)].path2,
                        "label": int(label),
                        "similarity": float(similarity),
                    }
                    for index, label, similarity in zip(
                        output["indices"], output["labels"], output["similarities"]
                    )
                ],
                os.path.join(run_dir, f"{tag}_val.json"),
            )

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
            embeddings = embed_images(eval_model, submission_loader, device, cfg)
            position = {path: i for i, path in enumerate(submission_paths)}
            write_json(
                [
                    {
                        "image_path_1": first,
                        "image_path_2": second,
                        "similarity": float(
                            np.dot(embeddings[position[first]], embeddings[position[second]])
                        ),
                        "comment": "OK",
                    }
                    for first, second in submission_pairs
                ],
                os.path.join(run_dir, f"{tag}_resilience_track.json"),
            )

        row["seconds"] = round(time.time() - started, 1)
        row["best"] = is_best
        report.add_epoch(row)
        LOGGER.info(
            "epoch %d/%d done | %s%s",
            epoch,
            cfg["epochs"],
            " | ".join(
                f"{key} {value:.4f}"
                for key, value in row.items()
                if key not in ("epoch", "lr", "seconds", "best")
                and isinstance(value, float)
                and math.isfinite(value)
            ),
            " | best so far" if is_best else "",
        )
        scheduler.end_epoch(selection_value)
        save_train_state(
            os.path.join(run_dir, "last_state.pt"),
            epoch=epoch,
            model=unwrap_model(model).state_dict(),
            criterion=criterion.state_dict(),
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

    best_path = os.path.join(run_dir, "best.pt")
    if cfg["umap_plot"] and val_loader is not None:
        best_checkpoint = load_checkpoint(best_path)
        best_model = build_embedding_model(best_checkpoint["model_config"]).to(device)
        best_model.load_state_dict(best_checkpoint["model_state"])
        plot_path = os.path.join(run_dir, "umap_val_embeddings.png")
        save_umap_plot(best_model, val_loader, device, cfg, plot_path)
        LOGGER.info("Saved %s", plot_path)

    summary = [f"best epoch: {tracker.best_epoch}", f"best checkpoint: {best_path}"]
    if tracker.best is not None:
        summary.insert(1, f"best {cfg['selection_metric']}: {tracker.best:.4f}")
    report.section("Summary", summary)
    for line in summary:
        LOGGER.info(line)
    return {"run_dir": run_dir, "best_epoch": tracker.best_epoch, "best_value": tracker.best}


if __name__ == "__main__":
    main()
