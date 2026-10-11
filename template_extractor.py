"""Resilience-track inference: face templates (embeddings) and their comparison.

Loads one or more checkpoints written by ``train_resilience_track.py``. The bare state
dict saved by the original challenge script (``resilience_best.pt``) is accepted too.

With several checkpoints the template is the concatenation of the L2-normalised
embeddings divided by sqrt(number of models), so the cosine similarity of two templates
equals the average of the per-model cosine similarities.

Environment variables (all optional):

    AAC_RESILIENCE_WEIGHTS  comma-separated checkpoint paths (default: weights/resilience.pt)
    AAC_DEVICE              "cpu", "cuda", "cuda:1", ... (default: CUDA when available)
    AAC_TTA                 1 / 0: average the embeddings of the image and its mirror image
    AAC_INPUT_DEFENSE       replaces the purification stored in the checkpoint
                            ("none", "gaussian_blur", "median", "jpeg", "bit_depth")
    AAC_DEFENSE_PARAM       strength of that defence (sigma, kernel size, quality or bits)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from aac.checkpoint import load_checkpoint
from aac.corruptions import apply_defense
from aac.inference import env_flag, env_float, resolve_device, split_paths
from aac.models import LEGACY_RESILIENCE_CONFIG, build_embedding_model
from aac.transforms import eval_transform, load_image

LOGGER = logging.getLogger(__name__)

DEFAULT_WEIGHTS = os.path.join("weights", "resilience.pt")


@dataclass
class _Member:
    model: torch.nn.Module
    transform: Any
    defense: str
    defense_param: Optional[float]


class TemplateExtractor:
    """Siamese-branch embedder used for template extraction and comparison.

    Args:
        weights: checkpoint path(s); defaults to ``AAC_RESILIENCE_WEIGHTS``.
        device: torch device name; defaults to ``AAC_DEVICE`` or CUDA when available.
        tta: average the embeddings of the image and its horizontal mirror.
        defense: input purification overriding the one stored in the checkpoints.
        defense_param: strength of ``defense``.
    """

    def __init__(
        self,
        weights: Optional[Union[str, Sequence[str]]] = None,
        device: Optional[str] = None,
        tta: Optional[bool] = None,
        defense: Optional[str] = None,
        defense_param: Optional[float] = None,
    ) -> None:
        paths = split_paths(weights or os.environ.get("AAC_RESILIENCE_WEIGHTS") or DEFAULT_WEIGHTS)
        if not paths:
            raise ValueError("No resilience checkpoint given")
        self.device = resolve_device(device)
        defense = defense or os.environ.get("AAC_INPUT_DEFENSE") or None
        if defense_param is None:
            defense_param = env_float("AAC_DEFENSE_PARAM")

        self.members: List[_Member] = []
        stored_tta = []
        for path in paths:
            checkpoint = load_checkpoint(path)
            config = checkpoint["model_config"] or dict(LEGACY_RESILIENCE_CONFIG)
            stored_tta.append(bool(config.get("tta_flip", False)))
            if config.get("task") != "resilience":
                raise ValueError(f"{path} is not a resilience checkpoint")
            model = build_embedding_model(config)
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            member_defense = defense or config.get("input_defense", "none")
            if defense_param is not None:
                member_param = defense_param
            elif defense and defense != config.get("input_defense"):
                member_param = None  # a different defence starts from its default strength
            else:
                member_param = config.get("defense_param")
            self.members.append(
                _Member(
                    model, eval_transform(int(config["image_size"])), member_defense, member_param
                )
            )
            LOGGER.info(
                "Resilience model %s: %s, %d-d, defence %s",
                path,
                config["backbone"],
                model.embedding_dim,
                member_defense,
            )
        if tta is None:
            tta = env_flag("AAC_TTA")
        self.tta = stored_tta[0] if tta is None else tta

    @torch.no_grad()
    def extract(self, image_path: str) -> np.ndarray:
        """Unit-length float32 template of one image."""
        image = load_image(image_path)
        parts = []
        for member in self.members:
            purified = apply_defense(image, member.defense, member.defense_param)
            batch = member.transform(purified).unsqueeze(0).to(self.device)
            embedding = member.model(batch)
            if self.tta:
                embedding = F.normalize(
                    embedding + member.model(torch.flip(batch, dims=[3])), dim=1
                )
            parts.append(embedding[0].float().cpu())
        template = torch.cat(parts) / math.sqrt(len(parts))
        return template.numpy().astype(np.float32)

    @staticmethod
    def compare(reference: Sequence[float], probe: Sequence[float]) -> float:
        """Cosine similarity of two templates."""
        reference = np.asarray(reference, dtype=np.float64).ravel()
        probe = np.asarray(probe, dtype=np.float64).ravel()
        if reference.shape != probe.shape:
            raise ValueError(f"Template sizes differ: {reference.size} vs {probe.size}")
        norm = np.linalg.norm(reference) * np.linalg.norm(probe)
        if norm == 0:
            raise ValueError("Cannot compare an all-zero template")
        return float(np.dot(reference, probe) / norm)
