"""Detection-track inference: how likely is a face image to be adversarial?

Loads one or more checkpoints written by ``train_detection_track.py``. The bare state
dict saved by the original challenge script (``detection_model_final.pt``) is accepted
too. With several checkpoints the scores are averaged.

Environment variables (all optional):

    AAC_DETECTION_WEIGHTS    comma-separated checkpoint paths (default: weights/detection.pt)
    AAC_DETECTION_THRESHOLD  decision threshold (default: from the checkpoints, else 0.5)
    AAC_DEVICE               "cpu", "cuda", "cuda:1", ... (default: CUDA when available)
    AAC_TTA                  1 / 0: average the scores of the image and its mirror image
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from aac.checkpoint import load_checkpoint
from aac.inference import env_flag, env_float, resolve_device, split_paths
from aac.models import LEGACY_DETECTION_CONFIG, build_detection_model, scores_from_logits
from aac.transforms import eval_transform, load_image

LOGGER = logging.getLogger(__name__)

DEFAULT_WEIGHTS = os.path.join("weights", "detection.pt")


class Classifier:
    """Adversarial-image detector (single model or score-averaging ensemble).

    Args:
        weights: checkpoint path(s); defaults to ``AAC_DETECTION_WEIGHTS``.
        device: torch device name; defaults to ``AAC_DEVICE`` or CUDA when available.
        threshold: decision threshold on the score; see the module docstring.
        tta: average the scores of the image and its horizontal mirror.
    """

    def __init__(
        self,
        weights: Optional[Union[str, Sequence[str]]] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        tta: Optional[bool] = None,
    ) -> None:
        paths = split_paths(weights or os.environ.get("AAC_DETECTION_WEIGHTS") or DEFAULT_WEIGHTS)
        if not paths:
            raise ValueError("No detection checkpoint given")
        self.device = resolve_device(device)
        self.members: List[Tuple[torch.nn.Module, Dict[str, Any], Any]] = []
        stored_thresholds = []
        for path in paths:
            checkpoint = load_checkpoint(path)
            config = checkpoint["model_config"] or dict(LEGACY_DETECTION_CONFIG)
            if config.get("task") != "detection":
                raise ValueError(f"{path} is not a detection checkpoint")
            model = build_detection_model(config)
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            self.members.append((model, config, eval_transform(int(config["image_size"]))))
            if checkpoint.get("threshold") is not None:
                stored_thresholds.append(float(checkpoint["threshold"]))
            LOGGER.info(
                "Detection model %s: %s, %s", path, config["backbone"], config["label_mode"]
            )

        if threshold is None:
            threshold = env_float("AAC_DETECTION_THRESHOLD")
        if threshold is None:
            threshold = (
                sum(stored_thresholds) / len(stored_thresholds) if stored_thresholds else 0.5
            )
        self.threshold = float(threshold)
        if tta is None:
            tta = env_flag("AAC_TTA")
        self.tta = bool(self.members[0][1].get("tta_flip", False)) if tta is None else tta
        LOGGER.info(
            "Detection threshold %.4f, TTA %s, device %s", self.threshold, self.tta, self.device
        )

    @torch.no_grad()
    def score(self, image_path: str) -> float:
        """Adversarial score in [0, 1] (mean over the ensemble)."""
        image = load_image(image_path)
        scores = []
        for model, config, transform in self.members:
            batch = transform(image).unsqueeze(0).to(self.device)
            score = scores_from_logits(model(batch), config["label_mode"])
            if self.tta:
                mirrored = scores_from_logits(
                    model(torch.flip(batch, dims=[3])), config["label_mode"]
                )
                score = 0.5 * (score + mirrored)
            scores.append(float(score.item()))
        return sum(scores) / len(scores)

    def predict(self, image_path: str) -> Tuple[float, bool]:
        """``(score, decision)``; ``decision`` is True when the image is flagged as adversarial."""
        score = self.score(image_path)
        return score, score >= self.threshold
