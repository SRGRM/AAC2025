"""REST API used by the challenge evaluation (port 7007 by default).

Endpoints (GET, parameters in the query string):

    /resilience/extract_template   image_path                  -> data, comment
    /resilience/compare_templates  template_1, template_2      -> similarity, comment
    /resilience/compare_images     image_path_1, image_path_2  -> similarity, comment
    /detection/detect              image_path                  -> score, decision, comment
    /health                                                    -> which models are loaded

Templates are JSON lists of floats. Failures are reported inside the response ("ERROR"
values and the message in ``comment``) rather than as HTTP errors, as the evaluation
clients expect.

Start with ``python main.py``. Settings come from environment variables:

    AAC_HOST, AAC_PORT        bind address (default 0.0.0.0:7007)
    AAC_DETECTION_WEIGHTS     detection checkpoint(s), or "none" to serve resilience only
    AAC_RESILIENCE_WEIGHTS    resilience checkpoint(s), or "none" to serve detection only
    AAC_IMAGE_ROOT            if set, only images inside this folder can be read
    (plus the model options documented in classifier.py and template_extractor.py)
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI

from aac import __version__
from classifier import Classifier
from template_extractor import TemplateExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
LOGGER = logging.getLogger("aac.api")


def _disabled(variable: str) -> bool:
    return os.environ.get(variable, "").strip().lower() == "none"


def load_models() -> Tuple[Optional[TemplateExtractor], Optional[Classifier]]:
    """Load the models selected by the environment variables (see the module docstring)."""
    extractor = None if _disabled("AAC_RESILIENCE_WEIGHTS") else TemplateExtractor()
    classifier = None if _disabled("AAC_DETECTION_WEIGHTS") else Classifier()
    if extractor is None and classifier is None:
        raise RuntimeError("Both tracks are disabled; nothing to serve")
    return extractor, classifier


def _checked_path(path: str) -> str:
    """Refuse paths outside AAC_IMAGE_ROOT when that variable is set."""
    if not path:
        raise ValueError("image path is empty")
    root = os.environ.get("AAC_IMAGE_ROOT")
    if root:
        real_root = os.path.realpath(root)
        real_path = os.path.realpath(path)
        if os.path.commonpath([real_root, real_path]) != real_root:
            raise PermissionError(f"{path} is outside AAC_IMAGE_ROOT")
    return path


def create_app(extractor: Optional[TemplateExtractor], classifier: Optional[Classifier]) -> FastAPI:
    """FastAPI application around the two models; a missing model disables its endpoints."""
    app = FastAPI(title="AAC 2025 face recognition security API", version=__version__)

    def require_extractor() -> TemplateExtractor:
        if extractor is None:
            raise RuntimeError("resilience model not loaded")
        return extractor

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "detection": classifier is not None,
            "resilience": extractor is not None,
        }

    @app.get("/resilience/extract_template")
    def extract_template(image_path: str = "") -> Dict[str, Any]:
        """Template (embedding) of one image."""
        try:
            template = require_extractor().extract(_checked_path(image_path))
            return {"data": template.tolist(), "comment": "OK"}
        except Exception as exc:  # noqa: BLE001 - reported to the client
            LOGGER.warning("extract_template(%s): %s", image_path, exc)
            return {"data": "ERROR", "comment": str(exc)}

    @app.get("/resilience/compare_templates")
    def compare_templates(template_1: str = "", template_2: str = "") -> Dict[str, Any]:
        """Cosine similarity of two templates given as JSON lists."""
        try:
            first = np.asarray(json.loads(template_1), dtype=float)
            second = np.asarray(json.loads(template_2), dtype=float)
            if first.shape != second.shape:
                return {"similarity": "ERROR", "comment": "Templates size mismatch"}
            return {"similarity": TemplateExtractor.compare(first, second), "comment": "OK"}
        except Exception as exc:  # noqa: BLE001 - reported to the client
            return {"similarity": "ERROR", "comment": str(exc)}

    @app.get("/resilience/compare_images")
    def compare_images(image_path_1: str = "", image_path_2: str = "") -> Dict[str, Any]:
        """Cosine similarity of the templates of two images."""
        response: Dict[str, Any] = {"image_path_1": image_path_1, "image_path_2": image_path_2}
        try:
            model = require_extractor()
            first = model.extract(_checked_path(image_path_1))
            second = model.extract(_checked_path(image_path_2))
            response.update(similarity=model.compare(first, second), comment="OK")
        except Exception as exc:  # noqa: BLE001 - reported to the client
            LOGGER.warning("compare_images(%s, %s): %s", image_path_1, image_path_2, exc)
            response.update(similarity="ERROR", comment=str(exc))
        return response

    @app.get("/detection/detect")
    def detect(image_path: str = "") -> Dict[str, Any]:
        """Adversarial score and decision (True = adversarial) for one image."""
        response: Dict[str, Any] = {"image_path": image_path}
        try:
            if classifier is None:
                raise RuntimeError("detection model not loaded")
            score, decision = classifier.predict(_checked_path(image_path))
            response.update(score=score, decision=bool(decision), comment="OK")
        except Exception as exc:  # noqa: BLE001 - reported to the client
            LOGGER.warning("detect(%s): %s", image_path, exc)
            response.update(score="ERROR", decision="ERROR", comment=str(exc))
        return response

    return app


app = create_app(*load_models())


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("AAC_HOST", "0.0.0.0"),
        port=int(os.environ.get("AAC_PORT", "7007")),
    )
