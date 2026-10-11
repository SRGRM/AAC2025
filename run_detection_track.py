"""Detection-track evaluation client.

Sends every image listed in ``<images_dir>/detection_track.txt`` to the running API
(``/detection/detect``) and writes the answers to ``<save_dir>/detection_track.json``.
The folders come from the ``images_dir`` and ``save_dir`` environment variables (set
by the challenge Docker scripts) or from the command line:

    images_dir=/data/test save_dir=/data/out python run_detection_track.py
    python run_detection_track.py --images-dir /data/test --save-dir /data/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import requests

from aac.client import call_api, wait_for_api
from aac.protocols import (
    DETECTION_OUTPUT,
    DETECTION_PROTOCOL,
    IMAGES_DIR_ENV,
    SAVE_DIR_ENV,
    read_detection_protocol,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection-track evaluation client")
    parser.add_argument(
        "--images-dir",
        default=os.environ.get(IMAGES_DIR_ENV),
        help=f"folder with detection_track.txt and the images (env: {IMAGES_DIR_ENV})",
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get(SAVE_DIR_ENV),
        help=f"output folder (env: {SAVE_DIR_ENV})",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("AAC_API_URL", "http://127.0.0.1:7007"),
        help="API base URL (env: AAC_API_URL)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    parser.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the API")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip listed images that do not exist instead of stopping",
    )
    args = parser.parse_args(argv)
    if not args.images_dir or not args.save_dir:
        parser.error("images_dir and save_dir are required (environment or command line)")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    protocol = os.path.join(args.images_dir, DETECTION_PROTOCOL)
    if not os.path.isfile(protocol):
        print(f"Protocol file not found: {protocol}", file=sys.stderr)
        return 1
    paths = [os.path.join(args.images_dir, name) for name in read_detection_protocol(protocol)]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        print(f"{len(missing)} listed images do not exist, e.g. {missing[:3]}", file=sys.stderr)
        if not args.skip_missing:
            return 1
        absent = set(missing)
        paths = [path for path in paths if path not in absent]
    print(f"{len(paths)} images to score from {protocol}")

    base_url = args.url.rstrip("/")
    wait_for_api(base_url, args.wait)
    endpoint = f"{base_url}/detection/detect"
    results, errors = [], 0
    started = time.time()
    with requests.Session() as session:
        for count, path in enumerate(paths, 1):
            answer, error = call_api(session, endpoint, {"image_path": path}, args.timeout)
            if answer is None:
                answer = {
                    "image_path": path,
                    "score": "ERROR",
                    "decision": "ERROR",
                    "comment": error,
                }
            if answer.get("comment") != "OK":
                errors += 1
                print(f"[{count}/{len(paths)}] {path}: {answer.get('comment')}", file=sys.stderr)
            results.append(answer)
            if count % 100 == 0 or count == len(paths):
                rate = count / max(time.time() - started, 1e-9)
                print(f"{count}/{len(paths)} images scored ({rate:.1f} per second)")

    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, DETECTION_OUTPUT)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(f"Saved {len(results)} results to {output_path} ({errors} errors)")
    return 1 if results and errors == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
