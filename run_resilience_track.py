"""Resilience-track evaluation client.

Sends every image pair listed in ``<images_dir>/resilience_track.txt`` to the running
API (``/resilience/compare_images``) and writes the answers to
``<save_dir>/resilience_track.json``. The folders come from the ``images_dir`` and
``save_dir`` environment variables (set by the challenge Docker scripts) or from the
command line:

    images_dir=/data/test save_dir=/data/out python run_resilience_track.py
    python run_resilience_track.py --images-dir /data/test --save-dir /data/out
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
    IMAGES_DIR_ENV,
    RESILIENCE_OUTPUT,
    RESILIENCE_PROTOCOL,
    SAVE_DIR_ENV,
    read_resilience_protocol,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resilience-track evaluation client")
    parser.add_argument(
        "--images-dir",
        default=os.environ.get(IMAGES_DIR_ENV),
        help=f"folder with resilience_track.txt and the images (env: {IMAGES_DIR_ENV})",
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
        help="skip pairs with a missing image instead of stopping",
    )
    args = parser.parse_args(argv)
    if not args.images_dir or not args.save_dir:
        parser.error("images_dir and save_dir are required (environment or command line)")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    protocol = os.path.join(args.images_dir, RESILIENCE_PROTOCOL)
    if not os.path.isfile(protocol):
        print(f"Protocol file not found: {protocol}", file=sys.stderr)
        return 1
    pairs = [
        (os.path.join(args.images_dir, first), os.path.join(args.images_dir, second))
        for first, second in read_resilience_protocol(protocol)
    ]
    missing = sorted({path for pair in pairs for path in pair if not os.path.isfile(path)})
    if missing:
        print(f"{len(missing)} listed images do not exist, e.g. {missing[:3]}", file=sys.stderr)
        if not args.skip_missing:
            return 1
        absent = set(missing)
        pairs = [pair for pair in pairs if pair[0] not in absent and pair[1] not in absent]
    print(f"{len(pairs)} pairs to compare from {protocol}")

    base_url = args.url.rstrip("/")
    wait_for_api(base_url, args.wait)
    endpoint = f"{base_url}/resilience/compare_images"
    results, errors = [], 0
    started = time.time()
    with requests.Session() as session:
        for count, (first, second) in enumerate(pairs, 1):
            params = {"image_path_1": first, "image_path_2": second}
            answer, error = call_api(session, endpoint, params, args.timeout)
            if answer is None:
                answer = {**params, "similarity": "ERROR", "comment": error}
            if answer.get("comment") != "OK":
                errors += 1
                print(
                    f"[{count}/{len(pairs)}] {first}, {second}: {answer.get('comment')}",
                    file=sys.stderr,
                )
            results.append(answer)
            if count % 100 == 0 or count == len(pairs):
                rate = count / max(time.time() - started, 1e-9)
                print(f"{count}/{len(pairs)} pairs compared ({rate:.1f} per second)")

    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, RESILIENCE_OUTPUT)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(f"Saved {len(results)} results to {output_path} ({errors} errors)")
    return 1 if results and errors == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
