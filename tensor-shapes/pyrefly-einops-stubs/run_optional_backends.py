#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check real einops stubs with exactly one minimal shaped backend fixture.

No runtime packages or builds are needed. Fixtures model only the nominal types
in einops signatures, not backend constructors or runtime implementations. Missing
imports in each probe also verify that the other backends cannot be resolved.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BACKENDS = ("numpy", "jax", "torch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyrefly", type=Path, required=True, help="existing binary")
    parser.add_argument("--backend", choices=BACKENDS, action="append", default=[])
    parser.add_argument("--preset", choices=("default", "strict"), default="default")
    parser.add_argument("--nocapture", action="store_true")
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    test_root = package_root / "test" / "optional_backends"
    failed = 0
    for backend in args.backend or BACKENDS:
        command = [
            str(args.pyrefly.resolve()),
            "check",
            "--config",
            os.devnull,
            "--preset",
            args.preset,
            "--python-version",
            "3.13",
            "--skip-interpreter-query",
            "--disable-search-path-heuristics",
            "--expectations",
        ]
        for search_path in (
            test_root / "fixtures" / backend,
            package_root,
            package_root.parent / "pyrefly-shape-extensions",
        ):
            command.extend(["--search-path", str(search_path)])
        command.append(str(test_root / f"{backend}_probe.py"))
        result = subprocess.run(command, capture_output=True, text=True)
        if args.nocapture or result.returncode:
            print("+ " + " ".join(command), flush=True)
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            failed = result.returncode
        else:
            print(f"PASS {backend}-only", flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
