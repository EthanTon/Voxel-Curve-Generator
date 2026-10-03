from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .Assembly import run


class _ProgressBar:
    """Adapt assembly stage callbacks to tqdm without coupling it to the build."""

    def __init__(self, stream: TextIO) -> None:
        # Progress is optional, so normal runs and imports do not need tqdm.
        from tqdm import tqdm

        self.tqdm = tqdm
        self.stream = stream
        self.stage = ""
        self.bar = None
        self.disabled = False

    def __call__(self, stage: str, completed: int, total: int) -> None:
        if self.disabled:
            return
        try:
            if self.bar is None or stage != self.stage:
                self.close()
                if self.disabled:
                    return
                self.stage = stage
                self.bar = self.tqdm(
                    total=max(1, total), desc=stage, file=self.stream,
                    mininterval=0.1, miniters=1, dynamic_ncols=True,
                )
            # A stage with no work is already complete.
            count = completed if total else 1
            self.bar.update(count - self.bar.n)
        except (OSError, ValueError):
            self.disabled = True
            self.close()

    def close(self) -> None:
        if self.bar is not None:
            try:
                self.bar.close()
            except (OSError, ValueError):
                self.disabled = True
            finally:
                self.bar = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a voxel curve from a config")
    parser.add_argument("config", help="path to the build's TOML file")
    parser.add_argument(
        "--manifest", nargs="?", const=True, default=False, type=Path, metavar="PATH",
        help="write a TOML list of every output and print its path; optionally choose PATH",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="show tqdm progress bars on stderr (requires tqdm)",
    )
    arguments = parser.parse_args(argv)

    progress = None
    if arguments.verbose:
        try:
            progress = _ProgressBar(sys.stderr)
        except ModuleNotFoundError as error:
            if error.name != "tqdm":
                raise
            print(
                "Progress unavailable: install tqdm with 'python -m pip install tqdm'.",
                file=sys.stderr,
            )
    try:
        result = run(arguments.config, manifest=arguments.manifest, progress=progress)
    finally:
        if progress is not None:
            progress.close()
    print(result.manifest or result.primary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
