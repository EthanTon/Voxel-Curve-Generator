from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .Assembly import assemble, write
from .Parser import load


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a voxel curve from a config")
    parser.add_argument("config", help="path to the build's TOML file")
    arguments = parser.parse_args(argv)

    spec = load(arguments.config)
    written = write(assemble(spec), spec)
    print(written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
