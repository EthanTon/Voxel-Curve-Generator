from __future__ import annotations

import argparse
import json
from pathlib import Path

from .selector import Selector


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview and write a build's automatic assignments")
    parser.add_argument("build", type=Path, help="build TOML with an [auto] table")
    parser.add_argument("--dry-run", action="store_true", help="show runs without writing")
    parser.add_argument("--station", type=int, help="also show the sampled columns at one station")
    parser.add_argument("--point", nargs="+", type=float,
                        help="inspect the nearest station to X Z or X Y Z")
    arguments = parser.parse_args()
    if arguments.point is not None and len(arguments.point) not in (2, 3):
        parser.error("--point needs X Z or X Y Z")
    selector = Selector.load(arguments.build)
    print(json.dumps(selector.summary(), indent=2))
    if arguments.station is not None:
        print(json.dumps(selector.station_detail(arguments.station), indent=2))
    if arguments.point is not None:
        print(json.dumps(selector.point_detail(arguments.point), indent=2))
    if not arguments.dry_run and selector.settings.output is not None:
        print(f"Wrote {selector.write()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
