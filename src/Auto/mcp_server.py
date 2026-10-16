from __future__ import annotations

import argparse
from pathlib import Path

from .selector import Selector


def server_for(selector: Selector):
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise RuntimeError("install the optional MCP dependency with: uv sync --group mcp") from exc

    server = MCPServer("Voxel world inspector")

    @server.tool()
    def route_summary() -> dict:
        """List selected section runs and raw/final section counts."""
        return selector.summary()

    @server.tool()
    def inspect_station(index: int) -> dict:
        """Inspect one route point, its nine terrain samples, and raw/final choices."""
        return selector.station_detail(index)

    @server.tool()
    def inspect_point(point: list[float]) -> dict:
        """Find the nearest route station to [x, z] or [x, y, z] and inspect its assignment."""
        return selector.point_detail(point)

    @server.tool()
    def inspect_route(start: int, count: int = 16) -> dict:
        """Read up to 64 consecutive route stations and their selected sections."""
        if start < 0 or count < 1 or count > 64:
            raise ValueError("start must be nonnegative and count must be 1..64")
        return {"total": len(selector.stations), "stations": [
            station.as_dict() for station in selector.stations[start:start + count]
        ]}

    @server.tool()
    def inspect_area(x: int, z: int, radius: int = 2) -> dict:
        """Read ground, water, and structure heights in a square near x/z (radius 0..8)."""
        if radius < 0 or radius > 8:
            raise ValueError("radius must be 0..8")
        return {"x": x, "z": z, "radius": radius, "columns": [
            selector.column(cx, cz).as_dict()
            for cz in range(z - radius, z + radius + 1)
            for cx in range(x - radius, x + radius + 1)
        ]}

    @server.tool()
    def inspect_column(x: int, z: int) -> dict:
        """Read block names and states in the top of one world column."""
        return selector.column_detail(x, z)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only MCP tools for a build's route and world")
    parser.add_argument("build", type=Path, help="build TOML with an [auto] table")
    arguments = parser.parse_args()
    server_for(Selector.load(arguments.build)).run(transport="stdio")


if __name__ == "__main__":
    main()
