#!/usr/bin/env python3
"""Compact Mojang's generated blocks report into the runtime catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blocks_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--faces-report",
        type=Path,
        help="Optional output from DumpVanillaFaces for exact sturdy-face masks",
    )
    parser.add_argument(
        "--block-tags-dir",
        type=Path,
        help="Optional extracted data/minecraft/tags/block directory",
    )
    args = parser.parse_args()

    source_bytes = args.blocks_report.read_bytes()
    report = json.loads(source_bytes)
    state_metadata = (
        _read_state_metadata(args.faces_report) if args.faces_report else {}
    )
    block_tags = (
        _read_block_tags(args.block_tags_dir) if args.block_tags_dir else {}
    )
    if state_metadata:
        expected_ids = {
            state["id"]
            for block in report.values()
            for state in block["states"]
        }
        missing = expected_ids - state_metadata.keys()
        if missing:
            raise ValueError(
                f"Faces report is missing {len(missing)} block states"
            )
    blocks: dict[str, object] = {}
    for name, block in sorted(report.items()):
        default = next(state for state in block["states"] if state.get("default"))
        entry = {
            "type": block.get("definition", {})
            .get("type", "minecraft:block")
            .removeprefix("minecraft:"),
            "properties": block.get("properties", {}),
            "default": default.get("properties", {}),
        }
        tags = sorted(tag for tag, members in block_tags.items() if name in members)
        if tags:
            entry["tags"] = tags
        if state_metadata:
            default_values = state_metadata[default["id"]]
            default_mask, default_exception, default_flammable = default_values[:3]
            entry["faces"] = default_mask
            entry["connection_exception"] = default_exception
            entry["flammable"] = default_flammable
            property_order = sorted(block.get("properties", {}))
            differences = {
                _state_key(state.get("properties", {}), property_order): state_metadata[
                    state["id"]
                ][0]
                for state in block["states"]
                if state_metadata[state["id"]][0] != default_mask
            }
            if differences:
                entry["face_states"] = differences
            exception_differences = {
                _state_key(state.get("properties", {}), property_order): state_metadata[
                    state["id"]
                ][1]
                for state in block["states"]
                if state_metadata[state["id"]][1] != default_exception
            }
            if exception_differences:
                entry["connection_exception_states"] = exception_differences
            flammable_differences = {
                _state_key(state.get("properties", {}), property_order): state_metadata[
                    state["id"]
                ][2]
                for state in block["states"]
                if state_metadata[state["id"]][2] != default_flammable
            }
            if flammable_differences:
                entry["flammable_states"] = flammable_differences
            for index, (default_key, states_key) in enumerate(
                (
                    ("center_faces", "center_face_states"),
                    ("rigid_faces", "rigid_face_states"),
                    ("signal_source", "signal_source_states"),
                    ("wall_cover", "wall_cover_states"),
                ),
                start=3,
            ):
                default_value = default_values[index]
                entry[default_key] = default_value
                state_differences = {
                    _state_key(state.get("properties", {}), property_order):
                    state_metadata[state["id"]][index]
                    for state in block["states"]
                    if state_metadata[state["id"]][index] != default_value
                }
                if state_differences:
                    entry[states_key] = state_differences
        blocks[name] = entry

    payload = {
        "version": args.version,
        "source": "Mojang server data generator: net.minecraft.data.Main --reports",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "blocks": blocks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_state_metadata(
    path: Path,
) -> dict[int, tuple[int, bool, bool, int, int, bool, int]]:
    metadata: dict[int, tuple[int, bool, bool, int, int, bool, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(
            r"(\d+)\t(\d+)\t(true|false)\t(true|false)"
            r"\t(\d+)\t(\d+)\t(true|false)\t(\d+)$",
            line,
        )
        if match:
            metadata[int(match.group(1))] = (
                int(match.group(2)),
                match.group(3) == "true",
                match.group(4) == "true",
                int(match.group(5)),
                int(match.group(6)),
                match.group(7) == "true",
                int(match.group(8)),
            )
    return metadata


def _read_block_tags(path: Path) -> dict[str, frozenset[str]]:
    raw_tags: dict[str, list[tuple[str, bool]]] = {}
    for tag_file in path.rglob("*.json"):
        tag_name = tag_file.relative_to(path).with_suffix("").as_posix()
        payload = json.loads(tag_file.read_text(encoding="utf-8"))
        values: list[tuple[str, bool]] = []
        for raw_value in payload.get("values", ()):
            if isinstance(raw_value, str):
                values.append((raw_value, True))
            else:
                values.append(
                    (str(raw_value["id"]), raw_value.get("required", True))
                )
        raw_tags[tag_name] = values

    resolved: dict[str, frozenset[str]] = {}

    def resolve(
        tag_name: str,
        visiting: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        tag_name = tag_name.removeprefix("minecraft:")
        if tag_name in resolved:
            return resolved[tag_name]
        if tag_name in visiting:
            raise ValueError(f"Cyclic block tag reference: {tag_name}")
        members: set[str] = set()
        for value, required in raw_tags.get(tag_name, ()):
            if value.startswith("#"):
                referenced = value[1:].removeprefix("minecraft:")
                if referenced not in raw_tags and required:
                    raise ValueError(f"Missing required block tag: {referenced}")
                members.update(resolve(referenced, visiting | {tag_name}))
            else:
                members.add(value)
        result = frozenset(members)
        resolved[tag_name] = result
        return result

    for tag_name in raw_tags:
        resolve(tag_name)
    return resolved


def _state_key(properties: dict[str, str], order: list[str]) -> str:
    return ",".join(properties[name] for name in order)


if __name__ == "__main__":
    main()
