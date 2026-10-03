from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomli_w

from ..Parser import load
from ..Parser.spec import BuildSpec
from .assemble import build
from .export import write_outputs
from .progress import Progress, report


@dataclass(frozen=True, slots=True)
class RunResult:
    """Absolute paths to the primary output, all outputs, and optional manifest."""

    primary: Path
    outputs: tuple[Path, ...]
    manifest: Path | None = None


def run(
    config: str | Path | BuildSpec,
    *,
    manifest: bool | str | Path = False,
    progress: Progress | None = None,
) -> RunResult:
    """Load, build, and export a config, using the same pipeline as the CLI.

    ``from src.Assembly import run``
    ``result = run("build.toml", manifest=True)``

    ``result.outputs`` contains every file written, including both the split
    placement manifest and this run's manifest when enabled. ``manifest=True``
    writes ``<primary stem>.outputs.toml`` beside the primary output; a path
    selects a custom location (relative to the caller's working directory).
    An already parsed BuildSpec is also accepted. Progress callbacks receive
    ``(stage, completed, total)``; counts refer to that stage, not elapsed time.
    No console output is produced by default, and build errors propagate.
    """
    report(progress, "Loading config", 0)
    if isinstance(config, BuildSpec):
        source = None
        spec = config
    else:
        source = Path(config).resolve()
        spec = load(source)
    report(progress, "Loading config", 1)

    target = None
    if manifest is not False:
        split = spec.output.split
        placement = None
        primary = spec.output.path
        if split is not None:
            placement = Path(split.directory) / f"{spec.output.name}.toml"
            if not split.combined:
                primary = placement
        target = (
            primary.with_name(f"{primary.stem}.outputs.toml")
            if manifest is True else Path(manifest)
        ).resolve()
        if target.suffix.lower() != ".toml":
            raise ValueError("the output manifest must have a .toml suffix")
        reserved = {spec.output.path.resolve(), source}
        if placement is not None:
            reserved.add(placement.resolve())
        if target in reserved:
            raise ValueError("the output manifest must not replace the config or an export")

    built = build(spec, progress=progress)
    written = write_outputs(built.world, spec, built.terraforming, progress=progress)
    primary = written.primary.resolve()
    outputs = tuple(path.resolve() for path in written.outputs)
    if target is not None:
        report(progress, "Writing output manifest", 0)
        outputs = (*outputs, target)
        data = {
            "schema_version": 1,
            "primary": str(primary),
            "manifest": str(target),
            "outputs": [str(path) for path in outputs],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(tomli_w.dumps(data), encoding="utf-8")
        report(progress, "Writing output manifest", 1)
    return RunResult(primary, outputs, target)
