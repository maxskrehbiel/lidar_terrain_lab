"""Land report: a JSON summary plus the same content as Markdown and standalone HTML."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rasterio.crs import CRS

from . import __version__
from .pipeline import Products
from .validation import Check

Table = tuple[list[str], list[list[str]]]


@dataclass(frozen=True)
class RunMetadata:
    """Descriptive fields printed in reports.

    Attributes:
        title: Report title.
        data_label: Short provenance label, e.g. ``SYNTHETIC`` or ``USGS 3DEP``.
        source: One-sentence description of the input data.
        notes: Extra caveats shown under the title.
    """

    title: str
    data_label: str
    source: str
    notes: str = ""


def _crs_label(crs: str | None) -> str:
    if crs is None:
        return "none"
    parsed = CRS.from_user_input(crs)
    epsg = parsed.to_epsg()
    return f"EPSG:{epsg}" if epsg else str(parsed.to_dict().get("proj", "custom"))


def build_summary(
    products: Products, meta: RunMetadata, checks: Sequence[Check] | None = None
) -> dict[str, Any]:
    """Collect the numbers every report format shares.

    Args:
        products: Pipeline output.
        meta: Descriptive fields.
        checks: Optional validation checks (synthetic runs only).

    Returns:
        A JSON-serializable dictionary.
    """
    summary: dict[str, Any] = {
        "title": meta.title,
        "data": {"label": meta.data_label, "source": meta.source, "notes": meta.notes},
        "software": f"lidar_terrain_lab {__version__}",
        "grid": {
            "crs": _crs_label(products.grid.crs),
            "resolution_m": products.grid.res_m,
            "width": products.grid.width,
            "height": products.grid.height,
        },
        "statistics": asdict(products.stats),
        "vegetation_classes": [
            {**asdict(area), "label": veg.label, "color": veg.color}
            for veg, area in zip(products.classes, products.class_table, strict=True)
        ],
        "zones": [
            {
                "name": z.name,
                "total_acres": z.total_acres,
                "woody_acres": z.woody_acres,
                "treatment_acres": z.treatment_acres,
                "mean_slope_deg": z.mean_slope_deg,
                "class_acres": {a.name: a.acres for a in z.classes},
            }
            for z in products.zone_table
        ],
        "drainage_reaches": [
            {
                "strahler_order": s.strahler_order,
                "length_m": s.length_m,
                "contributing_area_ha": s.contributing_area_ha,
            }
            for s in products.hydrology.segments
        ],
    }
    if checks is not None:
        summary["validation"] = [asdict(c) for c in checks]
    return summary


def write_summary_json(summary: dict[str, Any], path: Path) -> Path:
    """Write the summary with rounded floats for readable diffs."""

    def rounded(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value

    path.write_text(json.dumps(rounded(summary), indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _key_numbers(summary: dict[str, Any]) -> Table:
    s = summary["statistics"]
    rows = [
        ["Area analyzed", f"{s['area_acres']:.1f} ac ({s['area_hectares']:.2f} ha)"],
        ["Cell size", f"{s['resolution_m']:g} m"],
        ["Points (noise removed)", f"{s['points_total'] - s['points_noise']:,}"],
        ["Point density", f"{s['point_density_per_m2']:.1f} per m2"],
        ["Ground method", s["ground_method"]],
        ["DEM cells interpolated", f"{s['dem_interpolated_percent']:.1f}%"],
        ["Elevation range", f"{s['elevation_min_m']:.1f} to {s['elevation_max_m']:.1f} m"],
        ["Mean slope", f"{s['slope_mean_deg']:.1f} deg"],
        ["Woody cover", f"{s['woody_acres']:.2f} ac ({s['woody_percent']:.1f}%)"],
        [
            f"Treatment acreage ({' + '.join(s['treatment_classes'])})",
            f"{s['treatment_acres']:.2f} ac",
        ],
        [
            "Drainage line length",
            f"{s['channel_length_m']:.0f} m in {s['channel_reaches']} reaches",
        ],
        ["Largest contributing area", f"{s['largest_contributing_area_ha']:.2f} ha"],
        ["Filled depressions deeper than 0.1 m", f"{s['sink_acres']:.2f} ac"],
    ]
    return ["Measure", "Value"], rows


def _class_table(summary: dict[str, Any]) -> Table:
    rows = [
        [c["label"], f"{c['acres']:.2f}", f"{c['hectares']:.2f}", f"{c['percent']:.1f}%"]
        for c in summary["vegetation_classes"]
    ]
    return ["Class", "Acres", "Hectares", "Share"], rows


def _zone_table(summary: dict[str, Any]) -> Table | None:
    zones = summary["zones"]
    if not zones:
        return None
    names = list(zones[0]["class_acres"])
    header = ["Zone", "Total ac", *[f"{n} ac" for n in names], "Treatment ac", "Mean slope"]
    rows = [
        [
            z["name"],
            f"{z['total_acres']:.2f}",
            *[f"{z['class_acres'][n]:.2f}" for n in names],
            f"{z['treatment_acres']:.2f}",
            f"{z['mean_slope_deg']:.1f} deg",
        ]
        for z in zones
    ]
    return header, rows


def _validation_table(summary: dict[str, Any]) -> Table | None:
    checks = summary.get("validation")
    if not checks:
        return None
    rows = []
    for c in checks:
        rows.append(
            [
                c["name"],
                f"{c['value']:.3f} {c['unit']}",
                f"{c['comparison']} {c['limit']:g}",
                "pass" if c["passed"] else "FAIL",
            ]
        )
    return ["Check", "Measured", "Limit", "Result"], rows


def _markdown_table(table: Table) -> str:
    header, rows = table
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _html_table(table: Table) -> str:
    header, rows = table
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _sections(summary: dict[str, Any]) -> list[tuple[str, str, Table | None]]:
    """(heading, lead paragraph, table) triples shared by both formats."""
    s = summary["statistics"]
    sections: list[tuple[str, str, Table | None]] = [
        ("Key numbers", "", _key_numbers(summary)),
        (
            "Vegetation height classes",
            "Each cell is classed by the height of whatever stands on it (canopy height = "
            "surface minus bare earth). Treatment acreage sums the classes configured as "
            "treatment targets.",
            _class_table(summary),
        ),
    ]
    zone_table = _zone_table(summary)
    if zone_table:
        sections.append(("By zone", "Cells are assigned to a zone by their center.", zone_table))
    shares = ", ".join(f"{k} deg: {v:.1f}%" for k, v in s["slope_share_percent"].items())
    sections.append(("Terrain", f"Share of area by slope class: {shares}.", None))
    sections.append(
        (
            "Drainage",
            f"Drainage lines are drawn where more than {s['channel_threshold_ha']:g} ha drains "
            f"through a cell. Highest Strahler order: {s['max_strahler_order']}. Deepest filled "
            f"depression: {s['max_sink_depth_m']:.2f} m.",
            None,
        )
    )
    validation = _validation_table(summary)
    if validation:
        sections.append(
            (
                "Validation against synthetic truth",
                "The synthetic scene's true ground, crowns and channel are known, so each "
                "product can be scored.",
                validation,
            )
        )
    return sections


def write_markdown(summary: dict[str, Any], maps: dict[str, Path], path: Path) -> Path:
    """Write the report as Markdown with images referenced relative to ``path``.

    Args:
        summary: Output of :func:`build_summary`.
        maps: Map name to PNG path.
        path: Output ``.md`` file.

    Returns:
        The written path.
    """
    data = summary["data"]
    parts = [
        f"# {summary['title']}",
        "",
        f"> **Data: {data['label']}.** {data['source']} {data['notes']}".rstrip(),
        "",
    ]
    for heading, lead, table in _sections(summary):
        parts += [f"## {heading}", ""]
        if lead:
            parts += [lead, ""]
        if table:
            parts += [_markdown_table(table), ""]
    parts += ["## Maps", ""]
    for name, map_path in maps.items():
        rel = map_path.relative_to(path.parent).as_posix()
        parts += [f"![{name.replace('_', ' ')}]({rel})", ""]
    parts.append(f"_Generated by {summary['software']}._")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    return path


_CSS = """
:root { --ink: #0b0b0b; --ink-2: #52514e; --rule: #dedcd5; --surface: #fcfcfb;
  --accent: #1c552d; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #f2f1ec; --ink-2: #c3c2b7; --rule: #3a3a37; --surface: #1a1a19;
    --accent: #8cbb5c; }
}
body { background: var(--surface); color: var(--ink); margin: 0;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.7rem; margin: 0 0 8px; }
h2 { font-size: 1.15rem; margin: 36px 0 8px; border-bottom: 1px solid var(--rule);
  padding-bottom: 4px; }
.note { color: var(--ink-2); border-left: 3px solid var(--accent); padding-left: 12px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: 8px 0; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); }
th { color: var(--ink-2); font-weight: 600; }
.table-wrap { overflow-x: auto; }
figure { margin: 20px 0; }
img { max-width: 100%; height: auto; border: 1px solid var(--rule); background: #fff; }
footer { color: var(--ink-2); font-size: 0.85rem; margin-top: 40px; }
"""


def write_html(summary: dict[str, Any], maps: dict[str, Path], path: Path) -> Path:
    """Write the report as a standalone HTML page (images referenced relative to ``path``).

    Args:
        summary: Output of :func:`build_summary`.
        maps: Map name to PNG path.
        path: Output ``.html`` file.

    Returns:
        The written path.
    """
    data = summary["data"]
    title = html.escape(summary["title"])
    body = [
        f"<h1>{title}</h1>",
        f'<p class="note"><strong>Data: {html.escape(data["label"])}.</strong> '
        f"{html.escape(data['source'])} {html.escape(data['notes'])}</p>",
    ]
    for heading, lead, table in _sections(summary):
        body.append(f"<h2>{html.escape(heading)}</h2>")
        if lead:
            body.append(f"<p>{html.escape(lead)}</p>")
        if table:
            body.append(f'<div class="table-wrap">{_html_table(table)}</div>')
    body.append("<h2>Maps</h2>")
    for name, map_path in maps.items():
        rel = html.escape(map_path.relative_to(path.parent).as_posix())
        label = html.escape(name.replace("_", " "))
        body.append(f'<figure><img src="{rel}" alt="{label} map" loading="lazy"></figure>')
    body.append(f"<footer>Generated by {html.escape(summary['software'])}.</footer>")
    page = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )
    path.write_text(page, encoding="utf-8", newline="\n")
    return path
