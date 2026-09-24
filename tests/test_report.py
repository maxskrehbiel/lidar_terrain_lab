"""Summary JSON, Markdown and HTML reports."""

from __future__ import annotations

import json
from pathlib import Path

from lidar_terrain_lab.pipeline import Products
from lidar_terrain_lab.report import (
    RunMetadata,
    _crs_label,
    build_summary,
    write_html,
    write_markdown,
    write_summary_json,
)
from lidar_terrain_lab.synthetic import SyntheticScene
from lidar_terrain_lab.validation import validate

META = RunMetadata("Test report", "SYNTHETIC", "Generated points.", "Not a real place.")


def test_summary_holds_every_section(products: Products, scene: SyntheticScene) -> None:
    summary = build_summary(products, META, validate(products, scene))
    assert summary["grid"]["crs"] == "EPSG:32631"
    assert summary["statistics"]["treatment_classes"] == ("Shrub", "Small tree")
    assert [c["name"] for c in summary["vegetation_classes"]] == [
        "Open",
        "Shrub",
        "Small tree",
        "Tree",
    ]
    assert [z["name"] for z in summary["zones"]] == ["Unit A", "Unit B", "Unit C"]
    assert len(summary["drainage_reaches"]) == len(products.hydrology.segments)
    assert summary["validation"][0]["key"] == "ground_recall"
    assert "validation" not in build_summary(products, META)


def test_summary_json_is_rounded_and_stable(products: Products, tmp_path: Path) -> None:
    summary = build_summary(products, META)
    first = write_summary_json(summary, tmp_path / "a.json").read_bytes()
    second = write_summary_json(build_summary(products, META), tmp_path / "b.json").read_bytes()
    assert first == second
    assert first.endswith(b"}\n") and b"\r\n" not in first
    stats = json.loads(first)["statistics"]
    assert all(len(repr(v).split(".")[-1]) <= 4 for v in stats.values() if isinstance(v, float))


def test_markdown_and_html_reports(
    products: Products, scene: SyntheticScene, tmp_path: Path
) -> None:
    summary = build_summary(products, META, validate(products, scene))
    maps = {"terrain": tmp_path / "maps" / "terrain.png"}
    markdown = write_markdown(summary, maps, tmp_path / "report.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Test report\n")
    assert "## Validation against synthetic truth" in markdown
    assert "| Unit A |" in markdown
    assert "![terrain](maps/terrain.png)" in markdown
    page = write_html(summary, maps, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<title>Test report</title>" in page
    assert 'src="maps/terrain.png"' in page
    assert page.count("<table>") >= 4
    for path in (tmp_path / "report.md", tmp_path / "report.html"):
        assert b"\r\n" not in path.read_bytes()


def test_crs_label() -> None:
    assert _crs_label(None) == "none"
    assert _crs_label("EPSG:32614") == "EPSG:32614"
    assert _crs_label("+proj=tmerc +lat_0=0 +lon_0=-97 +k=1 +x_0=0 +y_0=0 +ellps=GRS80") == "tmerc"
