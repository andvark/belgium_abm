"""Empty CSV templates for the REAL data that has to be supplied (status: template)."""

from __future__ import annotations

from pathlib import Path

from belgium_abm.io.schemas import TABLE_SCHEMAS

TEMPLATE_DATASET_IDS = {t: f"REAL_TEMPLATE_{s.default_dataset_id}" for t, s in TABLE_SCHEMAS.items()}


def template_text(table: str) -> str:
    s = TABLE_SCHEMAS[table]
    header = [
        f"# dataset_id: {TEMPLATE_DATASET_IDS[table]}",
        "# synthetic_demo_only: false",
        "# status: template",
        "# reference_year: null",
        "# unit: persons" if table == "population" else "# unit: see description",
        "# source: FILL IN (real dataset, release and table id) -- this file is an EMPTY TEMPLATE",
        f"# description: {s.description}",
    ]
    cols = ",".join(name for name, _ in s.columns)
    return "\n".join([*header, cols]) + "\n"


def write_templates(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for table, schema in TABLE_SCHEMAS.items():
        p = directory / f"{schema.default_dataset_id}.csv"
        p.write_text(template_text(table), encoding="utf-8", newline="\n")
        out.append(p)
    return out
