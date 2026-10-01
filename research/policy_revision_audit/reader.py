"""Read merged table headers and uncertainty markers, with source locators.

Exploratory constrained readers designed after viewing these notices, not a
learned NLP model or a general eligibility interpreter.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def expand(cells):
    grid = {}
    for cell in cells:
        for r in range(cell["row"], cell["row"] + cell["rowspan"]):
            for c in range(cell["col"], cell["col"] + cell["colspan"]):
                if (r, c) in grid:
                    raise ValueError("Overlapping table cells")
                grid[r, c] = cell
    return grid


def cell(text, row, col, rowspan=1, colspan=1):
    text = compact(text)
    return dict(
        text=text,
        sha256=sha(text.encode()),
        row=row,
        col=col,
        rowspan=rowspan,
        colspan=colspan,
    )


def read_hwp(path):
    executable = shutil.which("hwp5proc")
    if not executable:
        raise RuntimeError("Install pyhwp and put hwp5proc on PATH")
    xml = subprocess.run(
        [executable, "xml", str(path)], capture_output=True, check=True
    ).stdout
    root = ET.fromstring(xml)
    parents = {child: parent for parent in root.iter() for child in parent}
    text = compact("".join(n.text or "" for n in root.iter("Text")))
    tables = []
    for index, table in enumerate(root.iter("TableControl")):
        own = []
        for item in table.iter("TableCell"):
            parent = parents[item]
            while parent.tag != "TableControl":
                parent = parents[parent]
            if parent is table:
                own.append(
                    cell(
                        "".join(n.text or "" for n in item.iter("Text")),
                        **{
                            k: int(item.attrib[k])
                            for k in ["row", "col", "rowspan", "colspan"]
                        }
                    )
                )
        table_text = compact("".join(n.text or "" for n in table.iter("Text")))
        start = text.find(table_text)
        unit = None
        for m in re.finditer(
            r"단위[:：]([^)]{0,25})", text[max(0, start - 250) : start]
        ):
            if "천원" in m.group(1):
                unit = dict(
                    multiplier_won=1000,
                    start=max(0, start - 250) + m.start(),
                    end=max(0, start - 250) + m.end(),
                )
        tables.append(
            dict(
                index=index,
                cells=own,
                start=start,
                end=start + len(table_text),
                unit=unit,
            )
        )
    return dict(text=text, tables=tables)


def read_html(path):
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")
    content = soup.select_one(".board_content_area")
    if content is None:
        raise ValueError("Missing official notice body")
    tables = []
    for index, table in enumerate(content.find_all("table")):
        cells, occupied = [], set()
        for r, row in enumerate(
            tr for tr in table.find_all("tr") if tr.find_parent("table") is table
        ):
            c = 0
            for item in row.find_all(["td", "th"], recursive=False):
                while (r, c) in occupied:
                    c += 1
                value = cell(
                    item.get_text("", strip=True),
                    r,
                    c,
                    int(item.get("rowspan", 1)),
                    int(item.get("colspan", 1)),
                )
                cells.append(value)
                for rr in range(r, r + value["rowspan"]):
                    for cc in range(c, c + value["colspan"]):
                        if (rr, cc) in occupied:
                            raise ValueError("Overlapping HTML table cells")
                        occupied.add((rr, cc))
                c += value["colspan"]
        tables.append(dict(index=index, cells=cells, unit=None))
    return dict(text=compact(content.get_text("", strip=True)), tables=tables)


def load(cache):
    result = {}
    for source in json.loads((ROOT / "sources.json").read_text()):
        path = Path(cache) / source["cache_file"]
        if sha(path.read_bytes()) != source["sha256"]:
            raise ValueError("Source hash changed: " + source["id"])
        if source["format"] == "hwp":
            result[source["id"]] = read_hwp(path)
        elif source["format"] == "html":
            result[source["id"]] = read_html(path)
        else:
            import fitz

            with fitz.open(path) as pdf:
                result[source["id"]] = dict(
                    text=compact("".join(p.get_text(sort=True) for p in pdf)), tables=[]
                )
    return result


def locate(document, table, value):
    return dict(
        document=document,
        table=table["index"],
        row=value["row"],
        col=value["col"],
        rowspan=value["rowspan"],
        colspan=value["colspan"],
        sha256=value["sha256"],
    )


def award_tables(document):
    return [
        t
        for t in document["tables"]
        if {"전국대회", "국제대회"} <= {c["text"] for c in t["cells"]}
    ]


def read_awards(table, document):
    """Preserve competition/group/rank paths; never infer a missing header."""
    grid = expand(table["cells"])
    records = []
    for (r, c), amount in sorted(grid.items()):
        if r < 3 or c == 0 or not re.fullmatch(r"\d[\d,]*", amount["text"]):
            continue
        headers = [grid.get((rr, c)) for rr in range(3)]
        rank_cell = grid.get((r, 0))
        if any(h is None for h in headers) or rank_cell is None:
            continue
        competition, group = headers[0]["text"], headers[1]["text"]
        if competition not in {"전국대회", "국제대회"}:
            continue
        if group == "단체":
            group += headers[2]["text"]
        if group not in {"개인", "단체2~4명", "단체5명이상"}:
            continue
        rank = re.search(r"([123])위", rank_cell["text"])
        if rank is None:
            continue
        evidence = [locate(document, table, x) for x in [*headers, rank_cell, amount]]
        number = int(amount["text"].replace(",", ""))
        records.append(
            dict(
                competition=competition,
                group=group,
                rank=int(rank[1]),
                raw_amount=number,
                amount_thousand_won=(
                    number
                    if table["unit"] and table["unit"]["multiplier_won"] == 1000
                    else None
                ),
                unit=table["unit"],
                evidence=evidence,
            )
        )
    keys = [(r["competition"], r["group"], r["rank"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Ambiguous header paths")
    return records


def lookup(records, competition, group, rank, method="merged_header_table_reader"):
    if not records or competition is None or group is None or rank is None:
        return None
    if method == "first_numeric_cell":
        return records[0]["amount_thousand_won"]
    if method == "rank_only_first_numeric_cell":
        selected = [r for r in records if r["rank"] == rank]
    else:
        selected = [
            r
            for r in records
            if (r["competition"], r["group"], r["rank"]) == (competition, group, rank)
        ]
    return selected[0]["amount_thousand_won"] if selected else None


def modality(text):
    """Constrained lexical probe; ambiguous/unsupported language stays unknown."""
    t = compact(text)
    if "제외대상이될수" in t and "확인" in t:
        return dict(
            state="needs_confirmation",
            may_exclude=True,
            definite_exclusion=False,
            reason="possibility_and_confirmation",
            named_example_added="댐주변지역학생장학금" in t,
        )
    if "제외대상이아닙니다" in t or "제외하지않" in t:
        return dict(
            state="not_excluded_by_this_clause",
            may_exclude=False,
            definite_exclusion=False,
            reason="explicit_negative",
            named_example_added=False,
        )
    if "지원대상에서제외합니다" in t:
        return dict(
            state="excluded_by_this_clause",
            may_exclude=False,
            definite_exclusion=True,
            reason="explicit_exclusion",
            named_example_added=False,
        )
    return dict(
        state="unknown",
        may_exclude=None,
        definite_exclusion=None,
        reason="unsupported_expression",
        named_example_added=False,
    )
