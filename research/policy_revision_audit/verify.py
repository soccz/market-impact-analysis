"""Recreate all results and verify source, cells, units and previous artifacts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from reader import ROOT, load, sha
from run import run


def verify(cache):
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_reader.py"], cwd=ROOT, check=True
    )
    for record in ["source_lock.json", "research_record.json"]:
        path = ROOT / record
        if path.exists():
            data = json.loads(path.read_text())
            for name, expected in data.get("files", data.get("artifacts", {})).items():
                assert sha((ROOT / name).read_bytes()) == expected, name
    docs = load(cache)
    for source in json.loads((ROOT / "sources.json").read_text()):
        assert source["notice_number"] in docs[source["id"]]["text"], source["id"]
    parsed = json.loads((ROOT / "results/parsed_tables.json").read_text())
    checked = 0
    for rows in parsed.values():
        for row in rows:
            for evidence in row["evidence"]:
                table = docs[evidence["document"]]["tables"][evidence["table"]]
                matches = [
                    c
                    for c in table["cells"]
                    if c["row"] == evidence["row"] and c["col"] == evidence["col"]
                ]
                assert len(matches) == 1 and matches[0]["sha256"] == evidence["sha256"]
                checked += 1
            unit = row["unit"]
            text = docs[row["evidence"][0]["document"]]["text"]
            assert "천원" in text[unit["start"] : unit["end"]]
    summary = json.loads((ROOT / "results/summary.json").read_text())
    assert summary["changed"] == 14 and summary["unchanged"] == 4
    assert summary["provenance"]["current_prior_and_correction_hwp_identical"]
    assert summary["family_modality"]["reference_match"]
    for evidence in summary["family_modality"]["evidence"]:
        text = docs[evidence["document"]]["text"][evidence["start"] : evidence["end"]]
        assert sha(text.encode()) == evidence["sha256"]
    for data in json.loads(
        (ROOT / "results/automatic_diffs.json").read_text()
    ).values():
        for change in data["changes"]:
            for side in ["before", "after"]:
                text = docs[data[side]]["text"][
                    change[side + "_start"] : change[side + "_end"]
                ]
                assert sha(text.encode()) == change[side + "_sha256"]
    with tempfile.TemporaryDirectory() as directory:
        run(cache, directory)
        for name in [
            "summary.json",
            "cases.json",
            "parsed_tables.json",
            "automatic_diffs.json",
            "source_text_hashes.json",
        ]:
            assert (Path(directory) / name).read_bytes() == (
                ROOT / "results" / name
            ).read_bytes(), name
    count = 0
    for folder in [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
        "policy_condition_reading",
        "policy_condition_transfer",
    ]:
        path = ROOT.parent / folder / "research_record.json"
        for name, expected in json.loads(path.read_text())["artifacts"].items():
            if isinstance(expected, str):
                assert sha((path.parent / name).read_bytes()) == expected, (
                    folder,
                    name,
                )
                count += 1
    print(
        json.dumps(
            dict(
                success=True,
                source_files=7,
                behavioral_tests=7,
                table_header_cell_locators=checked,
                replayed_results=5,
                previous_artifacts_unchanged=count,
                independent_human_reviews=0,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    verify(parser.parse_args().cache)
