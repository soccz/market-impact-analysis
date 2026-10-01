"""Replay public decisions, source locators and immutable first predictions."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evaluate import parsed
from infer import ROOT, digest
from report import enriched, read_predictions


def load(name):
    return json.loads((ROOT / name).read_text())


def references(name):
    return [{**r, "sentences": {k: "" for k in r["sentence_ids"]}} for r in load(name)]


def verify(cache=None):
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_scope.py", "test_extension.py"],
        cwd=ROOT,
        check=True,
    )
    for p in ROOT.glob("*freeze.json"):
        for name, expected in json.loads(p.read_text())["files"].items():
            assert digest((ROOT / name).read_bytes()) == expected, (p.name, name)
    dev = load("development.json")
    comp = load("composition.json")
    official = references("official_reference.json")
    replication = references("replication_reference.json")
    followup = references("followup_reference.json")
    supports = load("development_support.json")["support_sets"]
    first = {}
    for split, rows in [
        ("development", dev),
        ("composition", comp),
        ("official", official),
    ]:
        preds, _ = read_predictions(rows, ROOT / "results" / split)
        if split == "official":
            for v in ["base", "distractor", "removed"]:
                first[split + "_" + v] = enriched(
                    [r for r in rows if r.get("variant", "base") == v], preds
                )
        else:
            first[split] = enriched(
                rows, preds, supports if split == "development" else None
            )
    assert first == load("results/case_scores.json")
    assert {
        k: {m: s["metrics"] for m, s in v.items()} for k, v in first.items()
    } == load("results/summary.json")
    from extension_report import METHODS

    checked = 5
    for stage, splits, names in [
        (
            "extension",
            [
                ("development", dev),
                ("official", official),
                ("replication", replication),
            ],
            METHODS,
        ),
        (
            "thinking",
            [("replication", replication), ("followup", followup)],
            ["qwen-budget-control", "qwen-thinking"],
        ),
        (
            "sampling",
            [("followup", followup)],
            ["qwen-sampled-control", "qwen-sampled-thinking"],
        ),
    ]:
        groups = {}
        for split, rows in splits:
            preds = {}
            for name in names:
                folder = ROOT / "results" / stage / split / name
                if split == "development" and name == "qwen-baseline":
                    folder = ROOT / "results/development" / name
                preds[name] = {
                    r["id"]: parsed(
                        json.loads((folder / (r["id"] + ".json")).read_text()), r
                    )
                    for r in rows
                }
            for v in (
                ["base", "distractor", "removed"]
                if split in ["official", "replication"]
                else [None]
            ):
                selected = (
                    rows
                    if v is None
                    else [r for r in rows if r.get("variant", "base") == v]
                )
                key = split if v is None else split + "_" + v
                groups[key] = enriched(
                    selected, preds, supports if split == "development" else None
                )
        assert groups == load("results/" + stage + "/case_scores.json"), stage
        assert {
            k: {m: s["metrics"] for m, s in v.items()} for k, v in groups.items()
        } == load("results/" + stage + "/summary.json"), stage
        checked += len(groups)
    old = 0
    for folder in [
        "policy_reading_v3",
        "policy_reading_v4",
        "policy_reading_v5",
        "policy_semantic_diff",
        "policy_condition_reading",
        "policy_condition_transfer",
        "policy_revision_audit",
    ]:
        p = ROOT.parent / folder / "research_record.json"
        for name, expected in json.loads(p.read_text())["artifacts"].items():
            if isinstance(expected, str):
                assert digest((p.parent / name).read_bytes()) == expected, (
                    folder,
                    name,
                )
                old += 1
    from citation_report import metrics as citation_metrics, METHODS as CITATION_METHODS

    citation_groups = {}
    for split, rows in [("official", official), ("replication", replication)]:
        records = {}
        for method in CITATION_METHODS:
            stage = "citation" if method.endswith("citation-only") else "extension"
            records[method] = {
                r["id"]: json.loads(
                    (
                        ROOT / "results" / stage / split / method / (r["id"] + ".json")
                    ).read_text()
                )
                for r in rows
            }
        for variant in ["base", "distractor", "removed"]:
            citation_groups[split + "_" + variant] = citation_metrics(
                [r for r in rows if r.get("variant", "base") == variant], records
            )
    assert citation_groups == load("results/citation/case_scores.json")
    assert {
        g: {m: s["metrics"] for m, s in methods.items()}
        for g, methods in citation_groups.items()
    } == load("results/citation/summary.json")
    checked += len(citation_groups)
    from class_diagnostics import run as class_report
    from annotation_sensitivity import run as sensitivity_report

    for name, runner in [
        ("class_diagnostics.json", class_report),
        ("annotation_sensitivity.json", sensitivity_report),
    ]:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / name
            runner(path)
            assert path.read_bytes() == (ROOT / "results" / name).read_bytes(), name
    locators = 0
    replayed = 0
    if cache:
        cache = Path(cache)
        from build_official import build

        rebuilt = build(cache)
        assert rebuilt == json.loads((cache / "official_inputs.json").read_text())
        assert (
            digest((cache / "official_inputs.json").read_bytes())
            == load("test_freeze.json")["official_private_input_sha256"]
        )
        from build_replication import build as replication_build

        rows, refs, _ = replication_build(cache)
        assert refs == load("replication_reference.json")
        assert (
            digest((cache / "replication_inputs.json").read_bytes())
            == load("replication_freeze.json")["private_input_sha256"]
        )
        from build_followup import build as followup_build

        following, refs = followup_build(cache)
        assert refs == load("followup_reference.json")
        assert (
            digest((cache / "followup_inputs.json").read_bytes())
            == load("followup_freeze.json")["private_input_sha256"]
        )
        for row in rebuilt + rows + following:
            for key, locator in row.get("source_locators", {}).items():
                assert digest(row["sentences"][key].encode()) == locator["sha256"]
                locators += 1
        from report import run as first_run
        from extension_report import run as extension_run
        from thinking_report import run as thinking_run
        from sampling_report import run as sampling_run

        from repeat_report import run as repeat_run
        from citation_report import run as citation_run

        for stage, runner in [
            ("", first_run),
            ("extension", extension_run),
            ("thinking", thinking_run),
            ("sampling", sampling_run),
            ("repeat", repeat_run),
            ("citation", citation_run),
        ]:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
                io.StringIO()
            ):
                runner(cache, tmp)
                for path in Path(tmp).rglob("*.json"):
                    assert (
                        path.read_bytes()
                        == (
                            ROOT / "results" / stage / path.relative_to(tmp)
                        ).read_bytes()
                    ), str(path)
                    replayed += 1
        from audit_requests import audit

        requests_audit = audit(cache)
        assert requests_audit == load("results/request_audit.json")
    manifest = ROOT / "research_record.json"
    artifacts = {}
    if manifest.exists():
        artifacts = json.loads(manifest.read_text())["artifacts"]
        actual = {
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
            and p != manifest
        }
        assert actual == set(artifacts), "Public artifact inventory differs"
        for name, expected in artifacts.items():
            assert digest((ROOT / name).read_bytes()) == expected, name
    result = dict(
        success=True,
        behavioral_tests=17,
        recomputed_score_groups=checked,
        previous_artifacts_unchanged=old,
        private_source_locators=locators,
        byte_replayed_files=replayed,
        independent_human_reviews=0,
        public_artifacts_checked=len(artifacts),
    )
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    verify(p.parse_args().cache)
