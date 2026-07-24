from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_KINDS = {
    "manifest": "blind_tile_boundary_roi_manifest",
    "validation": "blind_tile_boundary_roi_validation",
    "scores": "blind_tile_boundary_roi_scores",
    "reveal": "blind_tile_boundary_roi_reveal",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path, expected_kind: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == 1, f"Unsupported schema: {path}")
    require(payload.get("kind") == expected_kind, f"Unexpected kind: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(values: list[str], source: str) -> set[str]:
    duplicates = sorted(key for key, count in Counter(values).items() if count > 1)
    require(not duplicates, f"Duplicate IDs in {source}: {duplicates}")
    return set(values)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_summary(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "complete", f"Incomplete benchmark: {path}")
    rounds = payload.get("rounds")
    require(isinstance(rounds, list) and rounds, f"No benchmark rounds: {path}")

    first_round: dict[tuple[int, str], dict[str, Any]] = {}
    stable_tiles: dict[tuple[int, str], int] = {}
    stable_hashes: dict[tuple[int, str], str] = {}
    for round_report in rounds:
        for page in round_report["pages"]:
            key = (int(page["index"]), str(page["route"]))
            tile = int(page["selected_tile"])
            pixel_hash = str(page["pixel_sha256"])
            if key not in stable_tiles:
                stable_tiles[key] = tile
                stable_hashes[key] = pixel_hash
            require(stable_tiles[key] == tile, f"Tile drift for {key}: {path}")
            require(stable_hashes[key] == pixel_hash, f"Pixel drift for {key}: {path}")
            if int(round_report["round"]) == 1:
                require(key not in first_round, f"Duplicate first-round page {key}: {path}")
                first_round[key] = page
    require(len(first_round) == 6, f"Expected six first-round pages: {path}")
    return first_round, sha256_file(path)


def verify_source_crops(
    crop_checks: dict[Path, list[tuple[Path, tuple[int, int, int, int], str]]]
) -> int:
    verified = 0
    for source_path, checks in crop_checks.items():
        with Image.open(source_path) as source:
            source.load()
            for crop_path, box, description in checks:
                with Image.open(crop_path) as actual:
                    actual.load()
                    expected = source.crop(box)
                    require(
                        actual.mode == expected.mode
                        and actual.size == expected.size
                        and actual.tobytes() == expected.tobytes(),
                        f"Blind crop does not match revealed source: {description}",
                    )
                    verified += 1
    return verified


def audit_comparison(root: Path, expected_candidate: str) -> dict[str, Any]:
    root = root.resolve()
    paths = {
        "manifest": root / "blind-manifest.json",
        "validation": root / "validation-report.json",
        "scores": root / "blind-scores-before-reveal.json",
        "reveal": root / "mapping-reveal.json",
    }
    manifest = read_json(paths["manifest"], EXPECTED_KINDS["manifest"])
    validation = read_json(paths["validation"], EXPECTED_KINDS["validation"])
    scores = read_json(paths["scores"], EXPECTED_KINDS["scores"])
    reveal = read_json(paths["reveal"], EXPECTED_KINDS["reveal"])
    require(scores.get("mapping_opened_before_scoring") is False, "Scoring was not blind")

    manifest_rois = manifest.get("rois")
    require(isinstance(manifest_rois, list), "Manifest rois must be a list")
    manifest_ids = unique_ids([str(roi["id"]) for roi in manifest_rois], "manifest")
    require(len(manifest_ids) == 60, "Manifest must contain exactly 60 ROI IDs")
    manifest_by_id = {str(roi["id"]): roi for roi in manifest_rois}

    validation_hashes = validation.get("roi_hashes")
    require(isinstance(validation_hashes, dict), "Validation roi_hashes must be an object")
    validation_ids = set(validation_hashes)
    require(validation_ids == manifest_ids, "Validation and manifest ROI IDs differ")
    require(validation.get("inventory") == manifest.get("inventory"), "Inventory drift")
    require(manifest.get("counts") == manifest["inventory"].get("counts"), "Count drift")

    annotation_path = Path(str(manifest["annotations"])).resolve()
    annotation_hash = sha256_file(annotation_path)
    require(annotation_hash == validation["annotations_sha256"], "Annotation hash drift")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation_rois = annotations.get("rois")
    require(isinstance(annotation_rois, list), "Annotation rois must be a list")
    annotation_ids = unique_ids(
        [str(roi["id"]) for roi in annotation_rois], "annotations"
    )
    require(annotation_ids == manifest_ids, "Annotation and manifest ROI IDs differ")
    annotations_by_id = {str(roi["id"]): roi for roi in annotation_rois}

    roi_files_verified = 0
    for roi_id, record in manifest_by_id.items():
        annotation = annotations_by_id[roi_id]
        require(int(record["page_index"]) == int(annotation["page_index"]), f"Page drift: {roi_id}")
        require(record["route"] == annotation["route"], f"Route drift: {roi_id}")
        require(record["category"] == annotation["category"], f"Category drift: {roi_id}")
        require(record["input_box"] == annotation["box"], f"Box drift: {roi_id}")
        for side, key in (("A", "a"), ("B", "b")):
            crop_path = Path(str(record[f"{key}_path"])).resolve()
            actual_hash = sha256_file(crop_path)
            require(actual_hash == record[f"{key}_sha256"], f"ROI hash drift: {roi_id}-{side}")
            require(
                actual_hash == validation_hashes[roi_id][side],
                f"ROI validation hash drift: {roi_id}-{side}",
            )
            roi_files_verified += 1

    sheets = manifest.get("contact_sheets")
    require(isinstance(sheets, list) and len(sheets) == 6, "Expected six contact sheets")
    validation_sheets = validation.get("contact_sheet_hashes")
    require(isinstance(validation_sheets, dict), "Missing contact-sheet hashes")
    sheet_keys = unique_ids(
        [f"{sheet['category']}-{int(sheet['part']):02d}" for sheet in sheets],
        "contact sheets",
    )
    require(sheet_keys == set(validation_sheets), "Contact-sheet inventory drift")
    for sheet in sheets:
        key = f"{sheet['category']}-{int(sheet['part']):02d}"
        actual_hash = sha256_file(Path(str(sheet["path"])).resolve())
        require(actual_hash == sheet["sha256"], f"Contact-sheet hash drift: {key}")
        require(actual_hash == validation_sheets[key], f"Contact validation drift: {key}")

    mappings = reveal.get("mapping")
    require(isinstance(mappings, list), "Reveal mapping must be a list")
    mapping_ids = unique_ids([str(item["id"]) for item in mappings], "reveal")
    require(mapping_ids == manifest_ids, "Reveal and manifest ROI IDs differ")
    mapping_by_id = {str(item["id"]): item for item in mappings}

    configurations = reveal.get("configurations")
    require(isinstance(configurations, dict) and len(configurations) == 2, "Need two configurations")
    config_labels = list(configurations)
    require("baseline-o32" in configurations, "Missing baseline-o32")
    require(expected_candidate in configurations, f"Missing {expected_candidate}")
    require(set(config_labels) == {"baseline-o32", expected_candidate}, "Unexpected configurations")

    summary_pages: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    summary_hashes: dict[str, str] = {}
    source_hash_cache: dict[Path, str] = {}
    for label, output_root_text in configurations.items():
        output_root = Path(str(output_root_text)).resolve()
        summary_path = output_root.parent / "batch_summary.json"
        pages, actual_summary_hash = load_summary(summary_path)
        expected_summary_hash = reveal["benchmark_summary_sha256"][label]
        require(actual_summary_hash == expected_summary_hash, f"Summary hash drift: {label}")
        summary_pages[label] = pages
        summary_hashes[label] = actual_summary_hash

    expected_validation_summaries = {
        f"configuration_{index}": summary_hashes[label]
        for index, label in enumerate(config_labels, start=1)
    }
    require(
        validation.get("source_summary_sha256") == expected_validation_summaries,
        "Validation and reveal summary hashes differ",
    )
    annotation_summary_hashes = set(annotations.get("source_summary_sha256", {}).values())
    require(
        set(summary_hashes.values()).issubset(annotation_summary_hashes),
        "Annotations do not identify both benchmark summaries",
    )

    crop_checks: dict[
        Path, list[tuple[Path, tuple[int, int, int, int], str]]
    ] = defaultdict(list)
    for roi_id in sorted(manifest_ids):
        record = manifest_by_id[roi_id]
        mapping = mapping_by_id[roi_id]
        require({mapping["A"], mapping["B"]} == set(config_labels), f"Bad A/B mapping: {roi_id}")
        key = (int(record["page_index"]), str(record["route"]))
        box = tuple(int(value) for value in record["output_box"])
        require(len(box) == 4, f"Bad output box: {roi_id}")
        for side, path_key in (("A", "a_path"), ("B", "b_path")):
            label = str(mapping[side])
            output_root = Path(str(configurations[label])).resolve()
            expected_source = output_root / f"{key[0]:02d}_{key[1]}.png"
            source_path = Path(str(mapping[f"source_{side.lower()}"])).resolve()
            require(source_path == expected_source, f"Source path drift: {roi_id}-{side}")
            summary_page = summary_pages[label].get(key)
            require(summary_page is not None, f"Summary page missing: {roi_id}-{side}")
            require(
                int(mapping[f"{side}_tile"]) == int(summary_page["selected_tile"]),
                f"Tile mapping drift: {roi_id}-{side}",
            )
            if source_path not in source_hash_cache:
                source_hash_cache[source_path] = sha256_file(source_path)
            require(
                source_hash_cache[source_path] == summary_page["png_sha256"],
                f"Source PNG hash drift: {roi_id}-{side}",
            )
            crop_path = Path(str(record[path_key])).resolve()
            crop_checks[source_path].append((crop_path, box, f"{roi_id}-{side}"))

    crop_mappings_verified = verify_source_crops(crop_checks)
    require(crop_mappings_verified == 120, "Expected 120 source-to-crop checks")

    ties = [str(roi_id) for roi_id in scores.get("ties", [])]
    preferences = scores.get("preferences")
    require(isinstance(preferences, list), "Preferences must be a list")
    preference_ids = [str(item["roi"]) for item in preferences]
    tie_ids = unique_ids(ties, "score ties")
    preference_id_set = unique_ids(preference_ids, "score preferences")
    require(not tie_ids.intersection(preference_id_set), "A scored ROI appears twice")
    require(tie_ids.union(preference_id_set) == manifest_ids, "Score and manifest ROI IDs differ")

    summary = scores.get("summary", {})
    require(int(summary.get("roi_count", -1)) == 60, "Score ROI count drift")
    require(int(summary.get("tie_count", -1)) == len(ties), "Tie count drift")
    prefer_counts = Counter(str(item["preference"]) for item in preferences)
    require(set(prefer_counts).issubset({"A", "B"}), "Invalid anonymous preference")
    require(int(summary.get("prefer_a_count", -1)) == prefer_counts["A"], "A preference drift")
    require(int(summary.get("prefer_b_count", -1)) == prefer_counts["B"], "B preference drift")
    artifact_counts = {
        side: sum(bool(item[f"{side.lower()}_boundary_artifact"]) for item in preferences)
        for side in ("A", "B")
    }
    require(
        int(summary.get("a_boundary_artifact_count", -1)) == artifact_counts["A"],
        "A artifact count drift",
    )
    require(
        int(summary.get("b_boundary_artifact_count", -1)) == artifact_counts["B"],
        "B artifact count drift",
    )

    preferences_by_config = Counter({label: 0 for label in config_labels})
    artifacts_by_config = Counter({label: 0 for label in config_labels})
    preference_details: list[dict[str, Any]] = []
    for preference in preferences:
        roi_id = str(preference["roi"])
        mapping = mapping_by_id[roi_id]
        anonymous_preference = str(preference["preference"])
        preferred_config = str(mapping[anonymous_preference])
        preferences_by_config[preferred_config] += 1
        artifact_sides = [
            side
            for side in ("A", "B")
            if bool(preference[f"{side.lower()}_boundary_artifact"])
        ]
        artifact_configs = [str(mapping[side]) for side in artifact_sides]
        for label in artifact_configs:
            artifacts_by_config[label] += 1
        preference_details.append(
            {
                "roi": roi_id,
                "anonymous_preference": anonymous_preference,
                "preferred_configuration": preferred_config,
                "artifact_sides": artifact_sides,
                "artifact_configurations": artifact_configs,
                "reason": preference.get("reason"),
            }
        )

    candidate_artifacts = artifacts_by_config[expected_candidate]
    quality_pass = candidate_artifacts == 0
    if preferences_by_config[expected_candidate] and not candidate_artifacts:
        verdict = "candidate-preferred-with-no-observed-candidate-boundary-artifact"
    elif not preferences and not candidate_artifacts:
        verdict = "no-observed-candidate-regression"
    else:
        verdict = "candidate-boundary-artifact-observed"

    return {
        "root": root,
        "candidate": expected_candidate,
        "paths": paths,
        "quality_pass": quality_pass,
        "quality_verdict": verdict,
        "evidence_sha256": {
            "blind_manifest": sha256_file(paths["manifest"]),
            "validation_report": sha256_file(paths["validation"]),
            "scores_before_reveal": sha256_file(paths["scores"]),
            "mapping_reveal": sha256_file(paths["reveal"]),
            "annotations": annotation_hash,
            "benchmark_summaries": summary_hashes,
        },
        "integrity_validation": {
            "status": "passed",
            "manifest_roi_ids": len(manifest_ids),
            "annotation_roi_ids": len(annotation_ids),
            "validation_roi_ids": len(validation_ids),
            "score_roi_ids": len(tie_ids) + len(preference_id_set),
            "reveal_mapping_ids": len(mapping_ids),
            "roi_file_sha256_checks": roi_files_verified,
            "contact_sheet_sha256_checks": len(sheets),
            "benchmark_summary_sha256_checks": len(summary_hashes),
            "source_png_sha256_checks": len(source_hash_cache),
            "source_to_blind_crop_pixel_checks": crop_mappings_verified,
        },
        "revealed_outcome": {
            "ties": len(ties),
            "preferences_by_configuration": dict(preferences_by_config),
            "boundary_artifacts_by_configuration": dict(artifacts_by_config),
            "preference_details": preference_details,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and reveal the o32-vs-o24/o16 blind boundary quality gates."
    )
    parser.add_argument("o24_root", type=Path)
    parser.add_argument("o16_root", type=Path)
    args = parser.parse_args()

    audits = {
        "candidate-o24": audit_comparison(args.o24_root, "candidate-o24"),
        "candidate-o16": audit_comparison(args.o16_root, "candidate-o16"),
    }
    o24 = audits["candidate-o24"]
    o16 = audits["candidate-o16"]
    o16_artifacts = o16["revealed_outcome"]["boundary_artifacts_by_configuration"]

    if o16_artifacts["candidate-o16"] > 0:
        require(o24["quality_pass"], "o16 failed but o24 did not pass")
        selected = "candidate-o24"
        rule = "o16-has-observed-artifacts-select-passing-o24"
    elif o16_artifacts["baseline-o32"] > 0 and o16_artifacts["candidate-o16"] == 0:
        selected = "candidate-o16"
        rule = "o32-has-observed-artifacts-o16-is-better"
    else:
        raise ValueError("The requested candidate-selection rule is inconclusive")

    selection_evidence = {
        label: {
            "scores_before_reveal": audit["evidence_sha256"]["scores_before_reveal"],
            "mapping_reveal": audit["evidence_sha256"]["mapping_reveal"],
        }
        for label, audit in audits.items()
    }
    for label, audit in audits.items():
        candidate = audit["candidate"]
        result = {
            "schema_version": 1,
            "kind": "blind_tile_boundary_roi_result",
            "decision": (
                f"{candidate}-passes-six-page-boundary-quality-gate"
                if audit["quality_pass"]
                else f"{candidate}-fails-six-page-boundary-quality-gate"
            ),
            "quality_gate": {
                "pass": audit["quality_pass"],
                "verdict": audit["quality_verdict"],
            },
            "evidence_sha256": audit["evidence_sha256"],
            "integrity_validation": audit["integrity_validation"],
            "revealed_outcome": audit["revealed_outcome"],
            "candidate_selection": {
                "selected": selected,
                "this_candidate_selected": candidate == selected,
                "rule_applied": rule,
                "selection_evidence_sha256": selection_evidence,
            },
            "scope_limit": (
                "The conclusion covers 60 reviewed boundary-crossing ROIs on six pages, "
                "balanced across text, screentone, and diagonal categories. It establishes "
                "the observed gate result, not universal perceptual equivalence."
            ),
            "next_gate": "Do not start the 30-page run until the reveal result is reported.",
        }
        write_json(audit["root"] / "blind-result-after-reveal.json", result)

    print(
        json.dumps(
            {
                "status": "passed",
                "selected": selected,
                "selection_rule": rule,
                "comparisons": {
                    label: {
                        "quality_pass": audit["quality_pass"],
                        "revealed_outcome": audit["revealed_outcome"],
                        "integrity_validation": audit["integrity_validation"],
                    }
                    for label, audit in audits.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
