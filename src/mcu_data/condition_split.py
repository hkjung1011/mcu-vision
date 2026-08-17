from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .common import sha256_file


SPLITS = ("train", "val", "test")
CONDITION_POLICY_VERSION = "condition_group_stratified_by_model_sha256_v1"
PHASH_POLICY_VERSION = "phash_condition_component_model_balanced_sha256_v2"


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_new_or_identical(path: Path, content: str) -> None:
    """Write an artifact without ever replacing different existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") == content:
            return
        raise FileExistsError(f"Refusing to replace an existing artifact: {path}")
    path.write_text(content, encoding="utf-8", newline="\n")


def _ensure_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.samefile(destination):
            return
        raise FileExistsError(f"Refusing to replace an existing image: {destination}")
    os.link(source, destination)


def _allocate_counts(group_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if group_count < len(SPLITS):
        raise ValueError(
            f"At least {len(SPLITS)} condition groups are required per model; got {group_count}"
        )
    if set(ratios) != set(SPLITS):
        raise ValueError(f"Ratios must contain exactly: {', '.join(SPLITS)}")
    if any(not 0.0 < float(ratios[name]) < 1.0 for name in SPLITS):
        raise ValueError("Every split ratio must be greater than 0 and less than 1")
    if not math.isclose(sum(float(ratios[name]) for name in SPLITS), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")

    exact = {name: group_count * float(ratios[name]) for name in SPLITS}
    counts = {name: math.floor(exact[name]) for name in SPLITS}
    for name in sorted(SPLITS, key=lambda item: (-(exact[item] - counts[item]), SPLITS.index(item))):
        if sum(counts.values()) == group_count:
            break
        counts[name] += 1

    # A tiny stratum can otherwise receive no validation/test group. Move one group
    # from the largest allocation while preserving the exact total.
    for empty_name in (name for name in SPLITS if counts[name] == 0):
        donor = max(SPLITS, key=lambda name: (counts[name], -SPLITS.index(name)))
        if counts[donor] <= 1:
            raise ValueError(f"Cannot allocate every split for a {group_count}-group model")
        counts[donor] -= 1
        counts[empty_name] += 1
    return counts


def _stable_group_order(group_ids: Iterable[str], model: str, seed: int) -> list[str]:
    def key(group_id: str) -> tuple[str, str]:
        material = f"{CONDITION_POLICY_VERSION}\0{seed}\0{model}\0{group_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), group_id

    return sorted(group_ids, key=key)


def assign_condition_groups(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign complete condition groups, stratified independently for every model."""
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    if not rows:
        raise ValueError("Input manifest is empty")

    group_models: dict[str, set[str]] = defaultdict(set)
    groups_by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = str(row.get("condition_group_id", "")).strip()
        model = str(row.get("model", "")).strip()
        if not group_id or not model:
            raise ValueError("Every row must have non-empty condition_group_id and model fields")
        group_models[group_id].add(model)
        groups_by_model[model].add(group_id)

    invalid = {group_id: models for group_id, models in group_models.items() if len(models) != 1}
    if invalid:
        first_group = sorted(invalid)[0]
        raise ValueError(
            f"Condition group {first_group!r} spans multiple models: {sorted(invalid[first_group])}"
        )

    assignment: dict[str, str] = {}
    for model in sorted(groups_by_model):
        ordered = _stable_group_order(groups_by_model[model], model, seed)
        counts = _allocate_counts(len(ordered), ratios)
        cursor = 0
        for split in SPLITS:
            next_cursor = cursor + counts[split]
            assignment.update({group_id: split for group_id in ordered[cursor:next_cursor]})
            cursor = next_cursor
        if cursor != len(ordered):
            raise AssertionError("Internal split allocation error")
    return assignment


def build_phash_condition_components(
    rows: Sequence[Mapping[str, str]], duplicates_report: Path
) -> tuple[dict[str, str], dict[str, object]]:
    """Collapse condition groups joined by any audited near-duplicate image edge."""
    name_to_group: dict[str, str] = {}
    condition_groups: set[str] = set()
    for row in rows:
        name = Path(str(row["source_relative_path"])).name
        if name in name_to_group:
            raise ValueError(f"Duplicate image filename cannot be mapped safely: {name}")
        group_id = str(row["condition_group_id"])
        name_to_group[name] = group_id
        condition_groups.add(group_id)

    parent = {group_id: group_id for group_id in condition_groups}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Lexical root selection makes component IDs independent of input pair order.
        low, high = sorted((left_root, right_root))
        parent[high] = low

    payload = json.loads(duplicates_report.read_text(encoding="utf-8"))
    pairs = payload.get("near_duplicate_pairs")
    if not isinstance(pairs, list):
        raise ValueError("Duplicate report must contain a near_duplicate_pairs list")
    distance_counts: Counter[str] = Counter()
    for pair in pairs:
        left_name = Path(str(pair["left_path"])).name
        right_name = Path(str(pair["right_path"])).name
        missing = [name for name in (left_name, right_name) if name not in name_to_group]
        if missing:
            raise ValueError(
                f"Duplicate report does not match the manifest; missing image: {missing[0]}"
            )
        union(name_to_group[left_name], name_to_group[right_name])
        distance_counts[str(pair.get("distance", "UNKNOWN"))] += 1

    component_groups: dict[str, set[str]] = defaultdict(set)
    for group_id in sorted(condition_groups):
        component_groups[find(group_id)].add(group_id)
    group_to_component: dict[str, str] = {}
    for groups in component_groups.values():
        component_id = "phashcc_" + hashlib.sha256(
            "\n".join(sorted(groups)).encode("utf-8")
        ).hexdigest()[:16]
        for group_id in groups:
            group_to_component[group_id] = component_id

    component_sizes = Counter(group_to_component.values())
    audit: dict[str, object] = {
        "duplicates_report": duplicates_report.as_posix(),
        "duplicates_report_sha256": sha256_file(duplicates_report),
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_distance_counts": dict(sorted(distance_counts.items())),
        "phash_condition_components": len(component_sizes),
        "largest_phash_component_condition_groups": max(component_sizes.values(), default=0),
    }
    return group_to_component, audit


def assign_phash_components(
    rows: Sequence[Mapping[str, str]],
    group_to_component: Mapping[str, str],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign pHash components with exact per-model targets when a solution exists.

    Components that span models are conservatively placed in train. Remaining
    single-model components are solved with a deterministic 2-D subset dynamic
    program so validation and test reach their target group counts exactly.
    """
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    group_model: dict[str, str] = {}
    for row in rows:
        group_id = str(row["condition_group_id"])
        model = str(row["model"])
        existing = group_model.setdefault(group_id, model)
        if existing != model:
            raise ValueError(f"Condition group {group_id!r} spans multiple models")
        if group_id not in group_to_component:
            raise ValueError(f"Condition group missing from pHash components: {group_id}")

    components: dict[str, set[str]] = defaultdict(set)
    for group_id, component_id in group_to_component.items():
        components[component_id].add(group_id)

    model_groups: dict[str, set[str]] = defaultdict(set)
    for group_id, model in group_model.items():
        model_groups[model].add(group_id)
    targets = {model: _allocate_counts(len(groups), ratios) for model, groups in model_groups.items()}

    component_assignment: dict[str, str] = {}
    cross_model_components: list[str] = []
    for component_id, groups in components.items():
        if len({group_model[group_id] for group_id in groups}) > 1:
            component_assignment[component_id] = "train"
            cross_model_components.append(component_id)

    forced_train: Counter[str] = Counter()
    for component_id in cross_model_components:
        forced_train.update(group_model[group_id] for group_id in components[component_id])
    for model, count in forced_train.items():
        if count > targets[model]["train"]:
            raise ValueError(
                f"Cross-model pHash components require {count} train groups for {model}, "
                f"exceeding target {targets[model]['train']}"
            )

    for model in sorted(model_groups):
        candidates = [
            component_id
            for component_id, groups in components.items()
            if component_id not in component_assignment
            and {group_model[group_id] for group_id in groups} == {model}
        ]

        def component_key(component_id: str) -> tuple[str, str]:
            material = f"{PHASH_POLICY_VERSION}\0{seed}\0{model}\0{component_id}".encode("utf-8")
            return hashlib.sha256(material).hexdigest(), component_id

        candidates.sort(key=component_key)
        val_target = targets[model]["val"]
        test_target = targets[model]["test"]
        # state -> tuple of choices for processed components. First arrival wins,
        # yielding a stable solution for a fixed seed and manifest.
        states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
        for component_id in candidates:
            size = len(components[component_id])
            next_states: dict[tuple[int, int], tuple[str, ...]] = {}
            for (val_count, test_count), choices in states.items():
                for split in ("train", "val", "test"):
                    new_val = val_count + (size if split == "val" else 0)
                    new_test = test_count + (size if split == "test" else 0)
                    if new_val <= val_target and new_test <= test_target:
                        next_states.setdefault((new_val, new_test), choices + (split,))
            states = next_states
        selected = states.get((val_target, test_target))
        if selected is None:
            raise ValueError(
                f"No exact model-balanced pHash-component split exists for {model}; "
                "change ratios or review near-duplicate edges"
            )
        component_assignment.update(zip(candidates, selected, strict=True))

    if set(component_assignment) != set(components):
        raise AssertionError("Internal pHash component assignment error")
    group_assignment = {
        group_id: component_assignment[component_id]
        for group_id, component_id in group_to_component.items()
    }
    actual: Counter[str] = Counter(
        f"{group_assignment[group_id]}:{model}" for group_id, model in group_model.items()
    )
    for model, expected in targets.items():
        for split in SPLITS:
            if actual[f"{split}:{model}"] != expected[split]:
                raise AssertionError(f"Model balance failed for {model}/{split}")

    details: dict[str, object] = {
        "cross_model_phash_components": len(cross_model_components),
        "cross_model_phash_components_forced_to_train": len(cross_model_components),
        "model_target_condition_group_counts": {
            f"{split}:{model}": counts[split]
            for model, counts in sorted(targets.items())
            for split in SPLITS
        },
    }
    return group_assignment, details


def _validate_unique_output_names(rows: Sequence[Mapping[str, str]]) -> None:
    names: Counter[str] = Counter()
    stems: Counter[str] = Counter()
    for row in rows:
        name = Path(str(row["source_relative_path"])).name.lower()
        names[name] += 1
        stems[Path(name).stem] += 1
    duplicate_names = sorted(name for name, count in names.items() if count > 1)
    duplicate_stems = sorted(stem for stem, count in stems.items() if count > 1)
    if duplicate_names or duplicate_stems:
        problem = duplicate_names[0] if duplicate_names else duplicate_stems[0]
        raise ValueError(f"Image filename/stem collision would corrupt YOLO output: {problem}")


def _class_map(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    names = sorted({str(row.get("class_name", "")).strip() for row in rows})
    if not names or not names[0]:
        raise ValueError("Every row must have a non-empty class_name")
    return {name: class_id for class_id, name in enumerate(names)}


def _yolo_label(row: Mapping[str, str], class_id: int) -> str:
    width = int(row["width"])
    height = int(row["height"])
    left = int(row["bbox_left"])
    top = int(row["bbox_top"])
    box_width = int(row["bbox_width"])
    box_height = int(row["bbox_height"])
    values = (
        (left + box_width / 2) / width,
        (top + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid image/bbox dimensions for {row.get('sample_id', '<unknown>')}")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Out-of-range bbox for {row.get('sample_id', '<unknown>')}: {values}")
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in values) + "\n"


def materialize_yolo(
    rows: Sequence[Mapping[str, str]], source_root: Path, output_root: Path
) -> None:
    _validate_unique_output_names(rows)
    classes = _class_map(rows)
    for row in rows:
        split = str(row["yolo_split"])
        source = source_root / str(row["source_relative_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        image_name = source.name
        _ensure_hardlink(source, output_root / "images" / split / image_name)
        _write_new_or_identical(
            output_root / "labels" / split / f"{source.stem}.txt",
            _yolo_label(row, classes[str(row["class_name"])]),
        )

    names_yaml = "\n".join(f"  {class_id}: {name}" for name, class_id in classes.items())
    dataset_yaml = (
        "# Paths resolve from this file, so the dataset remains portable across Windows and Ubuntu.\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    _write_new_or_identical(output_root / "dataset.yaml", dataset_yaml)


def materialize_coco(
    rows: Sequence[Mapping[str, str]],
    source_root: Path,
    output_root: Path,
    *,
    policy: str,
) -> None:
    _validate_unique_output_names(rows)
    classes = _class_map(rows)
    categories = [
        {"id": class_id + 1, "name": name, "supercategory": "mcu_vision"}
        for name, class_id in classes.items()
    ]
    for split in SPLITS:
        split_rows = sorted(
            (row for row in rows if row["yolo_split"] == split),
            key=lambda row: str(row["sample_id"]),
        )
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        for image_id, row in enumerate(split_rows, start=1):
            source = source_root / str(row["source_relative_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            _ensure_hardlink(source, output_root / f"{split}2017" / source.name)
            box_width = int(row["bbox_width"])
            box_height = int(row["bbox_height"])
            images.append(
                {
                    "id": image_id,
                    "file_name": source.name,
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )
            annotations.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": classes[str(row["class_name"])] + 1,
                    "bbox": [
                        int(row["bbox_left"]),
                        int(row["bbox_top"]),
                        box_width,
                        box_height,
                    ],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
        coco = {
            "info": {
                "description": "Condition-group-independent MCU vision split",
                "split_policy": policy,
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "CC BY 4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                }
            ],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        _write_new_or_identical(
            output_root / "annotations" / f"instances_{split}2017.json",
            json.dumps(coco, ensure_ascii=False, indent=2) + "\n",
        )


def _build_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    ratios: Mapping[str, float],
    input_manifest: Path,
    output_manifest: Path,
    assignment: Mapping[str, str],
    policy: str,
    phash_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    groups_by_split = {
        split: {str(row["condition_group_id"]) for row in rows if row["yolo_split"] == split}
        for split in SPLITS
    }
    overlap = {
        f"{left}_{right}": len(groups_by_split[left] & groups_by_split[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    model_image_counts: Counter[str] = Counter(
        f"{row['yolo_split']}:{row['model']}" for row in rows
    )
    model_group_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        model_group_sets[f"{row['yolo_split']}:{row['model']}"].add(
            str(row["condition_group_id"])
        )
    physical_groups_by_split = {
        split: {
            str(row.get("physical_item_group", "NOT_VERIFIED"))
            for row in rows
            if row["yolo_split"] == split
            and str(row.get("physical_item_group", "NOT_VERIFIED")) != "NOT_VERIFIED"
        }
        for split in SPLITS
    }
    physical_ids_complete = all(
        str(row.get("physical_item_group", "NOT_VERIFIED")) != "NOT_VERIFIED" for row in rows
    )
    physical_overlap = {
        f"{left}_{right}": len(physical_groups_by_split[left] & physical_groups_by_split[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    physical_independence_verified = physical_ids_complete and all(
        value == 0 for value in physical_overlap.values()
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "policy": policy,
        "seed": seed,
        "ratios": {name: float(ratios[name]) for name in SPLITS},
        "input_manifest": input_manifest.as_posix(),
        "input_manifest_sha256": sha256_file(input_manifest),
        "output_manifest": output_manifest.as_posix(),
        "output_manifest_sha256": sha256_file(output_manifest),
        "assignment_sha256": _sha256_json(sorted(assignment.items())),
        "total_images": len(rows),
        "total_condition_groups": len(assignment),
        "split_image_counts": dict(Counter(str(row["yolo_split"]) for row in rows)),
        "split_condition_group_counts": {
            split: len(groups_by_split[split]) for split in SPLITS
        },
        "model_split_image_counts": dict(sorted(model_image_counts.items())),
        "model_split_condition_group_counts": {
            key: len(value) for key, value in sorted(model_group_sets.items())
        },
        "condition_group_overlap": overlap,
        "condition_group_overlap_pass": all(value == 0 for value in overlap.values()),
        "physical_item_group_overlap": physical_overlap,
        "physical_item_independence_verified": physical_independence_verified,
        "evaluation_scope": (
            "Condition-group independence is verified. Physical-item independence remains NOT VERIFIED "
            "because the source dataset does not provide specimen identity."
            if not physical_independence_verified
            else "Condition-group and physical-item identifiers are present; audit their split overlap separately."
        ),
        "original_files_deleted": False,
    }
    if phash_audit is not None:
        summary["phash_audit"] = dict(phash_audit)
    return summary


def _audit_phash_assignment(
    rows: Sequence[Mapping[str, str]],
    duplicates_report: Path,
    group_to_component: Mapping[str, str],
) -> dict[str, object]:
    by_name = {
        Path(str(row["source_relative_path"])).name: row
        for row in rows
    }
    payload = json.loads(duplicates_report.read_text(encoding="utf-8"))
    pairs = payload["near_duplicate_pairs"]
    pair_counts: Counter[str] = Counter()
    cross_split_pairs = 0
    for pair in pairs:
        left = by_name[Path(str(pair["left_path"])).name]
        right = by_name[Path(str(pair["right_path"])).name]
        left_split = str(left["yolo_split"])
        right_split = str(right["yolo_split"])
        pair_counts["__".join(sorted((left_split, right_split)))] += 1
        cross_split_pairs += left_split != right_split

    component_splits: dict[str, set[str]] = defaultdict(set)
    component_groups: dict[str, set[str]] = defaultdict(set)
    component_images: Counter[str] = Counter()
    component_models: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        component_id = group_to_component[str(row["condition_group_id"])]
        component_splits[component_id].add(str(row["yolo_split"]))
        component_groups[component_id].add(str(row["condition_group_id"]))
        component_images[component_id] += 1
        component_models[component_id].add(str(row["model"]))
    leaking_components = [
        component_id for component_id, splits in component_splits.items() if len(splits) > 1
    ]
    return {
        "near_duplicate_pair_split_counts": dict(sorted(pair_counts.items())),
        "cross_split_near_duplicate_pairs": cross_split_pairs,
        "cross_split_near_duplicate_pairs_pass": cross_split_pairs == 0,
        "phash_component_count": len(component_splits),
        "cross_split_phash_components": len(leaking_components),
        "cross_split_phash_components_pass": not leaking_components,
        "largest_phash_component_condition_groups": max(map(len, component_groups.values())),
        "largest_phash_component_images": max(component_images.values()),
        "cross_model_phash_components": sum(
            len(models) > 1 for models in component_models.values()
        ),
    }


def build_condition_split(
    input_manifest: Path,
    output_manifest: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    source_root: Path | None = None,
    yolo_output_root: Path | None = None,
    coco_output_root: Path | None = None,
    duplicates_report: Path | None = None,
) -> dict[str, object]:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {
        "sample_id",
        "model",
        "class_name",
        "condition_group_id",
        "source_relative_path",
        "yolo_split",
        "width",
        "height",
        "bbox_left",
        "bbox_top",
        "bbox_width",
        "bbox_height",
    }
    missing = sorted(required - set(input_fields))
    if missing:
        raise ValueError(f"Input manifest is missing required fields: {', '.join(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique")

    phash_metadata: dict[str, object] | None = None
    if duplicates_report is not None:
        group_to_component, component_audit = build_phash_condition_components(
            rows, duplicates_report
        )
        assignment, assignment_audit = assign_phash_components(
            rows, group_to_component, seed=seed, ratios=ratios
        )
        policy = PHASH_POLICY_VERSION
        split_unit = "phash_condition_component"
        phash_metadata = {**component_audit, **assignment_audit}
    else:
        assignment = assign_condition_groups(rows, seed=seed, ratios=ratios)
        group_to_component = {
            str(row["condition_group_id"]): str(row["condition_group_id"]) for row in rows
        }
        policy = CONDITION_POLICY_VERSION
        split_unit = "condition_group_id"
    class_ids = _class_map(rows)
    for row in rows:
        row["previous_yolo_split"] = row["yolo_split"]
        row["yolo_split"] = assignment[row["condition_group_id"]]
        row["class_id"] = str(class_ids[row["class_name"]])
        row["split_seed"] = str(seed)
        row["split_unit"] = split_unit
        row["split_policy"] = policy
        row["leakage_component_id"] = group_to_component[row["condition_group_id"]]

    extra_fields = [
        "previous_yolo_split",
        "class_id",
        "split_seed",
        "split_unit",
        "split_policy",
        "leakage_component_id",
    ]
    output_fields = input_fields + [field for field in extra_fields if field not in input_fields]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_manifest.exists():
        raise FileExistsError(f"Refusing to replace an existing manifest: {output_manifest}")
    with output_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if yolo_output_root is not None or coco_output_root is not None:
        if source_root is None:
            raise ValueError("--source-root is required when materializing YOLO or COCO data")
        source_root = source_root.resolve()
    if yolo_output_root is not None:
        materialize_yolo(rows, source_root, yolo_output_root)
    if coco_output_root is not None:
        materialize_coco(rows, source_root, coco_output_root, policy=policy)

    if duplicates_report is not None:
        if phash_metadata is None:
            raise AssertionError("Internal pHash audit error")
        phash_metadata.update(
            _audit_phash_assignment(rows, duplicates_report, group_to_component)
        )

    summary = _build_summary(
        rows,
        seed=seed,
        ratios=ratios,
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        assignment=assignment,
        policy=policy,
        phash_audit=phash_metadata,
    )
    summary_path = output_manifest.with_suffix(".summary.json")
    _write_new_or_identical(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a model-balanced train/val/test split with zero condition-group overlap"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--yolo-output-root", type=Path)
    parser.add_argument("--coco-output-root", type=Path)
    parser.add_argument(
        "--duplicates-report",
        type=Path,
        help="Optional audit duplicates.json; pHash-connected condition groups stay atomic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_condition_split(
        args.input_manifest,
        args.output_manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        source_root=args.source_root,
        yolo_output_root=args.yolo_output_root,
        coco_output_root=args.coco_output_root,
        duplicates_report=args.duplicates_report,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
