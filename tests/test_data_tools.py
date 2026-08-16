from __future__ import annotations

from pathlib import Path

from PIL import Image

from mcu_data.audit import audit_dataset
from mcu_data.micropcb import prepare_dataset
from mcu_data.wikimedia import keyword_allowed, license_allowed


def test_license_allowlist_is_prefix_based() -> None:
    allowed = ["CC0", "CC BY", "Public domain"]
    assert license_allowed("CC BY-SA 4.0", allowed)
    assert license_allowed("Public domain", allowed)
    assert not license_allowed("All rights reserved", allowed)


def test_keyword_filters() -> None:
    config = {"include_any": ["nucleo", "discovery"], "exclude_any": ["logo"]}
    assert keyword_allowed("File:STM32 Nucleo F401RE.jpg", config)
    assert not keyword_allowed("File:STM32 logo.svg", config)
    assert not keyword_allowed("File:Generic board.jpg", config)


def test_audit_detects_duplicate_pixels(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    class_root = data_root / "wikimedia" / "raspberry_pi_sbc"
    class_root.mkdir(parents=True)
    image = Image.new("RGB", (640, 480), color=(20, 80, 120))
    image.save(class_root / "one.png")
    image.save(class_root / "two.png")
    classes = tmp_path / "classes.yaml"
    classes.write_text("classes:\n  raspberry_pi_sbc: {}\n", encoding="utf-8")

    summary = audit_dataset(
        data_root=data_root,
        classes_config=classes,
        report_root=tmp_path / "reports",
        min_width=320,
        min_height=240,
        phash_threshold=4,
        maximum_near_pairs=100,
    )

    assert summary["scanned_files"] == 2
    assert summary["decode_failures"] == 0
    assert summary["exact_pixel_duplicate_groups"] == 1


def test_prepare_micropcb_filters_rpi_and_writes_yolo(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split, suffix in (("train", "1"), ("test", "5")):
        image_root = source / f"{split}_coded" / f"{split}_coded"
        image_root.mkdir(parents=True)
        for code in ("A", "B", "H", "I"):
            Image.new("RGB", (100, 80), color=(20, 80, 120)).save(image_root / f"{code}AAA{suffix}.jpg")
        (source / f"{split}_sizes.csv").write_text(
            "Image,Width,Height\n" + "".join(f"{code}AAA{suffix}.jpg,100,80\n" for code in ("A", "B", "H", "I")),
            encoding="utf-8",
        )
        (source / f"{split}_bboxes.csv").write_text(
            "Image,Left,Top,Width,Height\n"
            + "".join(f"{code}AAA{suffix}.jpg,10,8,50,40\n" for code in ("A", "B", "H", "I")),
            encoding="utf-8",
        )
        (source / f"{split}_angles.csv").write_text(
            "Image,Angle\n" + "".join(f"{code}AAA{suffix}.jpg,0.0\n" for code in ("A", "B", "H", "I")),
            encoding="utf-8",
        )
        (source / f"{split}_ratio_top_to_bottom.csv").write_text(
            "Image,Ratio of Top to Bottom Edge Length\n"
            + "".join(f"{code}AAA{suffix}.jpg,1.0\n" for code in ("A", "B", "H", "I")),
            encoding="utf-8",
        )

    summary = prepare_dataset(source, tmp_path / "out", tmp_path / "manifest.csv")

    assert summary["total_images"] == 6
    assert summary["split_counts"] == {"train": 3, "val": 3}
    assert (tmp_path / "out" / "labels" / "train" / "AAAA1.txt").read_text(encoding="utf-8").startswith("0 ")
    assert not (tmp_path / "out" / "images" / "train" / "BAAA1.jpg").exists()
