import shutil
from pathlib import Path
from collections import Counter

# Canonical AEGIS-CUAS Class Taxonomy
CLASS_NAMES = ["drone", "bird", "airplane", "helicopter", "missile"]

# Map DAOD subfolder names -> Canonical Class ID
DAOD_FOLDER_MAP = {
    "drones": 0,            # drone
    "uav": 0,               # drone
    "birds": 1,             # bird
    "airplane": 2,          # airplane
    "militery_airplane": 2, # airplane
    "helicopter": 3,        # helicopter
    "missiles": 4,          # missile
}

SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_unified_dataset():
    daod_root = Path("datasets/raw_daod/DAOD")
    multi_root = Path("datasets/raw_multiclass")
    out_root = Path("datasets/aerial/unified")

    stats = Counter()

    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Process DAOD (class determined by parent folder name)
    print("[1/2] Processing DAOD dataset...")
    for split_src, split_dst in SPLIT_MAP.items():
        img_split_dir = daod_root / "images" / split_src
        lbl_split_dir = daod_root / "labels" / split_src
        if not img_split_dir.exists():
            continue

        for class_folder in img_split_dir.iterdir():
            if not class_folder.is_dir():
                continue
            target_cls = DAOD_FOLDER_MAP.get(class_folder.name.lower())
            if target_cls is None:
                continue

            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                lbl_path = lbl_split_dir / class_folder.name / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                remapped_lines = []
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        remapped_lines.append(f"{target_cls} {' '.join(parts[1:])}")
                        stats[(split_dst, CLASS_NAMES[target_cls])] += 1

                if remapped_lines:
                    new_stem = f"daod_{class_folder.name}_{img_path.stem}"
                    shutil.copy2(img_path, out_root / "images" / split_dst / f"{new_stem}{img_path.suffix}")
                    (out_root / "labels" / split_dst / f"{new_stem}.txt").write_text("\n".join(remapped_lines) + "\n")

    # 2. Process raw_multiclass (all 5 classes 0..4 are drone types -> remap to 0: drone)
    print("[2/2] Processing Multi-Class Drone dataset...")
    for split_src, split_dst in SPLIT_MAP.items():
        img_split_dir = multi_root / "images" / split_src
        lbl_split_dir = multi_root / "labels" / split_src
        if not img_split_dir.exists():
            continue

        for img_path in img_split_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_split_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            remapped_lines = []
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    remapped_lines.append(f"0 {' '.join(parts[1:])}")  # 0 = drone
                    stats[(split_dst, "drone")] += 1

            if remapped_lines:
                new_stem = f"multi_{img_path.stem}"
                shutil.copy2(img_path, out_root / "images" / split_dst / f"{new_stem}{img_path.suffix}")
                (out_root / "labels" / split_dst / f"{new_stem}.txt").write_text("\n".join(remapped_lines) + "\n")

    # 3. Write unified data.yaml
    yaml_content = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (out_root / "data.yaml").write_text(yaml_content)

    print("\n=== Unified Dataset Ready ===")
    for (split, cls_name), count in sorted(stats.items()):
        print(f"  {split:<6} | {cls_name:<12} | {count} boxes")
    print(f"\nSaved config to: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    prepare_unified_dataset()