import argparse
import shutil
import tarfile
from pathlib import Path

import numpy as np


DEFAULT_ARCHIVE = Path("dataset/baby_modern_bge_siglip.tar.gz")
DEFAULT_EXTRACT_DIR = Path("dataset/_extracted")
DEFAULT_OUTPUT_NAME = "baby_modern"


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {root}")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches[:5])
        raise RuntimeError(f"Found multiple matches for {pattern}: {joined}")
    return matches[0]


def copy_required_files(extracted_root: Path, output_dir: Path, output_name: str, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_sources = {
        "train": find_one(extracted_root, "*.train.inter"),
        "valid": find_one(extracted_root, "*.valid.inter"),
        "test": find_one(extracted_root, "*.test.inter"),
    }
    for split, src in split_sources.items():
        dst = output_dir / f"{output_name}.{split}.inter"
        if dst.exists() and not force:
            print(f"keep existing {dst}")
            continue
        shutil.copy2(src, dst)
        print(f"wrote {dst}")

    feature_sources = {
        "text": find_one(extracted_root, "text_features_bge.npy"),
        "image": find_one(extracted_root, "image_features_siglip.npy"),
    }
    expected_dim = 768
    for suffix, src in feature_sources.items():
        dst = output_dir / f"{output_name}.{suffix}"
        arr = np.load(src, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(f"{src} has shape {arr.shape}, expected (*, {expected_dim})")
        if dst.exists() and not force:
            print(f"keep existing {dst} shape={arr.shape}")
            continue
        shutil.copy2(src, dst)
        print(f"wrote {dst} shape={arr.shape}")


def copy_metadata(extracted_root: Path, output_dir: Path, force: bool) -> None:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "item2id.json",
        "user2id.json",
        "item_text.jsonl",
        "image_paths.jsonl",
        "image_download_failed.jsonl",
    ]:
        matches = sorted(extracted_root.rglob(name))
        if not matches:
            continue
        dst = meta_dir / name
        if dst.exists() and not force:
            print(f"keep existing {dst}")
            continue
        shutil.copy2(matches[0], dst)
        print(f"wrote {dst}")


def extract_archive(archive: Path, extract_dir: Path, force: bool) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".complete"
    if marker.exists() and not force:
        print(f"using existing extraction {extract_dir}")
        return extract_dir

    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracting {archive} -> {extract_dir}")
    with tarfile.open(archive, "r:gz") as tar:
        extract_root = extract_dir.resolve()
        for member in tar.getmembers():
            target = (extract_dir / member.name).resolve()
            if extract_root != target and extract_root not in target.parents:
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)
    marker.write_text(str(archive.resolve()), encoding="utf-8")
    return extract_dir


def prepare(args: argparse.Namespace) -> Path:
    archive = args.archive.resolve()
    extract_dir = (args.extract_base / args.output_name).resolve()
    output_dir = (args.output_base / args.output_name).resolve()

    extracted_root = extract_archive(archive, extract_dir, args.force)
    copy_required_files(extracted_root, output_dir, args.output_name, args.force)
    copy_metadata(extracted_root, output_dir, args.force)

    print("")
    print(f"prepared dataset: {output_dir}")
    print(f"train command: python run.py -d {args.output_name}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract baby_modern_bge_siglip.tar.gz and prepare RecBole/SGP4SR files."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-base", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output-base", type=Path, default=Path("dataset"))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--force", action="store_true", help="Overwrite extracted/prepared files.")
    parser.add_argument("--run-train", action="store_true", help="Run SGP4SR training after preparation.")
    args = parser.parse_args()

    prepare(args)

    if args.run_train:
        from run import run

        run(args.output_name)


if __name__ == "__main__":
    main()
