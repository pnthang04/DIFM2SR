import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch


REPO_ID = "thangkt/baby-modern-bge-siglip"
ARCHIVE_NAME = "baby_modern_siglip_large.tar.gz"
DEFAULT_ARCHIVE = Path("dataset/_archives") / ARCHIVE_NAME
DEFAULT_EXTRACT_DIR = Path("dataset/_extracted/baby_modern_siglip_large")
DEFAULT_OUTPUT_DIR = Path("dataset/Baby_Modern_SigLIP_Large")


def download_archive(archive: Path) -> None:
    if archive.exists():
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Archive is missing and huggingface_hub is not installed. "
            "Install huggingface_hub or place the archive at "
            f"{archive}."
        ) from exc

    archive.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=ARCHIVE_NAME,
        local_dir=str(archive.parent),
    )
    downloaded = Path(path)
    if downloaded.resolve() != archive.resolve():
        shutil.copy2(downloaded, archive)


def safe_extract(archive: Path, extract_dir: Path, force: bool) -> None:
    marker = extract_dir / ".complete"
    if marker.exists() and not force:
        return

    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        root = extract_dir.resolve()
        for member in tar.getmembers():
            target = (extract_dir / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)
    marker.write_text(str(archive.resolve()), encoding="utf-8")


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {root}")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches[:5])
        raise RuntimeError(f"Found multiple matches for {pattern}: {joined}")
    return matches[0]


def load_sequences(split_file: Path) -> dict[int, list[int]]:
    sequences = {}
    with split_file.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            user_raw, hist_raw, target_raw = line.rstrip("\n").split("\t")
            user = int(user_raw)
            hist = [int(item) for item in hist_raw.split()] if hist_raw else []
            target = int(target_raw)
            seq = hist + [target]
            if user not in sequences or len(seq) > len(sequences[user]):
                sequences[user] = seq
    return sequences


def write_mcd_inter(extracted_root: Path, output_dir: Path) -> tuple[int, int]:
    train_file = find_one(extracted_root, "*.train.inter")
    valid_file = find_one(extracted_root, "*.valid.inter")
    test_file = find_one(extracted_root, "*.test.inter")

    sequences = load_sequences(train_file)
    sequences.update(load_sequences(valid_file))
    sequences.update(load_sequences(test_file))

    output_dir.mkdir(parents=True, exist_ok=True)
    inter_path = output_dir / "baby_modern.inter"
    max_item_id = -1
    with inter_path.open("w", encoding="utf-8") as f:
        f.write("user item rating timestamp\n")
        for user in sorted(sequences):
            for ts, item in enumerate(sequences[user], start=1):
                max_item_id = max(max_item_id, item)
                f.write(f"{user} {item} 1.0 {ts}\n")

    num_users = max(sequences) + 1 if sequences else 0
    num_items = max_item_id + 1
    return num_users, num_items


def convert_feature(src: Path, dst: Path, expected_items: int) -> int:
    arr = np.load(src)
    if arr.ndim != 2:
        raise ValueError(f"{src} has shape {arr.shape}, expected a 2D array")
    if arr.shape[0] != expected_items:
        raise ValueError(
            f"{src} has {arr.shape[0]} rows, expected {expected_items} item rows"
        )
    tensor = torch.from_numpy(arr.astype(np.float32, copy=False))
    torch.save(tensor, dst)
    return arr.shape[1]


def copy_metadata(extracted_root: Path, output_dir: Path) -> None:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ["item2id.json", "user2id.json", "item_text.jsonl", "image_paths.jsonl"]:
        matches = sorted(extracted_root.rglob(name))
        if matches:
            shutil.copy2(matches[0], meta_dir / name)


def write_stats(output_dir: Path, stats: dict) -> None:
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def prepare(args: argparse.Namespace) -> None:
    download_archive(args.archive)
    safe_extract(args.archive, args.extract_dir, args.force)

    extracted_root = args.extract_dir
    output_dir = args.output_dir
    num_users, num_items = write_mcd_inter(extracted_root, output_dir)

    text_dim = convert_feature(
        find_one(extracted_root, "text_features_siglip_large.npy"),
        output_dir / "text_feat.pt",
        num_items,
    )
    image_dim = convert_feature(
        find_one(extracted_root, "image_features_siglip_large.npy"),
        output_dir / "image_feat.pt",
        num_items,
    )
    copy_metadata(extracted_root, output_dir)
    write_stats(
        output_dir,
        {
            "archive": str(args.archive),
            "num_users": num_users,
            "num_items": num_items,
            "text_dim": text_dim,
            "visual_dim": image_dim,
        },
    )

    print(f"prepared {output_dir}")
    print(f"num_users={num_users} num_items={num_items}")
    print(f"text_dim={text_dim} visual_dim={image_dim}")
    print("train command:")
    print(
        "python train_denoiser_main.py --benchmark Amazon --dataset baby_modern "
        "--lr_encoder 1e-4 --temperature 0.2 --exp_name baby_modern_siglip_large"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare thangkt/baby-modern-bge-siglip for MCD4SR."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
