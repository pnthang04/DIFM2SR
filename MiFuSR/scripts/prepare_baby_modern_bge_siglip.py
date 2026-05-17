#!/usr/bin/env python3
"""Prepare the Baby Modern BGE/SigLIP dataset for MiFuSR."""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np


DATA_NAME = "Baby_Modern_BGE_SigLIP"


def ensure_extracted(archive: Path, extracted_root: Path) -> Path:
    source_dir = extracted_root / "baby_modern"
    if source_dir.exists():
        return source_dir

    extracted_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted_root)
    return source_dir


def read_inter(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        header = fin.readline().strip().split("\t")
        user_idx = header.index("user_id:token")
        seq_idx = header.index("item_id_list:token_seq")
        item_idx = header.index("item_id:token")

        for line in fin:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(user_idx, seq_idx, item_idx):
                continue
            user = int(fields[user_idx])
            history = [int(x) for x in fields[seq_idx].split() if x]
            target = int(fields[item_idx])
            yield user, history, target


def longest_train_sequences(train_file: Path):
    sequences = {}
    for user, history, target in read_inter(train_file):
        seq = history + [target]
        if len(seq) > len(sequences.get(user, [])):
            sequences[user] = seq
    return sequences


def write_uirt(path: Path, rows):
    with path.open("w", encoding="utf-8") as fout:
        for user, item, rating, timestamp in rows:
            fout.write(f"{user}\t{item}\t{rating}\t{timestamp}\n")


def split_target_rows(inter_file: Path):
    for user, history, target in read_inter(inter_file):
        yield user, target, 1.0, len(history)


def write_map(path: Path, mapping):
    ordered = sorted(mapping.items(), key=lambda kv: kv[1])
    with path.open("w", encoding="utf-8") as fout:
        for token, idx in ordered:
            fout.write(f"{token}\t{idx}\n")


def prepare(archive: Path, output_dir: Path, extracted_root: Path):
    source_dir = ensure_extracted(archive, extracted_root)
    inter_dir = source_dir / "dataset" / "baby"
    target_dir = output_dir / DATA_NAME
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = target_dir / DATA_NAME
    train_sequences = longest_train_sequences(inter_dir / "baby.train.inter")
    train_rows = (
        (user, item, 1.0, timestamp)
        for user, seq in sorted(train_sequences.items())
        for timestamp, item in enumerate(seq)
    )

    write_uirt(prefix.with_suffix(".train"), train_rows)
    write_uirt(prefix.with_suffix(".valid"), split_target_rows(inter_dir / "baby.valid.inter"))
    write_uirt(prefix.with_suffix(".test"), split_target_rows(inter_dir / "baby.test.inter"))

    with (source_dir / "user2id.json").open("r", encoding="utf-8") as fin:
        user2id = json.load(fin)
    with (source_dir / "item2id.json").open("r", encoding="utf-8") as fin:
        item2id = json.load(fin)

    write_map(prefix.with_suffix(".user2id"), user2id)
    write_map(prefix.with_suffix(".item2id"), item2id)

    img_features = np.load(source_dir / "image_features_siglip.npy")
    txt_features = np.load(source_dir / "text_features_bge.npy")
    np.savez_compressed(prefix.with_suffix(".img.npz"), img_features)
    np.savez_compressed(prefix.with_suffix(".txt.npz"), txt_features)

    with prefix.with_suffix(".info").open("w", encoding="utf-8") as fout:
        fout.write(f"name={DATA_NAME}\n")
        fout.write(f"users={len(user2id)}\n")
        fout.write(f"items={len(item2id)}\n")
        fout.write(f"train_users={len(train_sequences)}\n")
        fout.write(f"image_features={img_features.shape}\n")
        fout.write(f"text_features={txt_features.shape}\n")

    return target_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("dataset/_archives/baby_modern_bge_siglip.tar.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--extracted-root", type=Path, default=Path("dataset/_extracted"))
    args = parser.parse_args()

    target_dir = prepare(args.archive, args.output_dir, args.extracted_root)
    print(target_dir)


if __name__ == "__main__":
    main()
