import argparse
import ast
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_line(line: bytes):
    s = line.decode("utf-8", errors="ignore").strip()
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def load_reviews(path):
    rows = []
    with gzip.open(path, "rb") as f:
        for line in tqdm(f, desc=f"Reading {path}"):
            if not line.strip():
                continue
            obj = parse_line(line)
            user = obj.get("reviewerID")
            item = obj.get("asin")
            ts = obj.get("unixReviewTime")
            rating = obj.get("overall", 1.0)
            if user is None or item is None or ts is None:
                continue
            rows.append((user, item, int(ts), float(rating)))
    return rows


def iterative_k_core(rows, min_user=5, min_item=5):
    rows = list(rows)

    while True:
        user_count = defaultdict(int)
        item_count = defaultdict(int)

        for u, i, _, _ in rows:
            user_count[u] += 1
            item_count[i] += 1

        new_rows = [
            r for r in rows
            if user_count[r[0]] >= min_user and item_count[r[1]] >= min_item
        ]

        if len(new_rows) == len(rows):
            break

        rows = new_rows

    return rows


def build_sequences(rows):
    user_hist = defaultdict(list)
    for u, i, ts, rating in rows:
        user_hist[u].append((ts, i, rating))

    sequences = {}
    for u, hist in user_hist.items():
        hist = sorted(hist, key=lambda x: x[0])
        seq = [i for _, i, _ in hist]
        if len(seq) >= 3:
            sequences[u] = seq

    return sequences


def remap_ids(sequences):
    users = sorted(sequences.keys())
    items = sorted({i for seq in sequences.values() for i in seq})

    user2id = {u: idx for idx, u in enumerate(users)}
    item2id = {i: idx for idx, i in enumerate(items)}

    remapped = {}
    for u, seq in sequences.items():
        remapped[user2id[u]] = [item2id[i] for i in seq]

    return remapped, user2id, item2id


def write_inter_files(remapped, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{name}.train.inter"
    valid_path = out_dir / f"{name}.valid.inter"
    test_path = out_dir / f"{name}.test.inter"

    header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(valid_path, "w", encoding="utf-8") as f_valid, \
         open(test_path, "w", encoding="utf-8") as f_test:

        f_train.write(header)
        f_valid.write(header)
        f_test.write(header)

        for u, seq in remapped.items():
            if len(seq) < 3:
                continue

            train_seq = seq[:-2]
            valid_target = seq[-2]
            test_target = seq[-1]

            # Train: tạo nhiều prefix-target pairs
            for idx in range(1, len(train_seq)):
                hist = train_seq[:idx]
                target = train_seq[idx]
                f_train.write(f"{u}\t{' '.join(map(str, hist))}\t{target}\n")

            # Valid: dùng toàn bộ train_seq để dự đoán item áp chót
            f_valid.write(f"{u}\t{' '.join(map(str, train_seq))}\t{valid_target}\n")

            # Test: dùng train_seq + valid_target để dự đoán item cuối
            test_hist = train_seq + [valid_target]
            f_test.write(f"{u}\t{' '.join(map(str, test_hist))}\t{test_target}\n")


def load_metadata(meta_path, kept_asins):
    meta = {}

    with gzip.open(meta_path, "rb") as f:
        for line in tqdm(f, desc=f"Reading {meta_path}"):
            if not line.strip():
                continue

            obj = parse_line(line)
            asin = obj.get("asin")
            if asin not in kept_asins:
                continue

            title = obj.get("title", "")
            brand = obj.get("brand", "")

            categories = obj.get("categories", [])
            if isinstance(categories, list):
                flat_cats = []
                for c in categories:
                    if isinstance(c, list):
                        flat_cats.extend(c)
                    else:
                        flat_cats.append(str(c))
                categories = " ".join(flat_cats)
            else:
                categories = str(categories)

            desc = obj.get("description", "")
            if isinstance(desc, list):
                desc = " ".join(map(str, desc))
            else:
                desc = str(desc)

            text = f"Title: {title}. Brand: {brand}. Category: {categories}. Description: {desc}"
            meta[asin] = text

    return meta


def save_item_text(meta_path, item2id, out_path):
    kept_asins = set(item2id.keys())
    meta = load_metadata(meta_path, kept_asins)

    id2item = {v: k for k, v in item2id.items()}

    with open(out_path, "w", encoding="utf-8") as f:
        for item_id in range(len(id2item)):
            asin = id2item[item_id]
            text = meta.get(asin, "")
            f.write(json.dumps({
                "item_id": item_id,
                "asin": asin,
                "text": text
            }, ensure_ascii=False) + "\n")


def read_image_features_b(image_path, item2id, out_npy):
    """
    McAuley image_features_*.b format:
    10 bytes ASIN + 4096 float32
    """
    dim = 4096
    num_items = len(item2id)
    feats = np.zeros((num_items, dim), dtype=np.float32)

    found = 0
    with open(image_path, "rb") as f:
        pbar = tqdm(desc=f"Reading {image_path}")
        while True:
            asin_bytes = f.read(10)
            if not asin_bytes:
                break

            asin = asin_bytes.decode("utf-8", errors="ignore")
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                break

            if asin in item2id:
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                feats[item2id[asin]] = vec
                found += 1

            pbar.update(1)

        pbar.close()

    np.save(out_npy, feats)
    print(f"Saved image features: {out_npy}")
    print(f"Found image features for {found}/{num_items} kept items")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="baby or office")
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--image-features", required=True)
    parser.add_argument("--out-dataset", required=True)
    parser.add_argument("--out-processed", required=True)
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_processed, exist_ok=True)

    rows = load_reviews(args.reviews)
    print("Raw interactions:", len(rows))

    rows = iterative_k_core(rows, args.min_user, args.min_item)
    print("After k-core interactions:", len(rows))

    sequences = build_sequences(rows)
    print("Users after sequence build:", len(sequences))

    remapped, user2id, item2id = remap_ids(sequences)

    print("Final users:", len(user2id))
    print("Final items:", len(item2id))

    write_inter_files(remapped, args.out_dataset, args.name)

    with open(Path(args.out_processed) / "user2id.json", "w", encoding="utf-8") as f:
        json.dump(user2id, f, ensure_ascii=False)

    with open(Path(args.out_processed) / "item2id.json", "w", encoding="utf-8") as f:
        json.dump(item2id, f, ensure_ascii=False)

    save_item_text(
        args.meta,
        item2id,
        Path(args.out_processed) / "item_text.jsonl"
    )

    read_image_features_b(
        args.image_features,
        item2id,
        Path(args.out_processed) / "image_features.npy"
    )

    print("Done.")


if __name__ == "__main__":
    main()