"""
download_datasets.py — minimal data acquisition to CSV

Supported datasets:
  - sst2  (GLUE/SST-2)
  - ag_news
  - imdb
  - tweeteval-sentiment

Examples:
  python download_datasets.py --dataset sst2 --split all --outdir data --with_label_text
  python download_datasets.py --dataset ag_news --split train --num_samples 50000 --outdir data
"""
import argparse, os, json, random
from typing import Dict, Any, List, Optional

import pandas as pd
from datasets import load_dataset


CONFIGS: Dict[str, Dict[str, Any]] = {
    "sst2": {
        "hf": "glue",
        "config": "sst2",
        "text_col": "sentence",
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"},
    },
    "ag_news": {
        "hf": "ag_news",
        "config": None,
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    },
    "imdb": {
        "hf": "imdb",
        "config": None,
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"},
    },
    "tweeteval-sentiment": {
        "hf": "tweet_eval",
        "config": "sentiment",
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
    },
}


def save_split_to_csv(ds, split: str, cfg: Dict[str, Any], outdir: str,
                      num_samples: int = -1, with_label_text: bool = False) -> str:
    """
    Save one split to CSV with columns: text, label, (optional) label_text
    Returns the output file path.
    """
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")
    dset = ds[split]

    if num_samples is not None and num_samples > 0:
        dset = dset.select(range(min(num_samples, len(dset))))

    text_col = cfg["text_col"]
    label_col = cfg["label_col"]

    # Build DataFrame explicitly to avoid bringing extra columns
    df = pd.DataFrame({
        "text": dset[text_col],
        "label": dset[label_col],
    })
    if with_label_text and "label_map" in cfg and cfg["label_map"]:
        df["label_text"] = df["label"].map(cfg["label_map"])

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{args.dataset}_{split}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def main(args):
    if args.dataset not in CONFIGS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: {list(CONFIGS.keys())}")
    cfg = CONFIGS[args.dataset]

    # Load the dataset from HF
    if cfg["config"]:
        ds = load_dataset(cfg["hf"], cfg["config"])
    else:
        ds = load_dataset(cfg["hf"])

    # If split='all', enumerate available splits; else use the provided split
    splits: List[str]
    if args.split == "all":
        splits = list(ds.keys())
    else:
        splits = [args.split]

    saved = []
    for sp in splits:
        out_path = save_split_to_csv(
            ds, sp, cfg, args.outdir, num_samples=args.num_samples, with_label_text=args.with_label_text
        )
        saved.append(out_path)

    # Write a tiny manifest
    manifest = {
        "dataset": args.dataset,
        "splits": splits,
        "outdir": args.outdir,
        "files": saved,
        "num_samples": args.num_samples,
        "with_label_text": args.with_label_text,
    }
    with open(os.path.join(args.outdir, f"{args.dataset}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved files:", *saved, sep="\n - ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(CONFIGS.keys()),
                        help="Which dataset to download and save as CSV.")
    parser.add_argument("--split", default="all",
                        help="Dataset split: train/validation/test or 'all' (default).")
    parser.add_argument("--outdir", default="data", help="Output directory for CSV files.")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Optional cap on number of rows per split (e.g., 10000).")
    parser.add_argument("--with_label_text", action="store_true",
                        help="Include human-readable label_text column when available.")
    args = parser.parse_args()
    main(args)
