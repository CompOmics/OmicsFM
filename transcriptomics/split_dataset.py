"""
Split the transcriptomics dataset by dataset_id into train/valid/test.

Reads expression.npz + metadata.parquet, splits by dataset_id using greedy
tissue-stratified balancing, and outputs per-split files:
  - train.npz, valid.npz, test.npz (sparse expression matrices)
  - train_metadata.parquet, valid_metadata.parquet, test_metadata.parquet

Usage:
    python transcriptomics/split_dataset.py \
      --expression transcriptomics/output/expression.npz \
      --metadata transcriptomics/output/metadata.parquet \
      --output-dir transcriptomics/split \
      --train-frac 0.9 --valid-frac 0.05 --test-frac 0.05
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import load_npz, save_npz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
sys.stderr.reconfigure(line_buffering=True)
log = logging.getLogger(__name__)


def greedy_stratified_split(meta, train_frac, valid_frac, test_frac, seed):
    """Assign each dataset to a split, stratified by tissue, greedy by cell deficit.

    Returns dict: dataset_id → split name.
    """
    total_cells = len(meta)
    targets = {
        "train": train_frac * total_cells,
        "valid": valid_frac * total_cells,
        "test": test_frac * total_cells,
    }
    counts = {"train": 0, "valid": 0, "test": 0}
    assignment = {}

    ds_info = (
        meta.groupby("dataset_id", observed=True)
        .agg(n_cells=("dataset_id", "size"), tissue=("tissue_general", "first"))
        .reset_index()
    )

    rng = np.random.RandomState(seed)
    ds_info["tissue"] = ds_info["tissue"].astype(str)
    tissues = sorted(ds_info["tissue"].unique())
    log.info(f"Splitting {len(ds_info)} datasets across {len(tissues)} tissues")

    # Phase 1: Reserve 1 dataset per tissue per split
    reserved = set()
    for tissue in tissues:
        tissue_ds = ds_info[ds_info["tissue"] == tissue].sort_values("n_cells")
        tissue_ids = tissue_ds["dataset_id"].values

        # Largest for train
        for ds_id in reversed(tissue_ids):
            if ds_id not in reserved:
                n = int(tissue_ds[tissue_ds["dataset_id"] == ds_id]["n_cells"].iloc[0])
                assignment[ds_id] = "train"
                counts["train"] += n
                reserved.add(ds_id)
                break

        # Smallest available for valid and test
        for split in ["valid", "test"]:
            for ds_id in tissue_ids:
                if ds_id not in reserved:
                    n = int(tissue_ds[tissue_ds["dataset_id"] == ds_id]["n_cells"].iloc[0])
                    assignment[ds_id] = split
                    counts[split] += n
                    reserved.add(ds_id)
                    break

    log.info(f"  Reserved {len(reserved)} datasets for tissue coverage")

    # Phase 2: Greedily assign remaining datasets by deficit
    for tissue in tissues:
        tissue_ds = ds_info[ds_info["tissue"] == tissue].copy()
        tissue_ds = tissue_ds[~tissue_ds["dataset_id"].isin(reserved)]
        tissue_ds = tissue_ds.sample(frac=1, random_state=rng.randint(2**31)).reset_index(drop=True)

        for _, row in tissue_ds.iterrows():
            ds_id = row["dataset_id"]
            n = row["n_cells"]

            deficits = {s: targets[s] - counts[s] for s in targets}
            best_split = max(deficits, key=deficits.get)

            assignment[ds_id] = best_split
            counts[best_split] += n

    # Log results
    for split in ["train", "valid", "test"]:
        n = counts[split]
        pct = n / total_cells * 100
        n_ds = sum(1 for s in assignment.values() if s == split)
        log.info(f"  {split}: {n:,} cells ({pct:.1f}%), {n_ds} datasets")

    # Verify tissue coverage
    for split in ["train", "valid", "test"]:
        split_ds_ids = {d for d, s in assignment.items() if s == split}
        split_tissues = ds_info[ds_info["dataset_id"].isin(split_ds_ids)]["tissue"].unique()
        missing = set(tissues) - set(split_tissues)
        if missing:
            log.warning(f"  {split} missing tissues: {missing}")
        else:
            log.info(f"  {split}: all {len(tissues)} tissues represented")

    return assignment


def main():
    parser = argparse.ArgumentParser(
        description="Split transcriptomics dataset",
        epilog="Use --config to read all settings from pipeline_config.yaml, "
               "or pass individual arguments.",
    )
    parser.add_argument("--config", help="Path to pipeline_config.yaml (reads output_dir, split, seed, compress_output)")
    parser.add_argument("--expression", help="Path to expression.npz")
    parser.add_argument("--metadata", help="Path to metadata.parquet")
    parser.add_argument("--output-dir", help="Output directory for split files")
    parser.add_argument("--train-frac", type=float)
    parser.add_argument("--valid-frac", type=float)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compress", action="store_true", help="Compress output npz files")
    args = parser.parse_args()

    # Load defaults from config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        split_cfg = cfg.get("split", {})
        build_output = cfg.get("output_dir", "transcriptomics/output")

        # Config-derived defaults (CLI args override)
        expression = args.expression or str(Path(build_output) / "expression.npz")
        metadata = args.metadata or str(Path(build_output) / "metadata.parquet")
        output_dir = Path(args.output_dir or split_cfg.get("output_dir", "transcriptomics/split"))
        train_frac = args.train_frac if args.train_frac is not None else split_cfg.get("train_frac", 0.9)
        valid_frac = args.valid_frac if args.valid_frac is not None else split_cfg.get("valid_frac", 0.05)
        test_frac = args.test_frac if args.test_frac is not None else split_cfg.get("test_frac", 0.05)
        seed = args.seed if args.seed is not None else cfg.get("seed", 42)
        compress = args.compress or cfg.get("compress_output", False)
    else:
        # All args required without config
        if not args.expression or not args.metadata or not args.output_dir:
            parser.error("--expression, --metadata, and --output-dir are required without --config")
        expression = args.expression
        metadata = args.metadata
        output_dir = Path(args.output_dir)
        train_frac = args.train_frac or 0.9
        valid_frac = args.valid_frac or 0.05
        test_frac = args.test_frac or 0.05
        seed = args.seed or 42
        compress = args.compress

    total_frac = train_frac + valid_frac + test_frac
    if abs(total_frac - 1.0) > 0.01:
        parser.error(f"Fractions must sum to 1.0, got {total_frac:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Expression: {expression}")
    log.info(f"Metadata:   {metadata}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Split:      {train_frac}/{valid_frac}/{test_frac}")

    # Load metadata
    log.info("Loading metadata...")
    meta = pd.read_parquet(metadata)
    log.info(f"  {len(meta):,} cells, {meta['dataset_id'].nunique()} datasets")

    # Assign splits
    assignment = greedy_stratified_split(meta, train_frac, valid_frac, test_frac, seed)
    meta["split"] = meta["dataset_id"].map(assignment)

    # Load expression matrix
    log.info("Loading expression matrix...")
    t0 = time.time()
    X = load_npz(expression)
    log.info(f"  Loaded {X.shape[0]:,} × {X.shape[1]:,} ({time.time()-t0:.1f}s)")

    # Write per-split files
    for split in ["train", "valid", "test"]:
        mask = meta["split"].values == split
        indices = np.where(mask)[0]

        # Expression
        X_split = X[indices]
        npz_path = output_dir / f"{split}.npz"
        log.info(f"Saving {npz_path}: {X_split.shape[0]:,} cells...")
        t0 = time.time()
        save_npz(str(npz_path), X_split, compressed=compress)
        log.info(f"  Saved ({time.time()-t0:.1f}s)")

        # Metadata
        meta_split = meta[mask].drop(columns=["split"]).reset_index(drop=True)
        meta_path = output_dir / f"{split}_metadata.parquet"
        meta_split.to_parquet(str(meta_path), index=False)
        log.info(f"  Saved {meta_path}: {len(meta_split):,} rows")

    del X
    log.info("Done!")


if __name__ == "__main__":
    main()
