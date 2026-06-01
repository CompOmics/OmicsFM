"""
Split the transcriptomics dataset by dataset_id into train/valid/test.

Reads the unified expression.h5ad, splits by dataset_id using greedy
tissue-stratified balancing, and outputs one unified file per split:
  - train.h5ad, valid.h5ad, test.h5ad  (sparse X + obs metadata + var accessions)

Usage:
    python transcriptomics/split_dataset.py \
      --expression transcriptomics/output/expression.h5ad \
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make `protgpt` importable when run as a script

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
    parser.add_argument("--config", help="Path to pipeline_config.yaml (reads output_dir, split, seed)")
    parser.add_argument("--expression", help="Path to expression.h5ad")
    parser.add_argument("--output-dir", help="Output directory for split files")
    parser.add_argument("--train-frac", type=float)
    parser.add_argument("--valid-frac", type=float)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Load defaults from config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        split_cfg = cfg.get("split", {})
        build_output = cfg.get("output_dir", "transcriptomics/output")

        # Config-derived defaults (CLI args override)
        expression = args.expression or str(Path(build_output) / "expression.h5ad")
        output_dir = Path(args.output_dir or split_cfg.get("output_dir", "transcriptomics/split"))
        train_frac = args.train_frac if args.train_frac is not None else split_cfg.get("train_frac", 0.9)
        valid_frac = args.valid_frac if args.valid_frac is not None else split_cfg.get("valid_frac", 0.05)
        test_frac = args.test_frac if args.test_frac is not None else split_cfg.get("test_frac", 0.05)
        seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    else:
        # All args required without config
        if not args.expression or not args.output_dir:
            parser.error("--expression and --output-dir are required without --config")
        expression = args.expression
        output_dir = Path(args.output_dir)
        train_frac = args.train_frac or 0.9
        valid_frac = args.valid_frac or 0.05
        test_frac = args.test_frac or 0.05
        seed = args.seed or 42

    total_frac = train_frac + valid_frac + test_frac
    if abs(total_frac - 1.0) > 0.01:
        parser.error(f"Fractions must sum to 1.0, got {total_frac:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Expression: {expression}")
    log.info(f"Output:     {output_dir}")
    log.info(f"Split:      {train_frac}/{valid_frac}/{test_frac}")

    # Load the unified source (X + obs metadata + var accessions)
    import anndata as ad
    from protgpt.convert import save_unified_h5ad

    log.info("Loading expression.h5ad...")
    t0 = time.time()
    adata = ad.read_h5ad(expression)
    X = adata.X.tocsr()
    meta = adata.obs
    var_names = list(adata.var_names)
    log.info(f"  Loaded {X.shape[0]:,} × {X.shape[1]:,}, "
             f"{meta['dataset_id'].nunique()} datasets ({time.time()-t0:.1f}s)")

    # Assign splits
    assignment = greedy_stratified_split(meta, train_frac, valid_frac, test_frac, seed)
    split_of = meta["dataset_id"].map(assignment).values

    # Write one unified .h5ad per split
    for split in ["train", "valid", "test"]:
        indices = np.where(split_of == split)[0]
        out_path = output_dir / f"{split}.h5ad"
        log.info(f"Saving {out_path}: {len(indices):,} cells...")
        t0 = time.time()
        save_unified_h5ad(str(out_path), X[indices], meta.iloc[indices], var_names,
                          modality="transcriptomics", value_semantics="umi_counts")
        log.info(f"  Saved ({time.time()-t0:.1f}s)")

    del X
    log.info("Done!")


if __name__ == "__main__":
    main()
