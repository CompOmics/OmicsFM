"""
Split transcriptomics dataset(s) into train/valid/test by dataset_id (project).

Whole projects (`dataset_id`) are assigned to a single split — no project leaks
across splits — using a random 90/5/5 grouped split (the proteomics-style "split
by project"). The assignment is deterministic in (sorted dataset_ids, seed), so
every gene set in `datasets_to_build` gets the *same* cells in the *same* split,
keeping the gene-level and protein datasets cell-aligned.

For each built dataset it reads   output_dir/<mode>/expression.h5ad
and writes                        split_dir/<mode>/{train,valid,test}.h5ad

Usage:
    # Split every dataset listed in datasets_to_build:
    python transcriptomics/split_dataset.py --config transcriptomics/pipeline_config.yaml

    # Split a single expression.h5ad ad hoc:
    python transcriptomics/split_dataset.py \
      --expression transcriptomics/output/all_transcripts/expression.h5ad \
      --output-dir transcriptomics/split/all_transcripts
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
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


def assign_splits(dataset_ids, valid_frac, test_frac, seed) -> dict:
    """Map each dataset_id → split name via a random grouped split.

    Whole datasets are shuffled (deterministically from `seed`) and assigned to
    train/valid/test. valid/test take their fractions, train gets the remainder;
    valid/test get at least one dataset each when there are ≥3 datasets, so no
    split is accidentally empty.
    """
    datasets = np.array(sorted({str(d) for d in dataset_ids}))
    rng = np.random.RandomState(seed)
    rng.shuffle(datasets)
    n = len(datasets)

    if n < 3:
        log.warning(f"Only {n} dataset(s) — assigning all to train (cannot form 3 splits)")
        return {d: "train" for d in datasets}

    n_valid = max(1, int(n * valid_frac)) if valid_frac > 0 else 0
    n_test = max(1, int(n * test_frac)) if test_frac > 0 else 0
    n_train = n - n_valid - n_test  # train takes the remainder (≥1 since n≥3)

    assignment = {}
    for d in datasets[:n_train]:
        assignment[d] = "train"
    for d in datasets[n_train:n_train + n_valid]:
        assignment[d] = "valid"
    for d in datasets[n_train + n_valid:]:
        assignment[d] = "test"
    return assignment


def split_expression(expression: str, output_dir: Path,
                     valid_frac: float, test_frac: float, seed: int):
    """Split one expression.h5ad into train/valid/test by dataset_id.

    train takes whatever fraction is left after valid_frac + test_frac.
    """
    import anndata as ad
    from protgpt.convert import save_unified_h5ad

    log.info(f"Loading {expression}...")
    t0 = time.time()
    adata = ad.read_h5ad(expression)
    X = adata.X.tocsr()
    meta = adata.obs
    var_names = list(adata.var_names)
    ds_col = meta["dataset_id"].astype(str).values
    log.info(f"  Loaded {X.shape[0]:,} × {X.shape[1]:,}, "
             f"{len(set(ds_col))} datasets ({time.time()-t0:.1f}s)")

    assignment = assign_splits(ds_col, valid_frac, test_frac, seed)
    split_of = np.array([assignment[d] for d in ds_col])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        indices = np.where(split_of == split)[0]
        n_ds = len({assignment_d for assignment_d, s in assignment.items() if s == split})
        out_path = output_dir / f"{split}.h5ad"
        log.info(f"Saving {out_path}: {len(indices):,} cells from {n_ds} datasets...")
        t0 = time.time()
        save_unified_h5ad(str(out_path), X[indices], meta.iloc[indices], var_names,
                          modality="transcriptomics", value_semantics="umi_counts")
        log.info(f"  Saved ({time.time()-t0:.1f}s)")

    # Leakage check: no dataset_id may appear in more than one split.
    overlaps = []
    for a in ["train", "valid", "test"]:
        for b in ["valid", "test"]:
            if a >= b:
                continue
            sa = set(ds_col[split_of == a])
            sb = set(ds_col[split_of == b])
            if sa & sb:
                overlaps.append((a, b, sa & sb))
    if overlaps:
        for a, b, shared in overlaps:
            log.error(f"  LEAKAGE: {len(shared)} dataset_id(s) shared between {a} and {b}")
        raise RuntimeError("dataset_id leakage detected across splits")
    log.info("  No dataset_id leakage across splits ✓")
    del X


def main():
    parser = argparse.ArgumentParser(
        description="Split transcriptomics dataset(s) by dataset_id",
        epilog="With --config, splits every dataset in datasets_to_build. "
               "With --expression/--output-dir, splits a single file.",
    )
    parser.add_argument("--config", help="Path to pipeline_config.yaml")
    parser.add_argument("--expression", help="Path to a single expression.h5ad (ad hoc)")
    parser.add_argument("--output-dir", help="Output directory (with --expression)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Override which datasets_to_build modes to split (with --config)")
    parser.add_argument("--train-frac", type=float)
    parser.add_argument("--valid-frac", type=float)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        split_cfg = cfg.get("split", {})
        train_frac = args.train_frac if args.train_frac is not None else split_cfg.get("train_frac", 0.9)
        valid_frac = args.valid_frac if args.valid_frac is not None else split_cfg.get("valid_frac", 0.05)
        test_frac = args.test_frac if args.test_frac is not None else split_cfg.get("test_frac", 0.05)
        seed = args.seed if args.seed is not None else cfg.get("seed", 42)

        _check_fracs(parser, train_frac, valid_frac, test_frac)

        build_dir = Path(cfg.get("output_dir", "transcriptomics/output"))
        split_dir = Path(split_cfg.get("output_dir", "transcriptomics/split"))
        modes = args.datasets or cfg.get("datasets_to_build", ["all_transcripts"])

        log.info(f"Splitting datasets: {modes}")
        log.info(f"Split: {train_frac}/{valid_frac}/{test_frac}, seed={seed}")
        for m in modes:
            expression = build_dir / m / "expression.h5ad"
            if not expression.exists():
                log.error(f"[{m}] missing {expression} — did build_sc_dataset.py run for it? Skipping.")
                continue
            log.info("")
            log.info(f"{'='*60}\n  Dataset: {m}\n{'='*60}")
            split_expression(str(expression), split_dir / m,
                             valid_frac, test_frac, seed)
    else:
        if not args.expression or not args.output_dir:
            parser.error("provide --config, or both --expression and --output-dir")
        train_frac = args.train_frac if args.train_frac is not None else 0.9
        valid_frac = args.valid_frac if args.valid_frac is not None else 0.05
        test_frac = args.test_frac if args.test_frac is not None else 0.05
        seed = args.seed if args.seed is not None else 42
        _check_fracs(parser, train_frac, valid_frac, test_frac)
        split_expression(args.expression, Path(args.output_dir),
                         valid_frac, test_frac, seed)

    log.info("Done!")


def _check_fracs(parser, train_frac, valid_frac, test_frac):
    total = train_frac + valid_frac + test_frac
    if abs(total - 1.0) > 0.01:
        parser.error(f"Fractions must sum to 1.0, got {total:.2f}")


if __name__ == "__main__":
    main()
