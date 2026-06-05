"""
Split bulk dataset(s) into train/valid/test by STUDY (GSE), streaming.

Whole studies are assigned to a single split so no study leaks across splits — the bulk
analogue of the single-cell "split by dataset_id". Because some ARCHS4 samples list
multiple GSEs in `series_id` (super-series), grouping is done with union-find over the
GSE ids: any samples sharing ANY GSE land in the same group, so a GSE can never straddle
two splits. Assignment is deterministic in (sorted groups, seed).

X is dense and large, so the matrix is never loaded: rows are streamed from the source
h5ad and routed to their split's resizable gzip dataset. obs (incl. harmonized columns)
and var are copied per split.

For each gene set it reads   output_dir/<mode>/expression.h5ad
and writes                   split.output_dir/<mode>/{train,valid,test}.h5ad

Usage:
    python transcriptomics/bulk/split_dataset.py --config transcriptomics/bulk/pipeline_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import h5py
from anndata.io import read_elem, write_elem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS = ("train", "valid", "test")


def _add_file_log(output_dir, name):
    logdir = Path(output_dir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)


def assign_splits(series_strings, fracs, seed):
    """Leakage-safe split label per sample.

    Union-find over the GSE ids in each sample's `series_id` -> groups of samples that
    share any study. Groups are shuffled (seed) and assigned to train/valid/test by
    fraction. Returns (labels[str], group_of_sample[str]).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:           # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    gses_per_sample = [[g.strip() for g in str(s).split(",") if g.strip()] or ["__none__"]
                       for s in series_strings]
    for gses in gses_per_sample:
        for g in gses[1:]:
            union(gses[0], g)

    group_of_sample = np.array([find(gses[0]) for gses in gses_per_sample])
    groups = sorted(set(group_of_sample))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_tr = round(fracs[0] * len(groups))
    n_va = round(fracs[1] * len(groups))
    split_of_group = {}
    for rank, gi in enumerate(perm):
        split_of_group[groups[gi]] = ("train" if rank < n_tr
                                      else "valid" if rank < n_tr + n_va else "test")
    labels = np.array([split_of_group[g] for g in group_of_sample])
    return labels, group_of_sample


def split_mode(src: Path, out_mode_dir: Path, labels, chunk=4000):
    """Stream src X -> {train,valid,test}.h5ad under out_mode_dir, routing rows by label."""
    out_mode_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as fin:
        var = read_elem(fin["var"])
        obs = read_elem(fin["obs"])
        Xin = fin["X"]
        n, g = Xin.shape
        dt = Xin.dtype

        files, dsets, idxs, rows = {}, {}, {s: [] for s in SPLITS}, {s: 0 for s in SPLITS}
        for s in SPLITS:
            f = h5py.File(out_mode_dir / f"{s}.h5ad", "w")
            d = f.create_dataset("X", shape=(0, g), maxshape=(None, g), chunks=(64, g),
                                 compression="gzip", compression_opts=4, dtype=dt)
            d.attrs["encoding-type"] = "array"
            d.attrs["encoding-version"] = "0.2.0"
            files[s], dsets[s] = f, d

        for r0 in range(0, n, chunk):
            r1 = min(r0 + chunk, n)
            block = np.asarray(Xin[r0:r1])
            lbl = labels[r0:r1]
            rowids = np.arange(r0, r1)
            for s in SPLITS:
                m = lbl == s
                if m.any():
                    sub = block[m]
                    dsets[s].resize(rows[s] + sub.shape[0], axis=0)
                    dsets[s][rows[s]:rows[s] + sub.shape[0]] = sub
                    rows[s] += sub.shape[0]
                    idxs[s].extend(rowids[m].tolist())
            if (r0 // chunk) % 10 == 0:
                log.info(f"    rows {r0}-{r1}/{n}  (train {rows['train']}, valid {rows['valid']}, test {rows['test']})")

        for s in SPLITS:
            f = files[s]
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"
            write_elem(f, "obs", obs.iloc[idxs[s]])
            write_elem(f, "var", var)
            f.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Split bulk dataset(s) into train/valid/test by study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", default=None, help="Override datasets_to_build")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / cfg["output_dir"]
    _add_file_log(out_dir, "split")
    split_cfg = cfg["split"]
    split_out = ROOT / split_cfg["output_dir"]
    fracs = (split_cfg.get("train_frac", 0.9), split_cfg.get("valid_frac", 0.05),
             split_cfg.get("test_frac", 0.05))
    seed = int(cfg.get("seed", 42))

    # protein_mapped is built by project_proteins.py; include it if present.
    modes = args.datasets or cfg.get("datasets_to_build", ["all_transcripts"])
    if "protein_mapped" not in modes and (out_dir / "protein_mapped" / "expression.h5ad").exists():
        modes = list(modes) + ["protein_mapped"]

    # Compute the split assignment ONCE (from all_transcripts obs) so every gene set gets
    # the SAME samples in the SAME split — keeps the datasets row-aligned.
    ref_src = out_dir / modes[0] / "expression.h5ad"
    with h5py.File(ref_src, "r") as f:
        series = read_elem(f["obs"])["series_id"].astype(str).values
    labels, group_of_sample = assign_splits(series, fracs, seed)
    chk = pd.DataFrame({"g": group_of_sample, "s": labels}).drop_duplicates()
    assert chk.groupby("g")["s"].nunique().max() == 1, "LEAKAGE: a study spans multiple splits"
    n_groups = len(set(group_of_sample))
    log.info(f"Split assignment: {n_groups:,} study-groups, "
             f"{(labels=='train').sum():,}/{(labels=='valid').sum():,}/{(labels=='test').sum():,} "
             f"samples (train/valid/test); leakage check passed")

    for m in modes:
        src = out_dir / m / "expression.h5ad"
        if not src.exists():
            log.warning(f"skip {m}: {src} not found")
            continue
        log.info(f"Splitting {m} -> {split_out / m}")
        rows = split_mode(src, split_out / m, labels)
        log.info(f"  {m}: train={rows['train']:,}  valid={rows['valid']:,}  test={rows['test']:,}")


if __name__ == "__main__":
    main()
