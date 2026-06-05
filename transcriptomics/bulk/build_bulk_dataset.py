"""
Build bulk transcriptomics dataset(s) from the ARCHS4 human gene-level HDF5.

Emits one expression.h5ad per gene set under output_dir/<mode>/ (same logical contract
as the single-cell arm: X + obs + var). UNLIKE the single-cell arm, bulk X is stored
DENSE + gzip (not sparse CSR): bulk is ~46% dense, so CSR's per-nonzero index overhead
balloons the file past the source. Dense uint32 + gzip matches ARCHS4's own ~58 GB.

The full matrix (1.09M x 67k) is ~294 GB dense in RAM, so X is STREAMED row-chunk by
row-chunk into a resizable gzip HDF5 dataset — only one ~0.5 GB block is held at a time.

This script writes a COMPLETE, valid h5ad with RAW metadata only. Metadata
harmonization is a SEPARATE, re-runnable step (normalize_metadata.py) so the vocabulary
can be tuned without re-streaming the matrix.

Source: ARCHS4 human gene-level counts (raw estimated counts, kallisto), via archs4py.
Download first (~58 GB):  python transcriptomics/bulk/_download_archs4.py

Usage:
    # 1. Smoke test on a small random subset:
    python transcriptomics/bulk/build_bulk_dataset.py --config transcriptomics/bulk/pipeline_config.yaml --n-random 500
    # 2. Full build (raw h5ad):
    python transcriptomics/bulk/build_bulk_dataset.py --config transcriptomics/bulk/pipeline_config.yaml
    # 3. Then harmonize metadata in place (separate, tunable):
    python transcriptomics/bulk/normalize_metadata.py  --config transcriptomics/bulk/pipeline_config.yaml

NOTE: ARCHS4's gene-level matrix is indexed by HGNC SYMBOL (not Ensembl); ~4.6k symbols
are duplicated and are made unique (suffixed). var_names are therefore symbols, so the
`protein_mapped` projection needs a symbol->UniProt map (distinct from the single-cell
arm's Ensembl map) — see that TODO.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import h5py
import anndata as ad
from anndata.io import write_elem

import archs4py as a4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (transcriptomics/bulk/ -> protgpt/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENE_SET_MODES = ("all_transcripts", "protein_mapped")
DTYPE_MAP = {"uint16": np.uint16, "uint32": np.uint32, "float16": np.float16, "float32": np.float32}


def _add_file_log(output_dir, name):
    """Also write logs to <output_dir>/logs/<name>.log so logs live with the bulk artifacts."""
    logdir = Path(output_dir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
META_FIELDS = ["geo_accession", "series_id", "source_name_ch1", "characteristics_ch1",
               "extract_protocol_ch1", "title"]


def _make_unique(names):
    """Suffix duplicate gene symbols (sym, sym-1, sym-2, ...) so var_names are unique."""
    seen, out = {}, []
    for nm in names:
        if nm in seen:
            seen[nm] += 1
            out.append(f"{nm}-{seen[nm]}")
        else:
            seen[nm] = 0
            out.append(nm)
    return out


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _select_candidates(F, cfg):
    """Sample indices to consider: all minus single-cell-flagged (singlecellprobability)."""
    sc_thresh = float(cfg.get("qc", {}).get("max_singlecell_probability", 0.5))
    n_total = len(np.asarray(a4.meta.field(F, "geo_accession")))
    idx = np.arange(n_total)
    try:
        scp = np.asarray(a4.meta.field(F, "singlecellprobability"), dtype=float)
        idx = idx[scp < sc_thresh]
        log.info(f"Samples: {n_total} total, {len(idx)} after single-cell filter (p<{sc_thresh})")
    except Exception as e:
        log.warning(f"No singlecellprobability field ({e}); keeping all {n_total} samples")
    return idx


def stream_build(F, candidate_idx, cfg, out_path: Path):
    """Stream ARCHS4 counts for candidate_idx into a dense gzip /X dataset at out_path.

    Applies per-sample QC on the fly. Holds one chunk in RAM at a time. Returns
    (kept_global_idx, gene_ids) — X rows are in kept_global_idx order, so obs built from
    the same indices stays aligned.
    """
    qc = cfg.get("qc", {})
    min_genes = int(qc.get("min_genes_per_sample", 0))
    min_counts = int(qc.get("min_total_counts", 0))
    dtype = DTYPE_MAP[cfg.get("dtype", "uint32")]
    chunk_size = int(cfg.get("chunk_size", 2000))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_global, gene_ids, n_genes = [], None, None
    row = 0
    with h5py.File(out_path, "w") as f:
        X = None
        for ci, chunk in enumerate(_chunks(list(candidate_idx), chunk_size)):
            df = a4.data.index(F, chunk, silent=True)        # genes x samples (uint32)
            if gene_ids is None:
                gene_ids = _make_unique(list(df.index))
                n_genes = len(gene_ids)
                X = f.create_dataset("X", shape=(0, n_genes), maxshape=(None, n_genes),
                                     chunks=(64, n_genes), compression="gzip", compression_opts=4,
                                     dtype=dtype)
                X.attrs["encoding-type"] = "array"
                X.attrs["encoding-version"] = "0.2.0"
            M = df.to_numpy().T                               # samples x genes
            genes_detected = (M > 0).sum(axis=1)
            total = M.sum(axis=1)
            ok = (genes_detected >= min_genes) & (total >= min_counts)
            if ok.any():
                block = M[ok].astype(dtype)
                X.resize(row + block.shape[0], axis=0)
                X[row:row + block.shape[0], :] = block
                row += block.shape[0]
                kept_global.extend(int(g) for g, k in zip(chunk, ok) if k)
            if ci % 10 == 0:
                log.info(f"  chunk {ci}: read {len(chunk)}, kept {int(ok.sum())} "
                         f"(rows written {row})")
        if X is None:
            raise RuntimeError("No samples selected.")
    log.info(f"Streamed X: {row} samples x {n_genes} genes -> {out_path}")
    return kept_global, gene_ids


def build_obs(F, kept_global_idx):
    """Per-sample metadata for kept samples, aligned to X row order."""
    cols = {}
    for fld in META_FIELDS:
        try:
            arr = np.asarray(a4.meta.field(F, fld))
            cols[fld] = arr[kept_global_idx]
        except Exception as e:
            log.warning(f"metadata field {fld!r} unavailable: {e}")
    obs = pd.DataFrame(cols)
    obs.insert(0, "sample_id", obs.get("geo_accession", pd.Series(range(len(obs))).astype(str)).values)
    obs.index = obs["sample_id"].astype(str)
    return obs


def finalize_h5ad(out_path: Path, obs: pd.DataFrame, gene_ids, cfg: dict):
    """Attach obs/var + AnnData encoding to the file whose /X was already streamed."""
    var = pd.DataFrame(index=pd.Index(gene_ids, name=None))
    with h5py.File(out_path, "a") as f:
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        write_elem(f, "obs", obs)
        write_elem(f, "var", var)
    with open(out_path.parent / "config.yaml", "w") as cf:
        yaml.safe_dump(cfg, cf, sort_keys=False)
    log.info(f"  Finalized {out_path}  ({len(obs)} samples x {len(gene_ids)} genes, dense gzip)")


def main():
    parser = argparse.ArgumentParser(description="Build bulk transcriptomics dataset(s) from ARCHS4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", default=None, choices=GENE_SET_MODES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-random", type=int, default=None,
                        help="Smoke test: build from N random samples instead of the full set")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    F = str(ROOT / cfg["archs4_h5"])
    output_dir = ROOT / cfg["output_dir"]
    _add_file_log(output_dir, "build")
    modes = args.datasets or cfg.get("datasets_to_build", ["all_transcripts"])
    log.info(f"ARCHS4 source: {F}")
    log.info(f"Output:        {output_dir}")
    log.info(f"Gene sets:     {modes}")

    if "protein_mapped" in modes:
        # TODO(protein_mapped): ARCHS4 var is HGNC symbols -> build a symbol->UniProt map
        # and stream X @ P. Not yet implemented; all_transcripts is the pretraining set.
        raise NotImplementedError("TODO(protein_mapped): symbol->UniProt projection")

    # --- sample selection ---
    if args.n_random:
        log.info(f"Smoke mode: {args.n_random} random samples (remove_sc=True)")
        rand = a4.data.rand(F, int(args.n_random), seed=int(cfg.get("seed", 42)),
                            remove_sc=True, silent=True)
        # map the random sample ids back to global indices for a uniform streaming path
        all_ids = list(np.asarray(a4.meta.field(F, "geo_accession")))
        pos = {sid: i for i, sid in enumerate(all_ids)}
        candidate_idx = np.array([pos[s] for s in rand.columns])
    else:
        candidate_idx = _select_candidates(F, cfg)

    out_path = output_dir / "all_transcripts" / "expression.h5ad"
    kept_idx, gene_ids = stream_build(F, candidate_idx, cfg, out_path)
    obs = build_obs(F, kept_idx)
    finalize_h5ad(out_path, obs, gene_ids, cfg)

    # Pristine, immutable snapshot of the raw metadata (never touched by harmonization).
    # obs rows align with X across both gene sets, so one snapshot covers all.
    raw_meta = output_dir / "metadata_raw.parquet"
    obs.to_parquet(raw_meta)
    log.info(f"Saved pristine raw metadata snapshot -> {raw_meta}")
    log.info("Done. Harmonize metadata separately: "
             "python transcriptomics/bulk/normalize_metadata.py --config transcriptomics/bulk/pipeline_config.yaml")


if __name__ == "__main__":
    main()
