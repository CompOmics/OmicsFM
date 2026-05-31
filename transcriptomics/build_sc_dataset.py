"""
Build single-cell transcriptomics dataset from CELLxGENE Census.

Config-driven, sparse throughout. No dense intermediaries.

Usage:
    python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.json
    python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.json --test
"""

import argparse
import json
import logging
import multiprocessing
import yaml
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, vstack as sparse_vstack, save_npz
from tqdm import tqdm


def _setup_logging():
    """Configure logging with line buffering. Safe to call in child processes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
        stream=sys.stderr,
        force=True,  # reconfigure even if already set (needed for child processes)
    )
    try:
        sys.stderr.reconfigure(line_buffering=True)
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass  # may fail in some subprocess contexts


_setup_logging()
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "float16": np.float16,
    "float32": np.float32,
}

DTYPE_MAX = {
    "uint8": 255,
    "uint16": 65535,
    "float16": 65504,
    "float32": np.finfo(np.float32).max,
}

META_COLUMNS = [
    "soma_joinid", "dataset_id", "cell_type", "tissue", "tissue_general",
    "disease", "donor_id", "sex", "assay", "development_stage",
    "self_reported_ethnicity", "raw_sum", "nnz",
]


# ── Gene-Protein Map ────────────────────────────────────────────────────


@dataclass
class GeneProteinMap:
    """Holds the gene→protein mapping and sparse projection matrix."""
    protein_columns: list[str]          # sorted UniProt accessions (output column order)
    projection: csr_matrix              # (n_census_genes × n_proteins), for X @ P
    mt_gene_indices: np.ndarray         # Census gene indices for mitochondrial genes
    census_var_df: pd.DataFrame         # Census gene metadata (feature_id, feature_name)


def get_model_proteins(parquet_path: str) -> list[str]:
    """Extract sorted UniProt accessions from proteomics training data columns."""
    cols = pd.read_parquet(ROOT / parquet_path, columns=None).columns.tolist()
    prots = sorted(c for c in cols if c not in ("pxd_accession", "run_name"))
    log.info(f"Model proteome: {len(prots)} proteins from {parquet_path}")
    return prots


def ensure_gene_protein_map(config: dict, census) -> GeneProteinMap:
    """Load or build gene→protein mapping, then construct the sparse projection matrix."""
    map_config = config["gene_protein_map"]
    cache_path = ROOT / map_config["cache_path"]
    force = map_config.get("force_rebuild", False)

    # Build mapping if needed
    if not cache_path.exists() or force:
        log.info("Building gene→protein mapping (this may take a few minutes)...")
        from transcriptomics.build_gene_protein_map import build_mapping
        build_mapping(
            parquet_path=str(ROOT / config["proteomics_reference"]),
            output_path=str(cache_path),
        )

    # Load mapping
    map_df = pd.read_parquet(cache_path)
    map_df = map_df[map_df["in_model"] == True]
    ensg_to_protein = dict(zip(map_df["ensembl_gene_id"], map_df["uniprot_accession"]))
    log.info(f"Gene→protein map: {len(ensg_to_protein)} genes → model proteins")

    # Get model protein columns
    protein_columns = get_model_proteins(config["proteomics_reference"])
    protein_to_idx = {p: i for i, p in enumerate(protein_columns)}

    # Get Census gene metadata
    log.info("Loading Census gene metadata...")
    var_df = (
        census["census_data"]["homo_sapiens"]
        .ms["RNA"]
        .var.read(column_names=["soma_joinid", "feature_id", "feature_name"])
        .concat()
        .to_pandas()
    )
    log.info(f"Census genes: {len(var_df)}")

    # Build sparse projection matrix: (n_census_genes × n_model_proteins)
    # P[gene_idx, protein_idx] = 1 where gene maps to protein
    n_genes = len(var_df)
    n_proteins = len(protein_columns)

    rows, cols = [], []
    for gene_idx, row in enumerate(var_df.itertuples()):
        ensg = row.feature_id.split(".")[0]  # strip version
        if ensg in ensg_to_protein:
            protein = ensg_to_protein[ensg]
            if protein in protein_to_idx:
                rows.append(gene_idx)
                cols.append(protein_to_idx[protein])

    projection = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_genes, n_proteins),
    ).tocsr()
    log.info(f"Projection matrix: {projection.shape}, {projection.nnz} mappings")

    # Identify mitochondrial gene indices (by gene symbol prefix)
    mt_mask = var_df["feature_name"].str.startswith("MT-", na=False)
    mt_gene_indices = np.where(mt_mask.values)[0]
    log.info(f"Mitochondrial genes: {len(mt_gene_indices)}")

    return GeneProteinMap(
        protein_columns=protein_columns,
        projection=projection,
        mt_gene_indices=mt_gene_indices,
        census_var_df=var_df,
    )


# ── Sampling ─────────────────────────────────────────────────────────────


def sample_cells(obs_df: pd.DataFrame, n: int, strategy: str, seed: int) -> pd.DataFrame:
    """Subsample cells from metadata. Returns filtered obs_df."""
    if len(obs_df) <= n:
        return obs_df

    rng = np.random.RandomState(seed)

    if strategy == "diversity":
        ct_counts = obs_df["cell_type"].map(obs_df["cell_type"].value_counts())
        weights = (1.0 / ct_counts).values
        weights /= weights.sum()
        idx = rng.choice(len(obs_df), size=n, replace=False, p=weights)
    else:  # random
        idx = rng.choice(len(obs_df), size=n, replace=False)

    return obs_df.iloc[idx].reset_index(drop=True)


# ── QC ───────────────────────────────────────────────────────────────────


def qc_filter_sparse(X: csr_matrix, obs_df: pd.DataFrame,
                     qc_config: dict, mt_indices: np.ndarray) -> tuple[csr_matrix, pd.DataFrame]:
    """Apply QC filters on a sparse expression matrix. No densification."""
    n_before = X.shape[0]

    # Genes per cell = number of non-zeros per row
    genes_per_cell = np.diff(X.indptr)

    # UMI per cell = row sums
    umi_per_cell = np.array(X.sum(axis=1)).ravel()

    # Mitochondrial fraction
    if len(mt_indices) > 0:
        mt_sum = np.array(X[:, mt_indices].sum(axis=1)).ravel()
        mt_frac = mt_sum / (umi_per_cell + 1e-8)
        mt_mask = mt_frac <= qc_config["max_mito_fraction"]
    else:
        mt_mask = np.ones(n_before, dtype=bool)

    mask = (
        (genes_per_cell >= qc_config["min_genes_per_cell"])
        & (umi_per_cell >= qc_config["min_umi_per_cell"])
        & mt_mask
    )

    X_filtered = X[mask]
    obs_filtered = obs_df[mask].reset_index(drop=True)
    n_after = X_filtered.shape[0]
    log.info(f"    QC: {n_before:,} → {n_after:,} cells (removed {n_before - n_after:,})")
    return X_filtered, obs_filtered


# ── Per-Tissue Fetch & Process ───────────────────────────────────────────


def fetch_and_process_tissue(
    census, tissue: str, config: dict, gene_map: GeneProteinMap,
    parallel: bool = False,
) -> tuple[csr_matrix, pd.DataFrame] | None:
    """Fetch, sample, QC, and translate one tissue. Returns (sparse_protein_matrix, metadata)."""
    import tiledbsoma

    t0 = time.time()
    human = census["census_data"]["homo_sapiens"]
    value_filter = (
        f"is_primary_data == True and tissue_general == '{tissue}'"
        " and suspension_type == 'cell'"
    )
    cells_per_tissue = config.get("cells_per_tissue")

    # Step 1: Read cell metadata (with early stop for large tissues)
    log.info(f"  [{tissue}] Reading metadata...")
    obs_chunks = []
    n_read = 0
    for chunk in human.obs.read(value_filter=value_filter, column_names=META_COLUMNS):
        obs_chunks.append(chunk.to_pandas())
        n_read += len(obs_chunks[-1])
        if cells_per_tissue and n_read >= cells_per_tissue * 10:
            break

    if not obs_chunks:
        log.info(f"  [{tissue}] No cells found, skipping")
        return None

    obs_df = pd.concat(obs_chunks, ignore_index=True)
    del obs_chunks
    log.info(f"  [{tissue}] Found {len(obs_df):,} cells ({time.time()-t0:.1f}s)")

    # Step 2: Sample cells
    if cells_per_tissue:
        obs_df = sample_cells(obs_df, cells_per_tissue, config["sampling_strategy"], config["seed"])
        log.info(f"  [{tissue}] Sampled to {len(obs_df):,} cells "
                 f"({obs_df['cell_type'].nunique()} cell types)")

    # Step 3: Stream expression COO → CSR
    joinids = np.sort(obs_df["soma_joinid"].values)
    var_joinids = gene_map.census_var_df["soma_joinid"].values

    query = human.axis_query(
        measurement_name="RNA",
        obs_query=tiledbsoma.AxisQuery(coords=(joinids,)),
    )

    # Build joinid → local index lookups
    obs_max = int(joinids.max()) + 1
    var_max = int(var_joinids.max()) + 1
    obs_lookup = np.full(obs_max, -1, dtype=np.int32)
    var_lookup = np.full(var_max, -1, dtype=np.int32)
    obs_lookup[joinids] = np.arange(len(joinids))
    var_lookup[var_joinids] = np.arange(len(var_joinids))

    n_obs = len(joinids)
    n_var = len(var_joinids)

    log.info(f"  [{tissue}] Streaming expression ({n_obs:,} × {n_var:,})...")
    rows, cols, vals = [], [], []
    n_entries = 0
    n_chunks = 0

    iterator = query.X("raw").tables()
    if not parallel:
        iterator = tqdm(iterator, desc=f"  [{tissue}]", unit="chunk")

    for arrow_tbl in iterator:
        obs_dim = arrow_tbl["soma_dim_0"].to_numpy()
        var_dim = arrow_tbl["soma_dim_1"].to_numpy()
        data = arrow_tbl["soma_data"].to_numpy().astype(np.float32)

        obs_local = obs_lookup[np.minimum(obs_dim, obs_max - 1)]
        var_local = var_lookup[np.minimum(var_dim, var_max - 1)]

        valid = (obs_local >= 0) & (var_local >= 0)
        rows.append(obs_local[valid])
        cols.append(var_local[valid])
        vals.append(data[valid])
        n_entries += valid.sum()
        n_chunks += 1

        if parallel and n_chunks % 20 == 0:
            log.info(f"  [{tissue}] {n_chunks} chunks, {n_entries:,} entries...")

    query.close()
    log.info(f"  [{tissue}] Streamed {n_entries:,} non-zero entries in {n_chunks} chunks ({time.time()-t0:.1f}s)")

    X_census = csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_obs, n_var),
    )
    del rows, cols, vals

    # Reorder obs_df to match joinids sort order
    obs_df = obs_df.set_index("soma_joinid").loc[joinids].reset_index(drop=True)

    # Step 4: QC filter (sparse)
    X_census, obs_df = qc_filter_sparse(X_census, obs_df, config["qc"], gene_map.mt_gene_indices)

    if X_census.shape[0] == 0:
        log.info(f"  [{tissue}] No cells survived QC, skipping")
        return None

    # Step 5: Project genes → proteins (sparse matmul)
    X_protein = X_census @ gene_map.projection
    del X_census

    # Step 6: Cast dtype
    dtype_name = config.get("dtype", "uint8")
    dtype = DTYPE_MAP[dtype_name]
    max_val = DTYPE_MAX[dtype_name]

    if np.issubdtype(dtype, np.integer):
        X_protein.data = np.clip(X_protein.data, 0, max_val).astype(dtype)
    else:
        X_protein.data = X_protein.data.astype(dtype)

    # Eliminate zeros that may have been introduced
    X_protein.eliminate_zeros()

    log.info(f"  [{tissue}] Done: {X_protein.shape[0]:,} cells × {X_protein.shape[1]:,} proteins, "
             f"nnz={X_protein.nnz:,}, dtype={dtype_name} ({time.time()-t0:.1f}s)")

    return X_protein, obs_df


# ── Split & Reorder ──────────────────────────────────────────────────────


def _split_and_reorder(
    X: csr_matrix, meta: pd.DataFrame, split_cfg: dict, seed: int,
) -> tuple[csr_matrix, pd.DataFrame, dict]:
    """Split by dataset_id and reorder rows as [train, valid, test].

    Returns (X_reordered, meta_reordered, split_indices) where split_indices
    maps split name → [start, end] row range.
    """
    train_frac = split_cfg.get("train_frac", 0.9)
    valid_frac = split_cfg.get("valid_frac", 0.05)

    datasets = meta["dataset_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(datasets)

    n = len(datasets)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train_ds = set(datasets[:n_train])
    valid_ds = set(datasets[n_train:n_train + n_valid])
    test_ds = set(datasets[n_train + n_valid:])

    ds_col = meta["dataset_id"].values
    train_mask = np.array([d in train_ds for d in ds_col])
    valid_mask = np.array([d in valid_ds for d in ds_col])
    test_mask = np.array([d in test_ds for d in ds_col])

    # Reorder: train rows first, then valid, then test
    order = np.concatenate([
        np.where(train_mask)[0],
        np.where(valid_mask)[0],
        np.where(test_mask)[0],
    ])

    X_reordered = X[order]
    meta_reordered = meta.iloc[order].reset_index(drop=True)

    # Add split column to metadata
    n_train_rows = int(train_mask.sum())
    n_valid_rows = int(valid_mask.sum())
    n_test_rows = int(test_mask.sum())

    split_labels = (
        ["train"] * n_train_rows +
        ["valid"] * n_valid_rows +
        ["test"] * n_test_rows
    )
    meta_reordered["split"] = split_labels

    split_indices = {
        "train": [0, n_train_rows],
        "valid": [n_train_rows, n_train_rows + n_valid_rows],
        "test": [n_train_rows + n_valid_rows, n_train_rows + n_valid_rows + n_test_rows],
    }

    log.info(f"Split by dataset_id ({len(datasets)} datasets → "
             f"{len(train_ds)} train, {len(valid_ds)} valid, {len(test_ds)} test):")
    log.info(f"  train: {n_train_rows:,} cells (rows {split_indices['train'][0]}–{split_indices['train'][1]})")
    log.info(f"  valid: {n_valid_rows:,} cells (rows {split_indices['valid'][0]}–{split_indices['valid'][1]})")
    log.info(f"  test:  {n_test_rows:,} cells (rows {split_indices['test'][0]}–{split_indices['test'][1]})")

    return X_reordered, meta_reordered, split_indices


# ── Save ─────────────────────────────────────────────────────────────────


def save_dataset(output_dir: Path, X: csr_matrix, metadata: pd.DataFrame,
                 protein_columns: list[str], config: dict):
    """Save the final dataset: sparse matrix, metadata, column labels, config."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expression matrix
    t0 = time.time()
    npz_path = output_dir / "expression.npz"
    log.info(f"Saving {npz_path} ({X.shape[0]:,} × {X.shape[1]:,}, nnz={X.nnz:,})...")
    compressed = config.get("compress_output", False)
    save_npz(str(npz_path), X, compressed=compressed)
    log.info(f"  compressed={compressed}")
    size_mb = npz_path.stat().st_size / 1e6
    log.info(f"  Saved expression.npz: {size_mb:.1f} MB ({time.time()-t0:.1f}s)")

    # Metadata
    meta_path = output_dir / "metadata.parquet"
    metadata.to_parquet(str(meta_path), index=False)
    log.info(f"  Saved metadata.parquet: {len(metadata):,} rows")

    # Protein column labels
    cols_path = output_dir / "protein_columns.json"
    with open(cols_path, "w") as f:
        json.dump(protein_columns, f)
    log.info(f"  Saved protein_columns.json: {len(protein_columns)} proteins")

    # Config copy
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"  Saved config.yaml")


# ── Main ─────────────────────────────────────────────────────────────────


def _process_tissue_worker(args):
    """Worker function for parallel tissue processing.

    Each worker opens its own Census connection since SOMA objects
    can't be shared across processes.
    """
    tissue, config, gene_map = args

    # Re-initialize logging in child process
    _setup_logging()
    # Name the process for log clarity
    multiprocessing.current_process().name = tissue

    import cellxgene_census

    try:
        with cellxgene_census.open_soma(census_version=config["census_version"]) as census:
            result = fetch_and_process_tissue(census, tissue, config, gene_map, parallel=True)
        return tissue, result
    except Exception as e:
        log.error(f"[{tissue}] Failed: {e}", exc_info=True)
        return tissue, None


def main():
    parser = argparse.ArgumentParser(description="Build SC transcriptomics dataset")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.json")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 tissues, 1k cells")
    parser.add_argument("--tissues", nargs="+", default=None, help="Override tissue list")
    parser.add_argument("--cells-per-tissue", type=int, default=None, help="Override cells_per_tissue")
    parser.add_argument("--workers", type=int, default=None, help="Override number of parallel workers")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.test:
        config["tissues"] = ["blood", "brain", "liver"]
        config["cells_per_tissue"] = 1000
    if args.tissues:
        config["tissues"] = args.tissues
    if args.cells_per_tissue:
        config["cells_per_tissue"] = args.cells_per_tissue

    output_dir = ROOT / config["output_dir"]
    tissues = config["tissues"]
    dtype_name = config.get("dtype", "uint8")
    n_workers = args.workers or config.get("workers", 1)

    log.info(f"Config: {len(tissues)} tissues, {config['cells_per_tissue']} cells/tissue, "
             f"dtype={dtype_name}, sampling={config['sampling_strategy']}, workers={n_workers}")
    log.info(f"Output: {output_dir}")

    import cellxgene_census

    t_start = time.time()

    # Build gene-protein map (needs one Census connection)
    with cellxgene_census.open_soma(census_version=config["census_version"]) as census:
        gene_map = ensure_gene_protein_map(config, census)

    # Process tissues
    X_blocks = []
    meta_blocks = []
    total_cells = 0

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        log.info(f"Processing {len(tissues)} tissues with {n_workers} parallel workers...")
        worker_args = [(tissue, config, gene_map) for tissue in tissues]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_tissue_worker, a): a[0] for a in worker_args}

            for future in as_completed(futures):
                tissue = futures[future]
                tissue_name, result = future.result()

                if result is None:
                    log.info(f"  [{tissue_name}] No result")
                    continue

                X_tissue, meta_tissue = result
                X_blocks.append(X_tissue)
                meta_blocks.append(meta_tissue)
                total_cells += X_tissue.shape[0]

                mem_mb = sum(
                    b.data.nbytes + b.indices.nbytes + b.indptr.nbytes for b in X_blocks
                ) / 1e6
                log.info(f"  [{tissue_name}] Collected. Running total: {total_cells:,} cells, {mem_mb:.0f} MB")
    else:
        # Sequential — use a single Census connection
        with cellxgene_census.open_soma(census_version=config["census_version"]) as census:
            for i, tissue in enumerate(tissues, 1):
                log.info(f"")
                log.info(f"{'='*60}")
                log.info(f"  Tissue {i}/{len(tissues)}: {tissue}")
                log.info(f"{'='*60}")

                try:
                    result = fetch_and_process_tissue(census, tissue, config, gene_map)
                except Exception as e:
                    log.error(f"  [{tissue}] Failed: {e}", exc_info=True)
                    continue

                if result is None:
                    continue

                X_tissue, meta_tissue = result
                X_blocks.append(X_tissue)
                meta_blocks.append(meta_tissue)
                total_cells += X_tissue.shape[0]

                mem_mb = sum(
                    b.data.nbytes + b.indices.nbytes + b.indptr.nbytes for b in X_blocks
                ) / 1e6
                log.info(f"  Running total: {total_cells:,} cells, {mem_mb:.0f} MB sparse")

    # Stack all tissues
    log.info(f"")
    log.info(f"Stacking {len(X_blocks)} tissue blocks...")
    X_all = sparse_vstack(X_blocks, format="csr")
    del X_blocks
    meta_all = pd.concat(meta_blocks, ignore_index=True)
    del meta_blocks

    mem_mb = (X_all.data.nbytes + X_all.indices.nbytes + X_all.indptr.nbytes) / 1e6
    density = X_all.nnz / (X_all.shape[0] * X_all.shape[1]) * 100
    log.info(f"Combined: {X_all.shape[0]:,} cells × {X_all.shape[1]:,} proteins, "
             f"nnz={X_all.nnz:,}, density={density:.2f}%, {mem_mb:.0f} MB")

    # Save
    save_dataset(output_dir, X_all, meta_all, gene_map.protein_columns, config)

    elapsed = time.time() - t_start
    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"  COMPLETE: {total_cells:,} cells in {elapsed:.0f}s")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
