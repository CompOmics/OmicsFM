"""
Build single-cell transcriptomics dataset(s) from CELLxGENE Census.

Config-driven, sparse throughout. No dense intermediaries.

One Census stream per tissue feeds every requested dataset (`datasets_to_build`):
each tissue's QC'd gene matrix is derived into all requested gene sets at once
(all_transcripts / protein_mapped), so adding a dataset costs almost nothing.
Each dataset is written to its own subdirectory under output_dir.

Splitting into train/valid/test is NOT done here — it lives in split_dataset.py.

Usage:
    python transcriptomics/single_cell/build_sc_dataset.py --config transcriptomics/single_cell/pipeline_config.yaml
    python transcriptomics/single_cell/build_sc_dataset.py --config transcriptomics/single_cell/pipeline_config.yaml --test
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

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (transcriptomics/single_cell/ -> protgpt/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # make `protgpt` and `transcriptomics` importable

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

# Supported gene sets (values for `datasets_to_build`):
#   all_transcripts  - every Census gene that survives QC, native Ensembl ids
#   protein_mapped   - gene->protein projection X @ P, UniProt columns (summed)
GENE_SET_MODES = ("all_transcripts", "protein_mapped")


# ── Gene-Protein Map ────────────────────────────────────────────────────


@dataclass
class GeneProteinMap:
    """Holds the gene→protein mapping, the projection matrix, and column orders.

    Census genes are columns of the streamed matrix in `census_var_df` row order.
    The column lists below name the output columns for each gene set.
    """
    protein_columns: list[str]          # sorted UniProt accessions (protein_mapped columns)
    projection: csr_matrix              # (n_census_genes × n_proteins), for X @ P
    mt_gene_indices: np.ndarray         # Census gene indices for mitochondrial genes
    census_var_df: pd.DataFrame         # Census gene metadata (feature_id, feature_name)
    all_gene_columns: list[str]         # Ensembl ids of all Census genes (all_transcripts columns)

    def columns_for(self, mode: str) -> list[str]:
        return {
            "all_transcripts": self.all_gene_columns,
            "protein_mapped": self.protein_columns,
        }[mode]


def get_model_proteins(proteomics_reference: str) -> list[str]:
    """Sorted UniProt accessions the proteomics model uses (its h5ad var_names)."""
    from transcriptomics.build_gene_protein_map import get_model_proteins as _read
    prots = sorted(_read(str(ROOT / proteomics_reference)))
    log.info(f"Model proteome: {len(prots)} proteins from {proteomics_reference}")
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
            proteomics_path=str(ROOT / config["proteomics_reference"]),
            output_path=str(cache_path),
        )

    # Load the full gene→protein mapping. We do NOT use the cached `in_model`
    # flag (it reflects whichever proteomics reference built the map); instead we
    # re-check each gene's protein against the current protein set below, so the
    # map stays correct even if proteomics_reference changes without a rebuild.
    map_df = pd.read_parquet(cache_path)
    ensg_to_protein = dict(zip(map_df["ensembl_gene_id"], map_df["uniprot_accession"]))
    log.info(f"Gene→protein map: {len(ensg_to_protein)} genes with a UniProt accession")

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

    # all_transcripts columns: every Census gene, by Ensembl id. Census genes are
    # columns in var_df row order, so feat_ids[j] names column j of the matrix.
    feat_ids = var_df["feature_id"].str.split(".").str[0].values
    all_gene_columns = feat_ids.tolist()
    log.info(f"Genes: {len(all_gene_columns)} total (all_transcripts), "
             f"{len(set(rows))} mapped to model proteins (protein_mapped)")

    # Identify mitochondrial gene indices (by gene symbol prefix)
    mt_mask = var_df["feature_name"].str.startswith("MT-", na=False)
    mt_gene_indices = np.where(mt_mask.values)[0]
    log.info(f"Mitochondrial genes: {len(mt_gene_indices)}")

    return GeneProteinMap(
        protein_columns=protein_columns,
        projection=projection,
        mt_gene_indices=mt_gene_indices,
        census_var_df=var_df,
        all_gene_columns=all_gene_columns,
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
    """Fetch, sample and QC one tissue.

    Returns (X_census, obs_df): the QC'd sparse gene matrix (all Census genes,
    float32 raw counts) and its per-cell metadata. Derivation into the requested
    gene sets happens afterwards in `derive_dataset`.
    """
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

    log.info(f"  [{tissue}] Done: {X_census.shape[0]:,} cells × {X_census.shape[1]:,} genes, "
             f"nnz={X_census.nnz:,} ({time.time()-t0:.1f}s)")
    return X_census, obs_df


# ── Derive gene sets from the streamed Census matrix ───────────────────────


def _cast_sparse(X: csr_matrix, dtype_name: str) -> csr_matrix:
    """Return a copy of X with data cast to dtype_name (clipped if integer).

    Always builds a fresh matrix, so the input is never mutated — the same
    Census matrix can be derived into several gene sets in turn.
    """
    dtype = DTYPE_MAP[dtype_name]
    if np.issubdtype(dtype, np.integer):
        data = np.clip(X.data, 0, DTYPE_MAX[dtype_name]).astype(dtype)
    else:
        data = X.data.astype(dtype)
    out = csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


def derive_dataset(X_census: csr_matrix, gene_map: GeneProteinMap,
                   mode: str, dtype_name: str) -> csr_matrix:
    """Derive one gene set from the full QC'd Census matrix, cast to dtype_name.

    - all_transcripts : every Census gene (columns = all_gene_columns)
    - protein_mapped  : gene→protein projection X @ P (sums genes per protein)
    """
    if mode == "all_transcripts":
        X_out = X_census
    elif mode == "protein_mapped":
        X_out = (X_census @ gene_map.projection).tocsr()
    else:
        raise ValueError(f"Unknown gene set '{mode}'; expected one of {GENE_SET_MODES}")
    return _cast_sparse(X_out, dtype_name)


# ── Save ─────────────────────────────────────────────────────────────────


def save_dataset(output_dir: Path, X: csr_matrix, metadata: pd.DataFrame,
                 columns: list[str], config: dict):
    """Save one gene set natively as a unified `expression.h5ad`.

    Bundles the sparse matrix (X), per-cell metadata (obs), and gene/protein
    accessions (var) in one file — the format consumed by split_dataset.py and
    protgpt.data.ExpressionDataset.
    """
    from protgpt.convert import save_unified_h5ad

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    metadata = metadata.reset_index(drop=True)
    metadata.index = pd.Index([f"cell_{i}" for i in range(len(metadata))], name="cell_id")

    h5ad_path = output_dir / "expression.h5ad"
    log.info(f"Saving {h5ad_path} ({X.shape[0]:,} × {X.shape[1]:,}, nnz={X.nnz:,})...")
    save_unified_h5ad(str(h5ad_path), X, metadata, columns,
                      modality="transcriptomics", value_semantics="umi_counts")
    size_mb = h5ad_path.stat().st_size / 1e6
    log.info(f"  Saved expression.h5ad: {size_mb:.1f} MB ({time.time()-t0:.1f}s)")

    # Config copy (provenance)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"  Saved config.yaml")


# ── Main ─────────────────────────────────────────────────────────────────


def _derive_all(X_census: csr_matrix, gene_map: GeneProteinMap,
                modes: list[str], dtype_name: str) -> dict[str, csr_matrix]:
    """Derive every requested gene set from one tissue's Census matrix."""
    return {m: derive_dataset(X_census, gene_map, m, dtype_name) for m in modes}


def _process_tissue_worker(args):
    """Worker for parallel tissue processing: fetch + derive all gene sets.

    Each worker opens its own Census connection (SOMA objects can't cross
    processes) and derives the gene sets in-process, so only the (smaller)
    per-mode matrices are pickled back — not the full Census gene matrix.

    Returns (tissue, {mode: X}, obs_df) or (tissue, None, None) on no/failed result.
    """
    tissue, config, gene_map, modes, dtype_name = args

    # Re-initialize logging in child process; name it for log clarity
    _setup_logging()
    multiprocessing.current_process().name = tissue

    import cellxgene_census

    try:
        with cellxgene_census.open_soma(census_version=config["census_version"]) as census:
            result = fetch_and_process_tissue(census, tissue, config, gene_map, parallel=True)
        if result is None:
            return tissue, None, None
        X_census, obs_df = result
        return tissue, _derive_all(X_census, gene_map, modes, dtype_name), obs_df
    except Exception as e:
        log.error(f"[{tissue}] Failed: {e}", exc_info=True)
        return tissue, None, None


def main():
    parser = argparse.ArgumentParser(description="Build SC transcriptomics dataset(s)")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 tissues, 1k cells")
    parser.add_argument("--tissues", nargs="+", default=None, help="Override tissue list")
    parser.add_argument("--cells-per-tissue", type=int, default=None, help="Override cells_per_tissue")
    parser.add_argument("--workers", type=int, default=None, help="Override number of parallel workers")
    parser.add_argument("--datasets", nargs="+", default=None, choices=GENE_SET_MODES,
                        help="Override datasets_to_build (gene sets to emit)")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config")
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
    if args.datasets:
        config["datasets_to_build"] = args.datasets
    if args.output_dir:
        config["output_dir"] = args.output_dir

    modes = config.get("datasets_to_build", ["all_transcripts"])
    bad = [m for m in modes if m not in GENE_SET_MODES]
    if bad:
        parser.error(f"Unknown gene set(s) {bad} in datasets_to_build; expected {GENE_SET_MODES}")

    output_dir = ROOT / config["output_dir"]
    tissues = config["tissues"]
    dtype_name = config.get("dtype", "uint8")
    n_workers = args.workers or config.get("workers", 1)

    log.info(f"Config: {len(tissues)} tissues, {config['cells_per_tissue']} cells/tissue, "
             f"dtype={dtype_name}, sampling={config['sampling_strategy']}, workers={n_workers}")
    log.info(f"Datasets to build: {modes}")
    log.info(f"Output: {output_dir}")

    import cellxgene_census

    t_start = time.time()

    # Build gene-protein map (needs one Census connection)
    with cellxgene_census.open_soma(census_version=config["census_version"]) as census:
        gene_map = ensure_gene_protein_map(config, census)

    # Per-mode tissue blocks (aligned: every mode + meta is appended together per
    # tissue, so all gene sets stay cell-aligned regardless of completion order).
    X_blocks = {m: [] for m in modes}
    meta_blocks = []
    total_cells = 0

    def collect(tissue_name, derived, meta_tissue):
        nonlocal total_cells
        if derived is None:
            log.info(f"  [{tissue_name}] No result")
            return
        for m in modes:
            X_blocks[m].append(derived[m])
        meta_blocks.append(meta_tissue)
        total_cells += len(meta_tissue)
        mem_mb = sum(
            b.data.nbytes + b.indices.nbytes + b.indptr.nbytes
            for blocks in X_blocks.values() for b in blocks
        ) / 1e6
        log.info(f"  [{tissue_name}] Collected. Running total: {total_cells:,} cells, {mem_mb:.0f} MB")

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        log.info(f"Processing {len(tissues)} tissues with {n_workers} parallel workers...")
        worker_args = [(tissue, config, gene_map, modes, dtype_name) for tissue in tissues]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_tissue_worker, a): a[0] for a in worker_args}
            for future in as_completed(futures):
                tissue_name, derived, meta_tissue = future.result()
                collect(tissue_name, derived, meta_tissue)
    else:
        # Sequential — use a single Census connection
        with cellxgene_census.open_soma(census_version=config["census_version"]) as census:
            for i, tissue in enumerate(tissues, 1):
                log.info("")
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

                X_census, meta_tissue = result
                collect(tissue, _derive_all(X_census, gene_map, modes, dtype_name), meta_tissue)
                del X_census

    if not meta_blocks:
        log.error("No tissues produced data — nothing to save.")
        return

    # Stack and save each gene set (shared obs across all of them)
    meta_all = pd.concat(meta_blocks, ignore_index=True)
    del meta_blocks
    for m in modes:
        log.info("")
        log.info(f"Stacking {len(X_blocks[m])} tissue blocks for '{m}'...")
        X_all = sparse_vstack(X_blocks[m], format="csr")
        X_blocks[m] = None  # free as we go
        density = X_all.nnz / (X_all.shape[0] * X_all.shape[1]) * 100
        log.info(f"  '{m}': {X_all.shape[0]:,} cells × {X_all.shape[1]:,} cols, "
                 f"nnz={X_all.nnz:,}, density={density:.2f}%")
        save_dataset(output_dir / m, X_all, meta_all, gene_map.columns_for(m), config)
        del X_all

    elapsed = time.time() - t_start
    log.info("")
    log.info(f"{'='*60}")
    log.info(f"  COMPLETE: {total_cells:,} cells, {len(modes)} dataset(s) in {elapsed:.0f}s")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
