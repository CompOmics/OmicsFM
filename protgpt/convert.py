"""
Convert legacy ProtGPT data files to the unified `.h5ad` source format.

Both modalities share one schema (see docs/unified_data_format_plan.md):
    X    : sparse CSR, n_obs x n_var, RAW (unbinned) values
    obs  : per-sample metadata (obs_names = sample id)
    var  : per-protein/gene metadata (var_names = UniProt accession)
    uns  : modality, value_semantics, provenance

Proteomics : dense parquet (+ metadata.tsv) -> h5ad   (float32 values)
Transcriptomics : scipy .npz (+ protein_columns.json + metadata.parquet) -> h5ad   (uint8 values)

The transcriptomics path streams the matrix in row-blocks so it never holds the
full (up to ~60 GB) index array in RAM, and writes CSR with int32 indices + int64
indptr directly (scipy would upcast both to int64 for billions of non-zeros).

Usage:
    python -m protgpt.convert --input data/nsaf_diann/quant_train.parquet \
                              --output data/nsaf_diann/train.h5ad
    python -m protgpt.convert --input transcriptomics/split/test.npz \
                              --output transcriptomics/split/test.h5ad \
                              --protein-columns transcriptomics/output/protein_columns.json
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# anndata's element writer moved across versions; try the known locations.
try:
    from anndata.io import write_elem
except ImportError:  # pragma: no cover - depends on anndata version
    try:
        from anndata.experimental import write_elem
    except ImportError:
        from anndata._io.specs import write_elem


# ── npz member memory-mapping (avoids loading 60 GB index arrays) ──────────

def _zip_member_data_offset(npz_path: str, member: str) -> int:
    """Absolute byte offset where an (uncompressed) zip member's data begins."""
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{npz_path}::{member} is compressed; memory-mapping requires an "
                f"uncompressed .npz (saved with compress_output: false)."
            )
    with open(npz_path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
    return info.header_offset + 30 + name_len + extra_len


def open_npz_member_memmap(npz_path: str, member: str) -> np.ndarray:
    """Memory-map one array inside an uncompressed .npz (e.g. 'indices.npy')."""
    from numpy.lib import format as npformat

    with zipfile.ZipFile(npz_path) as zf, zf.open(member) as f:
        version = npformat.read_magic(f)
        if version[0] == 1:
            shape, _fortran, dtype = npformat.read_array_header_1_0(f)
        else:
            shape, _fortran, dtype = npformat.read_array_header_2_0(f)
        header_len = f.tell()  # bytes consumed by the .npy header within the member

    data_offset = _zip_member_data_offset(npz_path, member) + header_len
    return np.memmap(npz_path, mode="r", dtype=dtype, shape=shape, offset=data_offset)


# ── shared h5ad writing ────────────────────────────────────────────────────

def _finalize_h5ad(f, obs: pd.DataFrame, var: pd.DataFrame, uns: dict):
    """Write obs/var/uns and the root AnnData encoding attrs into an open h5py file."""
    f.attrs["encoding-type"] = "anndata"
    f.attrs["encoding-version"] = "0.1.0"
    write_elem(f, "obs", obs)
    write_elem(f, "var", var)
    write_elem(f, "uns", uns)
    # layers/obsm/varm/obsp/varp are required-empty in the AnnData spec
    for grp in ("layers", "obsm", "varm", "obsp", "varp"):
        write_elem(f, grp, {})


def _write_csr_group(f, data, indices, indptr, shape, *,
                     value_dtype, index_dtype, block: int = 64_000_000):
    """Create an AnnData 'X' CSR group, streaming data/indices in blocks."""
    nnz = int(indptr[-1])
    g = f.create_group("X")
    g.attrs["encoding-type"] = "csr_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = np.array(shape, dtype=np.int64)

    d_data = g.create_dataset("data", shape=(nnz,), dtype=value_dtype,
                              compression="gzip", chunks=(min(nnz, block) or 1,))
    d_idx = g.create_dataset("indices", shape=(nnz,), dtype=index_dtype,
                             compression="gzip", chunks=(min(nnz, block) or 1,))
    for s in range(0, nnz, block):
        e = min(s + block, nnz)
        d_data[s:e] = np.asarray(data[s:e], dtype=value_dtype)
        d_idx[s:e] = np.asarray(indices[s:e], dtype=index_dtype)
        log.info(f"  streamed {e:,}/{nnz:,} non-zeros")
    g.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64),
                     compression="gzip")


def _index_dtype_for(n_var: int):
    """Smallest unsigned int dtype that indexes n_var columns (cache uses same rule)."""
    return np.uint16 if n_var <= np.iinfo(np.uint16).max else np.uint32


def _clean_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Make an obs DataFrame h5ad-writable.

    Object/mixed-type columns (common in metadata.tsv, e.g. numbers + strings + NaN)
    cannot be serialized to HDF5, so coerce them to plain strings. Numeric and
    boolean columns are left untouched.
    """
    obs = obs.copy()
    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].astype(str)
    return obs


# ── proteomics: dense parquet -> h5ad ──────────────────────────────────────

def proteomics_to_h5ad(parquet_path: str, out_path: str,
                       metadata_tsv: str | None = None):
    import anndata as ad
    from scipy.sparse import csr_matrix

    parquet_path, out_path = str(parquet_path), str(out_path)
    log.info(f"[proteomics] reading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    id_cols = [c for c in df.columns if df[c].dtype == object]
    prot_cols = sorted(c for c in df.columns if c not in id_cols)
    log.info(f"  {df.shape[0]:,} samples, {len(prot_cols):,} proteins, id cols {id_cols}")

    # X: detected (>0) values only, float32 (preserves rank for binning)
    X = csr_matrix(df[prot_cols].to_numpy(dtype=np.float32))
    X.eliminate_zeros()

    # obs: id columns + joined metadata.tsv; obs_names = "<pxd>_<run>"
    obs = df[id_cols].astype(str).reset_index(drop=True)
    sample_id = obs[id_cols].agg("_".join, axis=1)
    if metadata_tsv and Path(metadata_tsv).exists():
        meta = pd.read_csv(metadata_tsv, sep="\t", low_memory=False).astype(
            {c: str for c in id_cols})
        obs = obs.merge(meta, on=id_cols, how="left")
        log.info(f"  joined {Path(metadata_tsv).name}: obs now {obs.shape[1]} columns")
    obs.index = pd.Index(sample_id, name="sample_id")
    obs = _clean_obs(obs)

    var = pd.DataFrame(index=pd.Index(prot_cols, name="accession"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["modality"] = "proteomics"
    adata.uns["value_semantics"] = "raw_abundance"
    adata.uns["source_file"] = Path(parquet_path).name

    density = X.nnz / (X.shape[0] * X.shape[1]) * 100
    log.info(f"  X: {X.shape[0]:,} x {X.shape[1]:,}, nnz={X.nnz:,} ({density:.1f}%)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    log.info(f"[proteomics] wrote {out_path}")


# ── transcriptomics: scipy .npz -> h5ad (streaming) ────────────────────────

def transcriptomics_to_h5ad(npz_path: str, out_path: str,
                            protein_columns_json: str,
                            metadata_parquet: str | None = None):
    import h5py

    npz_path, out_path = str(npz_path), str(out_path)
    with open(protein_columns_json) as fh:
        prot_cols = json.load(fh)
    n_var = len(prot_cols)

    # memory-map the CSR arrays without loading them fully
    data_mm = open_npz_member_memmap(npz_path, "data.npy")
    idx_mm = open_npz_member_memmap(npz_path, "indices.npy")
    indptr = np.array(open_npz_member_memmap(npz_path, "indptr.npy"))  # small, load fully
    shape = tuple(int(x) for x in np.array(open_npz_member_memmap(npz_path, "shape.npy")))
    n_obs, n_cols = shape
    assert n_cols == n_var, f"npz has {n_cols} columns but protein_columns has {n_var}"
    log.info(f"[transcriptomics] {npz_path}: {n_obs:,} x {n_var:,}, nnz={int(indptr[-1]):,}, "
             f"data dtype={data_mm.dtype}")

    # obs from metadata.parquet (or a bare RangeIndex if absent)
    if metadata_parquet and Path(metadata_parquet).exists():
        obs = pd.read_parquet(metadata_parquet).reset_index(drop=True)
        assert len(obs) == n_obs, f"metadata has {len(obs)} rows, matrix has {n_obs}"
    else:
        obs = pd.DataFrame(index=range(n_obs))
    obs.index = pd.Index([f"cell_{i}" for i in range(n_obs)], name="cell_id")
    obs = _clean_obs(obs)
    var = pd.DataFrame(index=pd.Index(prot_cols, name="accession"))

    uns = {"modality": "transcriptomics", "value_semantics": "umi_counts",
           "source_file": Path(npz_path).name}

    index_dtype = _index_dtype_for(n_var)
    value_dtype = data_mm.dtype  # keep native (uint8) values
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        _write_csr_group(f, data_mm, idx_mm, indptr, shape,
                         value_dtype=value_dtype, index_dtype=index_dtype)
        _finalize_h5ad(f, obs, var, uns)
    log.info(f"[transcriptomics] wrote {out_path}")


# ── in-memory CSR -> unified h5ad (for pipeline scripts) ───────────────────

def save_unified_h5ad(out_path: str, X, obs: pd.DataFrame, var_index,
                      modality: str, value_semantics: str):
    """Write an in-memory sparse matrix + obs/var as a unified `.h5ad` source.

    Used by the transcriptomics pipeline (build_sc_dataset / split_dataset) to emit
    the final format natively. Indices are downcast to the smallest unsigned dtype.

    Args:
        out_path: destination `.h5ad`.
        X: scipy sparse matrix (n_obs x n_var), raw values.
        obs: per-sample metadata; its index becomes obs_names.
        var_index: iterable of column accessions (becomes var_names).
        modality: "proteomics" | "transcriptomics".
        value_semantics: e.g. "umi_counts" | "raw_abundance".
    """
    import h5py

    X = X.tocsr()
    X.eliminate_zeros()
    n_var = X.shape[1]
    var = pd.DataFrame(index=pd.Index(list(var_index), name="accession"))
    obs = _clean_obs(obs)
    uns = {"modality": modality, "value_semantics": value_semantics}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        _write_csr_group(f, X.data, X.indices, X.indptr, X.shape,
                         value_dtype=X.dtype, index_dtype=_index_dtype_for(n_var))
        _finalize_h5ad(f, obs, var, uns)
    log.info(f"[{modality}] wrote {out_path}  "
             f"({X.shape[0]:,} x {n_var:,}, nnz={X.nnz:,})")


# ── feature (gene) accession -> protein accession remapping ────────────────

def load_feature_protein_map(map_path: str, in_model_only: bool = False) -> dict[str, str]:
    """Load a feature→protein accession map (e.g. transcriptomics/gene_protein_map.parquet).

    Returns {feature_accession: protein_accession}. With in_model_only=True, keep only
    rows whose protein is flagged present in the proteomics model's protein set.
    """
    df = pd.read_parquet(map_path)
    if in_model_only and "in_model" in df.columns:
        df = df[df["in_model"]]
    return dict(zip(df["ensembl_gene_id"].astype(str), df["uniprot_accession"].astype(str)))


def remap_features_to_proteins(in_h5ad: str, out_h5ad: str,
                               mapping: dict[str, str] | str,
                               in_model_only: bool = False) -> dict:
    """Remap a gene/transcript-accession h5ad to a protein-accession h5ad.

    Standalone data-prep step (not part of training): each feature (var) is mapped to
    a UniProt accession via `mapping` (a dict or a path to a mapping parquet). Features
    with **no mapping are dropped**. Multiple features mapping to the same protein are
    **summed** (gene→protein aggregation). Output var_names are the sorted unique protein
    accessions; obs is carried over unchanged.

    Returns a stats dict and logs how many features were dropped.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    if isinstance(mapping, (str, Path)):
        mapping = load_feature_protein_map(mapping, in_model_only=in_model_only)

    adata = ad.read_h5ad(in_h5ad)
    feats = [str(v).split(".")[0] for v in adata.var_names]   # strip version suffixes
    mapped = [mapping.get(f) for f in feats]

    n_total = len(feats)
    keep = [i for i, p in enumerate(mapped) if p is not None]
    n_dropped = n_total - len(keep)

    proteins = sorted({mapped[i] for i in keep})
    prot_idx = {p: j for j, p in enumerate(proteins)}

    # sparse projection P (n_features x n_proteins): kept feature i → its protein column.
    # X_proteins = X_features @ P  sums features that map to the same protein.
    rows = np.asarray(keep, dtype=np.int64)
    cols = np.asarray([prot_idx[mapped[i]] for i in keep], dtype=np.int64)
    P = csr_matrix((np.ones(len(keep), dtype=np.float64), (rows, cols)),
                   shape=(n_total, len(proteins)))

    X_in = adata.X.tocsr()
    integer_in = np.issubdtype(X_in.dtype, np.integer)
    X_out = (X_in.astype(np.float64) @ P).tocsr()
    if integer_in:  # keep counts compact; summed counts stay well within uint16
        X_out.data = np.clip(np.rint(X_out.data), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    else:
        X_out.data = X_out.data.astype(np.float32)
    X_out.eliminate_zeros()

    modality = adata.uns.get("modality", "transcriptomics")
    value_semantics = adata.uns.get("value_semantics", "umi_counts")
    save_unified_h5ad(str(out_h5ad), X_out, adata.obs, proteins,
                      modality=modality, value_semantics=value_semantics)

    n_merged = len(keep) - len(proteins)
    log.info(f"[remap] {in_h5ad} → {out_h5ad}")
    log.info(f"  {n_total} features → {len(proteins)} proteins  |  "
             f"dropped {n_dropped} unmapped ({n_dropped / max(n_total, 1) * 100:.1f}%), "
             f"merged {n_merged} features sharing a protein")
    return {"n_features_in": n_total, "n_dropped": n_dropped,
            "n_proteins_out": len(proteins), "n_merged": n_merged}


# ── CLI ─────────────────────────────────────────────────────────────────────

def convert(input_path: str, output_path: str, metadata: str | None = None,
            protein_columns: str | None = None):
    """Dispatch by input extension."""
    ext = Path(input_path).suffix.lower()
    if ext == ".parquet":
        meta = metadata
        if meta is None:
            cand = Path(input_path).parent / "metadata.tsv"
            meta = str(cand) if cand.exists() else None
        proteomics_to_h5ad(input_path, output_path, metadata_tsv=meta)
    elif ext == ".npz":
        if protein_columns is None:
            raise ValueError("--protein-columns is required for transcriptomics .npz input")
        transcriptomics_to_h5ad(input_path, output_path, protein_columns,
                                metadata_parquet=metadata)
    else:
        raise ValueError(f"Unknown input type '{ext}' (expected .parquet or .npz)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Convert legacy data files to unified .h5ad")
    p.add_argument("--input", required=True, help="Input .parquet (proteomics) or .npz (transcriptomics)")
    p.add_argument("--output", required=True, help="Output .h5ad path")
    p.add_argument("--metadata", default=None,
                   help="metadata.tsv (proteomics) or metadata.parquet (transcriptomics). "
                        "Proteomics auto-detects metadata.tsv next to the input.")
    p.add_argument("--protein-columns", default=None,
                   help="protein_columns.json (required for transcriptomics .npz)")
    p.add_argument("--map", default=None,
                   help="gene→protein mapping parquet. If set, remaps an input .h5ad's "
                        "features to protein accessions (unmapped features dropped) "
                        "instead of converting a legacy file.")
    p.add_argument("--in-model-only", action="store_true",
                   help="with --map: keep only proteins flagged present in the model set")
    args = p.parse_args()

    if args.map:
        remap_features_to_proteins(args.input, args.output, args.map,
                                   in_model_only=args.in_model_only)
    else:
        convert(args.input, args.output, metadata=args.metadata,
                protein_columns=args.protein_columns)
