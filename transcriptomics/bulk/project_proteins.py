"""
Project bulk `all_transcripts` (HGNC-symbol genes) into the proteomics protein space.

Decoupled, streaming step: reads the already-built all_transcripts/expression.h5ad,
streams X in row-chunks, applies X @ P (genes SUMMED per protein), and writes
protein_mapped/expression.h5ad (samples x UniProt proteins) in the SAME protein set +
order as the proteomics model and the single-cell `protein_mapped` — so all three FMs
share one protein space. Never loads the full matrix; obs is copied verbatim.

P (gene-symbol -> protein) reuses single_cell/gene_protein_map.parquet's
`gene_symbol -> uniprot_accession` pairs (in_model), so bulk and single-cell project onto
identical protein definitions. Multiple genes mapping to one protein are summed by the
matmul (e.g. histone copies -> one accession), exactly like the single-cell arm.

Usage:
    python transcriptomics/bulk/project_proteins.py --config transcriptomics/bulk/pipeline_config.yaml
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import h5py
from anndata.io import read_elem, write_elem
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

_SUFFIX = re.compile(r"-\d+$")   # the dedup suffix added by build (TSPAN6-1 -> TSPAN6)


def _add_file_log(output_dir, name):
    """Also write logs to <output_dir>/logs/<name>.log so logs live with the bulk artifacts."""
    logdir = Path(output_dir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)


def protein_order(ref_h5ad: Path):
    """UniProt accessions (+ order) that define the model protein columns."""
    with h5py.File(ref_h5ad, "r") as f:
        var = read_elem(f["var"])
    return list(var.index)


def build_P(gene_symbols, map_path: Path, proteins):
    """Sparse (n_genes x n_proteins) 0/1 map: gene symbol -> model protein column."""
    m = pd.read_parquet(map_path)
    if "in_model" in m.columns:
        m = m[m["in_model"]]
    prot_idx = {p: i for i, p in enumerate(proteins)}
    sym2uni = defaultdict(list)
    for s, u in m[["gene_symbol", "uniprot_accession"]].dropna().itertuples(index=False):
        sym2uni[str(s).upper()].append(u)

    rows, cols = [], []
    base = [_SUFFIX.sub("", s).upper() for s in gene_symbols]  # strip dedup suffix for lookup
    for gi, s in enumerate(base):
        for u in sym2uni.get(s, ()):
            pi = prot_idx.get(u)
            if pi is not None:
                rows.append(gi)
                cols.append(pi)
    P = csr_matrix((np.ones(len(rows), dtype=np.uint32), (rows, cols)),
                   shape=(len(gene_symbols), len(proteins)))
    genes_hit = len(set(rows))
    prots_hit = len(set(cols))
    log.info(f"P: {len(gene_symbols)} genes x {len(proteins)} proteins | "
             f"{genes_hit} genes map ({genes_hit/len(gene_symbols):.1%}), "
             f"{prots_hit}/{len(proteins)} proteins covered, {len(rows)} edges")
    return P


def main():
    parser = argparse.ArgumentParser(description="Project bulk all_transcripts -> protein_mapped")
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunk-rows", type=int, default=4000, help="X rows read/projected per block")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / cfg["output_dir"]
    _add_file_log(out_dir, "project")
    src = out_dir / "all_transcripts" / "expression.h5ad"
    dst = out_dir / "protein_mapped" / "expression.h5ad"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — build all_transcripts first.")
    dst.parent.mkdir(parents=True, exist_ok=True)

    proteins = protein_order(ROOT / cfg["proteomics_reference"])
    with h5py.File(src, "r") as f:
        gene_symbols = list(read_elem(f["var"]).index)
        obs = read_elem(f["obs"])
        n = f["X"].shape[0]
    log.info(f"Source: {src}  ({n} samples x {len(gene_symbols)} genes)")

    P = build_P(gene_symbols, ROOT / cfg["gene_protein_map"]["cache_path"], proteins)
    n_prot = len(proteins)
    dtype = np.uint32

    # Stream: read X rows, project (sum genes per protein), write dense gzip.
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        Xin = fin["X"]
        Xout = fout.create_dataset("X", shape=(n, n_prot), chunks=(64, n_prot),
                                   compression="gzip", compression_opts=4, dtype=dtype)
        Xout.attrs["encoding-type"] = "array"
        Xout.attrs["encoding-version"] = "0.2.0"
        for r0 in range(0, n, args.chunk_rows):
            r1 = min(r0 + args.chunk_rows, n)
            block = np.asarray(Xin[r0:r1])                    # (m x genes) uint32
            proj = (P.T @ block.T).T                          # (m x proteins), summed
            Xout[r0:r1, :] = np.asarray(proj, dtype=dtype)
            if (r0 // args.chunk_rows) % 10 == 0:
                log.info(f"  projected rows {r0}-{r1} / {n}")
        fout.attrs["encoding-type"] = "anndata"
        fout.attrs["encoding-version"] = "0.1.0"
        write_elem(fout, "obs", obs)
        write_elem(fout, "var", pd.DataFrame(index=pd.Index(proteins, name=None)))

    with open(dst.parent / "config.yaml", "w") as cf:
        yaml.safe_dump(cfg, cf, sort_keys=False)
    log.info(f"Wrote {dst}  ({n} samples x {n_prot} proteins, dense gzip)")


if __name__ == "__main__":
    main()
