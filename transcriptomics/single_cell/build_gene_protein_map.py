"""
Build a mapping from Ensembl gene IDs (ENSG) to UniProt accessions.

Uses mygene.info as primary source (fast, separates Swiss-Prot/TrEMBL),
with UniProt bulk download as fallback for unmapped genes.

Only keeps genes that map to proteins present in our proteomics model
(i.e., the var_names of the proteomics training .h5ad).

Output: transcriptomics/single_cell/gene_protein_map.parquet
    Columns: ensembl_gene_id, uniprot_accession, gene_symbol, reviewed
"""

import argparse
import logging
import re
import csv
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (transcriptomics/single_cell/ -> protgpt/)


def get_model_proteins(proteomics_path: str) -> set[str]:
    """Load the UniProt accessions the proteomics model is trained on.

    Reads them from a unified `.h5ad`'s var_names (the current format), or from a
    legacy dense parquet's columns (older format) for backward compatibility.
    """
    path = str(proteomics_path)
    if path.endswith(".h5ad"):
        import anndata as ad
        adata = ad.read_h5ad(path, backed="r")  # var_names only, no X load
        prots = {str(v) for v in adata.var_names}
        adata.file.close()
    else:  # legacy dense parquet
        all_cols = pd.read_parquet(path).columns.tolist()
        prots = {c for c in all_cols if c not in ("pxd_accession", "run_name")}
    log.info(f"Model proteome: {len(prots)} proteins from {Path(path).name}")
    return prots


def get_census_gene_ids() -> list[str]:
    """Fetch all human gene IDs from CELLxGENE Census."""
    try:
        import cellxgene_census
        with cellxgene_census.open_soma(census_version="stable") as census:
            var_df = (
                census["census_data"]["homo_sapiens"]
                .ms["RNA"]
                .var.read(column_names=["feature_id", "feature_name"])
                .concat()
                .to_pandas()
            )
        log.info(f"Census genes: {len(var_df)}")
        return var_df
    except ImportError:
        log.warning("cellxgene-census not installed, loading gene list from file")
        return None


def map_via_mygene(ensg_ids: list[str]) -> dict:
    """Map ENSG IDs to UniProt accessions using mygene.info."""
    import mygene
    mg = mygene.MyGeneInfo()

    log.info(f"Querying mygene.info for {len(ensg_ids)} genes...")
    results = mg.querymany(
        ensg_ids,
        scopes="ensembl.gene",
        fields="uniprot,symbol",
        species="human",
        returnall=True,
        verbose=False,
    )

    mapping = {}
    for hit in results["out"]:
        query = hit["query"]
        if "uniprot" not in hit:
            continue
        up = hit["uniprot"]
        symbol = hit.get("symbol", "")

        # Prefer Swiss-Prot (reviewed) over TrEMBL
        if "Swiss-Prot" in up:
            sp = up["Swiss-Prot"]
            accs = [sp] if isinstance(sp, str) else sp
            for acc in accs:
                mapping.setdefault(query, []).append({
                    "accession": acc, "reviewed": True, "symbol": symbol
                })
        if "TrEMBL" in up:
            tr = up["TrEMBL"]
            accs = [tr] if isinstance(tr, str) else tr
            for acc in accs:
                mapping.setdefault(query, []).append({
                    "accession": acc, "reviewed": False, "symbol": symbol
                })

    log.info(f"mygene.info mapped {len(mapping)}/{len(ensg_ids)} genes")
    return mapping


def map_via_uniprot_bulk(unmapped: list[str]) -> dict:
    """Fallback: download full human proteome cross-refs from UniProt."""
    if not unmapped:
        return {}

    log.info(f"Fetching UniProt bulk cross-refs for {len(unmapped)} unmapped genes...")
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": "(organism_id:9606)",
        "format": "tsv",
        "fields": "accession,xref_ensembl,reviewed,gene_names",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()

    ensg_to_up = defaultdict(list)
    reader = csv.DictReader(StringIO(r.text), delimiter="\t")
    for row in reader:
        ensembl_refs = row.get("Ensembl", "")
        for match in re.finditer(r"ENSG\d+", ensembl_refs):
            ensg_to_up[match.group()].append({
                "accession": row["Entry"],
                "reviewed": row["Reviewed"] == "reviewed",
                "symbol": row.get("Gene Names", "").split()[0] if row.get("Gene Names") else "",
            })

    unmapped_set = set(unmapped)
    mapping = {}
    for ensg in unmapped_set:
        if ensg in ensg_to_up:
            mapping[ensg] = ensg_to_up[ensg]

    log.info(f"UniProt bulk mapped {len(mapping)}/{len(unmapped)} additional genes")
    return mapping


def select_best_accession(entries: list[dict], model_proteins: set[str]) -> dict | None:
    """Pick the best UniProt accession, preferring those in our model's protein set."""
    # First priority: reviewed + in model
    for e in entries:
        if e["reviewed"] and e["accession"] in model_proteins:
            return e
    # Second: unreviewed + in model
    for e in entries:
        if e["accession"] in model_proteins:
            return e
    # Third: reviewed (not in model — won't have ESM-C embedding but keep for reference)
    for e in entries:
        if e["reviewed"]:
            return e
    # Last resort
    return entries[0] if entries else None


def build_mapping(proteomics_path: str, output_path: str):
    model_proteins = get_model_proteins(proteomics_path)

    # Get gene list from Census
    var_df = get_census_gene_ids()
    if var_df is not None:
        ensg_ids = var_df["feature_id"].tolist()
        gene_symbols = dict(zip(var_df["feature_id"], var_df["feature_name"]))
    else:
        raise RuntimeError("Cannot get gene list — install cellxgene-census")

    # Strip version suffixes (ENSG00000139618.17 → ENSG00000139618)
    ensg_ids_clean = [eid.split(".")[0] for eid in ensg_ids]
    ensg_ids_clean = list(set(ensg_ids_clean))

    # Step 1: mygene.info
    mygene_map = map_via_mygene(ensg_ids_clean)

    # Step 2: UniProt fallback for unmapped
    unmapped = [eid for eid in ensg_ids_clean if eid not in mygene_map]
    uniprot_map = map_via_uniprot_bulk(unmapped)

    # Merge
    full_map = {**mygene_map, **uniprot_map}

    # Select best accession per gene
    rows = []
    for ensg in ensg_ids_clean:
        if ensg not in full_map:
            continue
        best = select_best_accession(full_map[ensg], model_proteins)
        if best is None:
            continue
        rows.append({
            "ensembl_gene_id": ensg,
            "uniprot_accession": best["accession"],
            "gene_symbol": best.get("symbol", gene_symbols.get(ensg, "")),
            "reviewed": best["reviewed"],
            "in_model": best["accession"] in model_proteins,
        })

    result = pd.DataFrame(rows)
    in_model = result["in_model"].sum()
    log.info(f"Final mapping: {len(result)} genes → UniProt, {in_model} overlap with model proteome")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    log.info(f"Saved to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build Ensembl→UniProt gene-protein mapping")
    parser.add_argument("--proteomics", default=str(ROOT / "data/flashlfq_diann/train.h5ad"),
                        help="Proteomics training .h5ad (or legacy parquet) for the model protein list")
    parser.add_argument("--output", default=str(ROOT / "transcriptomics/single_cell/gene_protein_map.parquet"),
                        help="Output path for the mapping table")
    args = parser.parse_args()
    build_mapping(args.proteomics, args.output)


if __name__ == "__main__":
    main()
