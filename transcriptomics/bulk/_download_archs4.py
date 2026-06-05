"""One-off: download the latest ARCHS4 human gene-level HDF5 into the bulk output dir.

Run with the prot_gpt interpreter. ~58 GB — expect a long download.
Note: archs4py's download.counts() returns None (no return statement), so we reference
the known output path rather than its return value.
"""
import os
import archs4py as a4

OUT_DIR = "transcriptomics/bulk/output/archs4"
os.makedirs(OUT_DIR, exist_ok=True)
print("Starting ARCHS4 human GENE_COUNTS (latest) download -> data/archs4/ ...", flush=True)
a4.download.counts("human", path=OUT_DIR, type="GENE_COUNTS", version="latest")
expected = os.path.join(OUT_DIR, "human_gene_v2.latest.h5")
if os.path.exists(expected):
    print(f"DOWNLOADED: {expected}  ({os.path.getsize(expected) / 1e9:.1f} GB)", flush=True)
else:
    print(f"Download finished but {expected} not found; check {OUT_DIR}", flush=True)
