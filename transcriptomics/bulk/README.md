# Bulk Transcriptomics Pipeline (ARCHS4)

Builds large-scale **human bulk RNA-seq** datasets for pretraining a bulk-transcriptomics
foundation model, as the bulk counterpart to the single-cell arm (`../single_cell/`,
CELLxGENE Census). Source is **[ARCHS4](https://archs4.org/)** — ~1M+ uniformly
processed human bulk samples (kallisto, **raw estimated counts**) in one HDF5.

> **Status: skeleton.** `build_bulk_dataset.py` has the structure, config wiring, and
> output contract in place; the ARCHS4 HDF5 I/O and metadata harmonization are stubbed
> (`# TODO(archs4)` / `# TODO(metadata)`).

## Why ARCHS4

| Need | ARCHS4 |
|------|--------|
| Human, at scale | ~1M+ bulk RNA-seq samples |
| Raw-ish counts | gene-level **estimated counts** (not TPM) |
| Many tissues | broad GEO/SRA coverage (free-text tissue labels) |
| Slice-able like AnnData | single HDF5, sliced locally (`archs4py`) |

For **clean tissue labels** (eval / supervised arm), complement with **GTEx + TCGA via
recount3** — see the project notes. ARCHS4 is the scale corpus; recount3/GTEx is the
trustworthy-label corpus.

## Output contract (shared with single_cell)

Identical to the single-cell arm, so the same downstream binning/splitting applies:

```
transcriptomics/bulk/output/            # all bulk artifacts live here (gitignored)
├── archs4/
│   └── human_gene_v2.latest.h5  # raw ARCHS4 source (~58 GB), downloaded here
├── logs/                        # build.log / project.log / normalize.log / download.log
├── metadata_raw.parquet         # pristine raw obs snapshot (immutable; never harmonized)
├── normalization_map.tsv        # metadata QC: original -> normalized, per field (step 2; see below)
├── all_transcripts/
│   ├── expression.h5ad          # AnnData: DENSE uint32 X (raw counts, gzip) + obs + var (HGNC symbols)
│   └── config.yaml
└── protein_mapped/              # X @ P projection into the proteomics protein space (UniProt)
    ├── expression.h5ad
    └── config.yaml
```

**X is stored dense `uint32` + gzip, not sparse CSR.** Bulk is ~46% dense, so CSR's
per-nonzero index overhead would balloon the file well past the ~58 GB source; dense
`uint32` matches it (~45–65 GB) and is exact across the full count dynamic range
(`uint32` is exact to 4.29B; counts reach tens of millions — past float32's 2²⁴ exact
limit, so float32 would *round* high-abundance genes). The matrix is streamed row-chunk
by row-chunk into a resizable gzip dataset, so RAM stays flat (~0.5 GB/block) despite a
~294 GB dense-in-RAM full size.

`obs` carries per-sample metadata (`geo_accession`, `series_id`, free-text
`source_name_ch1`/`characteristics_ch1`/`title`, plus `tissue_harmonized` /
`disease_harmonized` after step 2). Splits group by **`series_id`** (GSE / SRA study) so
no study leaks across train/valid/test — the bulk analogue of the single-cell "split by
`dataset_id`".

## Usage — two decoupled steps

The expensive matrix build and the (iteratively tuned) metadata harmonization are
**separate scripts**, so the vocabulary can be re-tuned without re-streaming 58 GB:

```bash
# 1. Build the raw h5ad (slow, one-time): streams ARCHS4 -> dense uint32 + raw obs
python transcriptomics/bulk/build_bulk_dataset.py  --config transcriptomics/bulk/pipeline_config.yaml
#    (smoke test first: add --n-random 500)

# 2. Harmonize metadata IN PLACE (fast, re-runnable): reads only obs, never loads X
python transcriptomics/bulk/normalize_metadata.py  --config transcriptomics/bulk/pipeline_config.yaml
#    tune-only (write the QC map but don't modify the h5ad): add --dry-run
```

Step 2 reads only the `obs` group, adds `*_harmonized` columns, and writes `obs` back in
place — X and var are untouched (verified by md5). Grow `tissues.txt`, re-run step 2,
repeat; the matrix is never rebuilt.

## Metadata harmonization

ARCHS4 sample metadata is free-text and noisy. We map it to the **same controlled
vocabulary as the proteomics + single-cell FMs**, reusing the SapBERT normalizer in
[`../../agentic-metadata`](../../agentic-metadata):

1. (optional) **MetaSRA** join on GSM/SRS ids for pre-curated ontology terms.
2. **SapBERT** entity-linking for the rest (NER span → nearest vocab term, abstain
   below `similarity_threshold`).
3. **Cellosaurus** check → `is_cell_line` flag (excluded by default for a tissue FM).

Because all three modalities normalize to one ontology, tissue/disease labels are
aligned across the proteomics, single-cell, and bulk FMs by construction.

**QC artifact — `output/normalization_map.tsv`.** Every mapping is logged (reusing
`agentic-metadata`'s `NormalizationTracker`) and written as a sorted TSV:

| field | raw_value | normalized_to | similarity | count | status |
|-------|-----------|---------------|-----------|-------|--------|
| tissue | hepatic tissue | liver | 0.91 | 1240 | mapped |
| tissue | sample 1 |  | 0.42 | 95 | unmapped |

Mapped rows first, then `unmapped`, count-descending. Scan the low-similarity and
`unmapped` rows by hand to spot bad matches and missing vocabulary terms, then grow the
`.txt` ontologies in `../../agentic-metadata` and re-run.

## TODO

- [x] Streaming matrix build (`build_bulk_dataset.py`) — dense `uint32` + gzip, on-the-fly QC + single-cell filter.
- [x] Separate, re-runnable metadata harmonization (`normalize_metadata.py`) — obs-only, in place, emits `normalization_map.tsv`.
- [ ] Grow `tissues.txt` / add disease terms from the QC map (first pass maps only ~33% tissue, ~12% disease; `brain cortex → cortex of kidney` is a known misfire).
- [ ] NER span-extraction before SapBERT (ARCHS4 free-text mixes tissue/disease/cell-line/treatment in one field) + MetaSRA join + Cellosaurus `is_cell_line` flag.
- [ ] `protein_mapped` projection — build a **symbol→UniProt** map (ARCHS4 var is HGNC symbols, not Ensembl) for `X @ P`.
- [ ] `split_dataset.py` for bulk (group by `series_id`) — adapt from `../single_cell/split_dataset.py`.
- [ ] Decide ARCHS4 vs recount3/GTEx roles for pretrain vs labeled eval.
