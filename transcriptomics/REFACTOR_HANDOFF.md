# Transcriptomics pipeline — refactor handoff

**Purpose:** carry full context from a Windows Claude Code session to the Linux build
machine. The Census download (`cellxgene-census` / `tiledbsoma`) cannot run on Windows
(no prebuilt `tiledbsoma.dll`), so the build must run on Linux. On Linux: `git pull`,
open Claude Code in the repo root, and point it at this file.

---

## Part 1 — Bug already found & FIXED (do not redo)

**Symptom:** the transcriptomics model's val loss was suspiciously low (6.55) vs the
older run (7.96). Proteomics retrained identically, so the model code was fine — the
difference was in the transcriptomics data/binning.

**Root cause:** rank-binning tie-break. Raw scRNA-seq UMI counts are ~99% ties
(~70% of detected genes have count = 1 — confirmed **intrinsic** on untouched 10x
PBMC3k: 98.7% tied, value=1 = 69.8%). Binning assigns each detected gene a rank
`1..n` then `bin = ceil(rank/n * B)`. With nearly everything tied, the bins are
decided by *how ties are broken*:
- **Old code** (`np.argsort`, unstable): tie order wobbled per cell → honest, high loss.
- **New code** (`np.lexsort`, stable): ties broken by **fixed gene column index** →
  a gene's bin became reproducible across cells → the model memorised a
  non-biological ordering → artificially low loss.

Proteomics was unaffected because its float abundances have ~no ties — confirming the
diagnosis.

**Fix applied in `protgpt/data.py`:**
- `_segmented_bins(vals, lengths, num_bins, rng)` now breaks ties **randomly**
  (`np.lexsort((tie, vals, rows))`). Equal-frequency bin structure preserved; tied
  genes scattered unpredictably (the scGPT approach).
- `ExpressionDataset(..., bin_seed=0)` — seed frozen into the cache, added to cache meta.
- `CACHE_VERSION` 2 → 3 so existing caches auto-rebuild.
- Verified: within-gene bin variance 2.63 (artifact) → 5.25 (honest); bins stay
  balanced (~equal members per bin); different seeds give different tie assignments.

**Expectation on retrain:** transcriptomics loss will *rise* back toward ~7.96 — that's
the correct, honest level. Proteomics curve unchanged.

**Settled data facts (don't relitigate):**
- `uint8` storage is fine — only 0.015% of values hit the 255 ceiling.
- Per-cell normalization (CPM / log1p) does **not** change rank-bins (monotonic →
  same ranks). Only **per-gene** normalization (Pearson residuals, count ÷ gene-mean)
  breaks within-cell ties.
- The discreteness is real measurement shot noise; learnable signal lives in gene
  identity, detection (zero vs non-zero), co-expression, and the high-count minority.
- Optional future upgrades (independent of the tie fix): HVG selection, more bins, or a
  negative-binomial count head instead of binning.

---

## Part 2 — Refactor goals (this is the task)

Rebuild the transcriptomics data cleanly. Locked decisions:

- **Cell selection:** keep the 27-tissue, cell-type-stratified ("diversity") sampling
  already in `build_sc_dataset.py`.
- **Metadata:** save per-cell `obs`.
- **Format:** unified `.h5ad` via `protgpt.convert.save_unified_h5ad`.
- **Datasets to produce:**
  1. **Gene-level** — native Ensembl gene ids. Gene set is a config knob:
     `all_transcripts` (~61k) or `protein_coding` (~19–20k). (Still pick which, or build both.)
  2. **`protein_mapped` (SUMMED)** — gene→protein projection `X @ P`, UniProt columns.
     Genes sharing a protein are **summed** into one column. KEEP this.
  - **DROPPED:** the gene-level "proteome_mapped" variant (keep only proteome-mapped genes,
    native gene ids). Reason: ~206 of the mapped genes collapse onto already-used proteins
    (≈200 proteins encoded by ≥2 genes), so that view gives multiple gene columns for one
    protein with different abundances — ambiguous. Summing (the projection) is the
    principled way to protein space, so we use that instead.
- **Split:** grouped by **project** (`dataset_id`), **no leakage**, 90/5/5, random —
  the proteomics-style "split by project". Replaces the current greedy
  tissue-stratified split.
- **Build env:** Linux (Census). **Old ~190 GB Windows data deleted** (done in this
  session — see Part 4).

---

## Part 3 — Concrete plan (files in `transcriptomics/`)

- **`pipeline_config.yaml`** — clean up; fix the stale `proteomics_reference`
  (`data/split_v2/train.parquet` is missing). Point it at the real proteomics `.h5ad`
  (e.g. `data/flashlfq_diann/train.h5ad`) and read protein columns from its
  `var_names`. Add `gene_set` and `datasets_to_build` options.
- **`build_gene_protein_map.py`** — read the model protein set from the proteomics
  `.h5ad` `var_names` instead of the missing parquet. Mapping logic otherwise unchanged.
- **`build_sc_dataset.py`** — refactor: support the `gene_set` modes; cleanly separate
  fetch / QC / output; emit native gene ids for the gene-level dataset and the
  projection for the protein dataset. Remove the in-script split (all splitting lives in
  `split_dataset.py`).
- **`split_dataset.py`** — replace the greedy tissue-stratified split with a clean
  grouped-by-`dataset_id` 90/5/5 split (shuffle datasets, assign whole datasets to
  splits, verify no `dataset_id` leakage). Save `obs` + `.h5ad` per split.
- **`README.md`** — update for the two datasets + new split.

**Run order on Linux:**
```bash
python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.yaml
python transcriptomics/split_dataset.py    --config transcriptomics/pipeline_config.yaml
```
Then point `config/setup.yaml` at the chosen split `.h5ad` files and train.

---

## Part 4 — Cleanup done in the Windows session

Deleted (authorized): `transcriptomics/output/` (expression.npz + metadata),
`transcriptomics/split/` (all `*.npz`, `*.h5ad`, `*_metadata.parquet`, `cache_*`),
scratch (`_tie_check.py`, `__pycache__`, `data/_pbmc3k.tar.gz`).
**Kept:** `gene_protein_map.parquet` (small, reusable), all `*.py` scripts,
`pipeline_config.yaml`, `README.md`, `inspect_data.ipynb`.

The `protgpt/data.py` binning fix (Part 1) is committed and travels with the repo.
