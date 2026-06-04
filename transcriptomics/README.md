# Transcriptomics Data Pipeline

Downloads single-cell RNA-seq data from [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) and emits sparse expression matrices ready for model training — either as native genes or projected into the proteomics model's protein space.

This enables pretraining ProtGPT on millions of single-cell transcriptomics samples

## How it works

1. **Fetch** SC expression data from CELLxGENE Census for human tissues
2. **Sample** cells per tissue (diversity-weighted by cell type)
3. **QC filter** cells by minimum genes detected, UMI count, and mitochondrial fraction
4. **Derive** one or more *gene sets* from each tissue's QC'd matrix (see below)
5. **Output** a unified `expression.h5ad` per gene set (sparse CSR X + obs + var)
6. **Split** into train/valid/test by `dataset_id` (grouped, no project leakage)

A single Census stream per tissue feeds every requested gene set, so producing more than one dataset is nearly free.

### Gene sets (`datasets_to_build`)

| Mode | Columns | Description |
|------|---------|-------------|
| `all_transcripts` | native Ensembl gene ids | every Census gene that survives QC |
| `protein_mapped`  | UniProt accessions | gene→protein projection `X @ P` (genes summed per protein) — directly comparable to the proteomics model |

The `protein_mapped` projection uses a sparse matrix `P` mapping Census gene indices to model protein columns: `X_proteins = X_genes @ P` does gene-to-protein translation and multi-gene aggregation (e.g. the ~14 histone-H4 gene copies sum into one `P62805` column) in one step, with no dense intermediaries.

## Quick start

All configuration lives in `pipeline_config.yaml`. Both scripts read from it.

```bash
# Step 1: Build the dataset(s) (~2-5M cells, ~1-2 h with 8 workers)
python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.yaml

# Step 2: Split each built dataset into train/valid/test
python transcriptomics/split_dataset.py --config transcriptomics/pipeline_config.yaml
```

Test mode (3 tissues, 1k cells each — good for verifying the pipeline works):
```bash
python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.yaml --test
python transcriptomics/split_dataset.py    --config transcriptomics/pipeline_config.yaml
```

Override which gene sets to build without editing the config:
```bash
python transcriptomics/build_sc_dataset.py --config ... --datasets all_transcripts protein_mapped
```

## Scripts

| Script | Purpose |
|--------|---------|
| `build_sc_dataset.py` | Main pipeline: fetch from Census, QC, derive gene sets, save |
| `split_dataset.py` | Grouped-by-`dataset_id` train/valid/test split, per dataset |
| `build_gene_protein_map.py` | Build Ensembl → UniProt mapping (called automatically, cached) |

## Configuration

Everything is in `pipeline_config.yaml`:

```yaml
census_version: "stable"         # CELLxGENE Census snapshot
output_dir: transcriptomics/output
proteomics_reference: data/flashlfq_diann/train.h5ad  # defines the model protein set

datasets_to_build:               # one subdirectory per entry under output_dir
  - all_transcripts
  # - protein_mapped

tissues:                         # 27 tissues to fetch
  - blood
  - brain
  - ...

cells_per_tissue: 200000         # max cells per tissue
sampling_strategy: diversity     # oversample rare cell types
dtype: uint8                     # expression values (UMI counts 0-255)
workers: 8                       # parallel Census connections

qc:
  min_genes_per_cell: 200
  min_umi_per_cell: 500
  max_mito_fraction: 0.20

split:
  output_dir: transcriptomics/split
  train_frac: 0.90
  valid_frac: 0.05
  test_frac: 0.05
```

**`proteomics_reference`** is read for its `var_names` (UniProt accessions): they define which proteins the `protein_mapped` set targets, and in what order. This keeps the transcriptomics and proteomics data on the same protein set — required for transfer learning. (A legacy dense `.parquet` is also accepted.)

## Output

### After `build_sc_dataset.py`

One subdirectory per built gene set:

```
transcriptomics/output/
├── all_transcripts/
│   ├── expression.h5ad     # AnnData: sparse CSR X (uint8) + obs + var (Ensembl ids)
│   └── config.yaml         # copy of the config used
└── protein_mapped/         # only if enabled in datasets_to_build
    ├── expression.h5ad     #   var = UniProt accessions (genes summed per protein)
    └── config.yaml
```

### After `split_dataset.py`

```
transcriptomics/split/
├── all_transcripts/
│   ├── train.h5ad          # one self-describing file per split
│   ├── valid.h5ad          #   (X + obs metadata + var accessions bundled)
│   └── test.h5ad
└── protein_mapped/         # only if enabled in datasets_to_build
    ├── train.h5ad
    ├── valid.h5ad
    └── test.h5ad
```

Everything (matrix, per-cell metadata, gene/protein accessions) lives in one `.h5ad` file — the unified format consumed by `protgpt.data.ExpressionDataset`.

## Loading the data

```python
import anndata as ad

A = ad.read_h5ad("transcriptomics/split/all_transcripts/train.h5ad")  # or backed="r"
print(A.shape)                # (n_cells, n_features)
print(A.X.dtype)              # uint8  (raw UMI counts)
print(list(A.var_names[:3]))  # Ensembl gene ids (or UniProt for protein_mapped)
print(A.obs.columns.tolist()) # per-cell metadata (dataset_id, cell_type, tissue, ...)
```

## Training with ProtGPT

Point `config/training/setup.yaml` at the split h5ad files (modality is read from the
file itself, so set `detect_groups: false` for transcriptomics):

```yaml
data:
  train_path: transcriptomics/split/all_transcripts/train.h5ad
  valid_path: transcriptomics/split/all_transcripts/valid.h5ad
  test_path:  transcriptomics/split/all_transcripts/test.h5ad
  num_bins: 10
  detect_groups: false   # no protein groups for transcriptomics
```

For transfer learning (pretrain on transcriptomics, fine-tune on proteomics), point the
fine-tuning config at the proteomics h5ad and load the pretrained checkpoint:

```yaml
data:
  train_path: data/nsaf_diann/train.h5ad
  detect_groups: true
transfer:
  checkpoint_path: model/pretrain_sc/best_model.ckpt
```

## Sampling strategy

When `sampling_strategy: diversity`, cells are sampled inversely proportional to cell type frequency within each tissue. This prevents common types (e.g., T cells in blood) from dominating the training set and ensures rare cell types are well-represented.

## Gene-protein mapping

Built automatically on first run and cached at `gene_protein_map.parquet`:
- ~61k Census genes → ~19k mapped to UniProt
- Swiss-Prot (reviewed) prioritized over TrEMBL
- Multiple genes mapping to the same protein are summed (in `protein_mapped`)

## Split strategy

`split_dataset.py` splits **by project** (`dataset_id`): whole datasets are shuffled
deterministically (from `seed`) and assigned to train/valid/test by fraction, so no
project leaks across splits. The assignment depends only on the sorted dataset ids and
the seed, so every gene set in `datasets_to_build` gets the **same cells in the same
split** — keeping the gene-level and protein datasets cell-aligned. A leakage check
runs after every split.

## Dependencies

```
cellxgene-census
tiledbsoma
mygene
anndata
h5py
pandas
pyarrow
scipy
numpy
pyyaml
tqdm
```
