# Transcriptomics Data Pipeline

Downloads single-cell RNA-seq data from [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/), translates gene expression to protein space, and outputs sparse matrices ready for model training.

This enables pretraining ProtGPT on millions of single-cell transcriptomics samples before fine-tuning on proteomics data (transfer learning).

## How it works

1. **Fetch** SC expression data from CELLxGENE Census for human tissues
2. **Map** Ensembl gene IDs to UniProt protein accessions (matching the proteomics model's protein set)
3. **QC filter** cells by minimum genes detected, UMI count, and mitochondrial fraction
4. **Project** the gene expression matrix to protein space via a sparse matrix multiplication
5. **Output** a scipy sparse matrix (`.npz`) + cell metadata (`.parquet`)
6. **Split** into train/valid/test by dataset ID (no data leakage, tissue-stratified)

The key trick: a sparse projection matrix `P` maps Census gene indices directly to model protein columns. `X_proteins = X_genes @ P` does gene-to-protein translation and multi-gene aggregation in one step, with no dense intermediaries.

## Quick start

All configuration lives in `pipeline_config.yaml`. Both scripts read from it.

```bash
# Step 1: Build the dataset (~2-5M cells, takes 1-2 hours with 8 workers)
python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.yaml

# Step 2: Split into train/valid/test
python transcriptomics/split_dataset.py --config transcriptomics/pipeline_config.yaml
```

Test mode (3 tissues, 1k cells each — good for verifying the pipeline works):
```bash
python transcriptomics/build_sc_dataset.py --config transcriptomics/pipeline_config.yaml --test
```

## Scripts

| Script | Purpose |
|--------|---------|
| `build_sc_dataset.py` | Main pipeline: fetch from Census, QC, project to protein space, save |
| `split_dataset.py` | Split into train/valid/test with tissue-stratified balancing |
| `build_gene_protein_map.py` | Build Ensembl → UniProt mapping (called automatically, cached) |

## Configuration

Everything is in `pipeline_config.yaml`:

```yaml
census_version: "stable"         # CELLxGENE Census snapshot
output_dir: transcriptomics/output
proteomics_reference: data/split_v2/train.parquet  # defines which proteins to include

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

**`proteomics_reference`** is important: the pipeline reads the column names from this parquet to determine which proteins (UniProt accessions) appear in the output and in what order. This ensures the transcriptomics and proteomics data share the same protein set, which is required for transfer learning.

## Output

### After `build_sc_dataset.py`:

```
transcriptomics/output/
├── expression.h5ad         # AnnData: sparse CSR X (uint8) + obs (cell metadata) + var (accessions)
└── config.yaml             # copy of the config used
```

### After `split_dataset.py`:

```
transcriptomics/split/
├── train.h5ad              # one self-describing file per split
├── valid.h5ad              #   (X + obs metadata + var accessions bundled)
└── test.h5ad
```

Everything (matrix, per-cell metadata, protein/gene accessions) lives in one
`.h5ad` file per split — the unified format consumed by `protgpt.data.ExpressionDataset`.

## Loading the data

```python
import anndata as ad

A = ad.read_h5ad("transcriptomics/split/train.h5ad")   # or backed="r" for out-of-core
print(A.shape)                # (n_cells, 20274)
print(A.X.dtype)              # uint8  (raw UMI counts)
print(list(A.var_names[:3]))  # UniProt accessions (column labels)
print(A.obs.columns.tolist()) # per-cell metadata (dataset_id, cell_type, tissue, ...)
```

## Training with ProtGPT

Point `config/training/setup.yaml` at the split h5ad files (modality is read from the
file itself, so set `detect_groups: false` for transcriptomics):

```yaml
data:
  train_path: transcriptomics/split/train.h5ad
  valid_path: transcriptomics/split/valid.h5ad
  test_path:  transcriptomics/split/test.h5ad
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
- 61,497 Census genes → 19,411 mapped to UniProt
- 18,945 / 20,274 model proteins covered (93.4%)
- Swiss-Prot (reviewed) prioritized over TrEMBL
- Multiple genes mapping to the same protein are summed

## Split strategy

`split_dataset.py` uses greedy tissue-stratified splitting:
1. Reserve 1 dataset per tissue for train (largest), valid (smallest), and test
2. Greedily assign remaining datasets to whichever split has the largest cell deficit

This ensures every split has cells from every tissue and cell counts are close to the target ratios.

## Dependencies

```
cellxgene-census
tiledbsoma
mygene
pandas
pyarrow
scipy
numpy
pyyaml
tqdm
```
