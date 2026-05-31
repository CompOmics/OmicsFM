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
├── expression.npz          # scipy sparse CSR (n_cells × n_proteins), uint8
├── metadata.parquet        # per-cell metadata (dataset_id, cell_type, tissue, ...)
├── protein_columns.json    # ordered list of UniProt accessions (column labels)
└── config.yaml             # copy of the config used
```

### After `split_dataset.py`:

```
transcriptomics/split/
├── train.npz               # training expression matrix
├── train_metadata.parquet   # training cell metadata
├── valid.npz
├── valid_metadata.parquet
├── test.npz
└── test_metadata.parquet
```

Row i in `train.npz` corresponds to row i in `train_metadata.parquet`.

## Loading the data

```python
from scipy.sparse import load_npz
import pandas as pd
import json

# Load one split
X_train = load_npz("transcriptomics/split/train.npz")  # sparse CSR, loads in seconds
meta_train = pd.read_parquet("transcriptomics/split/train_metadata.parquet")

# Column labels (shared across splits)
with open("transcriptomics/output/protein_columns.json") as f:
    protein_cols = json.load(f)

print(X_train.shape)      # (n_cells, 20274)
print(X_train.dtype)       # uint8
print(len(protein_cols))   # 20274
```

## Training with ProtGPT

Set `modality: "transcriptomics"` in `config/training/setup.yaml`:

```yaml
data:
  modality: "transcriptomics"
  transcriptomics:
    train_path: transcriptomics/split/train.npz
    valid_path: transcriptomics/split/valid.npz
    test_path: transcriptomics/split/test.npz
    protein_columns_path: transcriptomics/output/protein_columns.json
```

For transfer learning (pretrain on transcriptomics, fine-tune on proteomics):

```yaml
# config for fine-tuning step
data:
  modality: "proteomics"
transfer:
  checkpoint_path: runs/pretrain_sc/best_model.ckpt
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
