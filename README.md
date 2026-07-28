# OmicsFM

A transformer foundation model for different omics modalities: proteomics and
transcriptomics. It is trained by masked modelling on expression bins — given the
proteins or genes detected in a sample, predict the abundance of the ones held out.
Two things fall out of that:

- **Sample embeddings** (`compute_sst`) — one vector per proteome or transcriptome,
  which clusters by tissue without ever being told the tissue.
- **Attention maps** (`attention_map`) — the transformer is the only place where
  information moves *between* proteins, so its attention weights rank which proteins
  the model treats as related, recovering known interactions (CORUM, STRING,
  Reactome, BioPlex).

Six checkpoints: proteomics, bulk transcriptomics and single-cell transcriptomics,
each with learned feature embeddings or with ESM-C protein sequence embeddings.

---

## Setup

Needs [git](https://git-scm.com/downloads) and
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). Identical on Windows,
Linux and macOS.

**1. Clone**

```bash
git clone https://github.com/CompOmics/OmicsFM.git
cd OmicsFM
```

**2. Create the environment** — Python 3.12, PyTorch (CUDA 12.8) and OmicsFM with
its extras. A few minutes and a few GB of wheels.

```bash
conda env create -f envs/environment.yml
conda activate omicsfm
```

No NVIDIA GPU? First delete the `--extra-index-url` line and the two `+cu128` pins
from `envs/environment.yml`, leaving plain `torch`.

**3. Install the package** — already done by step 2, via the `--editable ..[all]`
line in the YAML. Repeat it only if you built the environment by hand or moved the
repository. It also registers the `omicsfm` command.

```bash
pip install -e .
```

**4. Get the models and data** — nothing needs downloading up front;
`get_checkpoint("proteomics")` fetches on first use and reuses from disk after. The
CLI is for staging ahead of time: offline machines, HPC nodes, or warming the cache.

```bash
hf auth login                       # checkpoints + proteomics are private until publication
omicsfm download                    # models tier
omicsfm download models test --yes  # models + test splits
```

| tier     | contents                                  | size    |
|----------|-------------------------------------------|---------|
| `models` | 6 checkpoints, their configs, ESM-C cache | 0.5 GB  |
| `test`   | test split of each of the 5 datasets      | 8.9 GB  |
| `valid`  | validation splits                         | 5.6 GB  |
| `train`  | training splits                           | 80.4 GB |

Over 1 GB needs `--yes`; `--dry-run` prints the plan only; re-running skips what is
already there. Files land in `model/<name>/` and `data/<name>/`, named as on the Hub.
Set `OMICSFM_HOME` to keep them elsewhere — another drive, or shared scratch so a
whole lab uses one copy.

For a single artifact, skip the CLI:

```python
from omicsfm.hub import get_checkpoint, get_dataset

ckpt = get_checkpoint("proteomics_esmc")            # 111 MB
data = get_dataset("proteomics_uniprot", "test")    #  34 MB
```

---

## First run

Open `notebooks/api.ipynb` — it loads a checkpoint, computes sample embeddings, runs
the masked-modelling evaluation, draws a tissue-coloured UMAP and extracts an
attention map, downloading the two files it needs as it goes.

## Repository layout

```
omicsfm/          the package
  api.py            compute_sst, predict, visualize_sst, attention_map
  attention.py      attention extraction, PPI ground truth, enrichment curves
  data.py           ExpressionDataset: .h5ad -> binned tensors
  architecture.py   the transformer
  train.py          training loop
  convert.py        raw tables -> .h5ad
  esmc_utils.py     ESM-C sequence embeddings
  hub.py            artifact resolution and download
notebooks/        api.ipynb and the dataset-inspection notebooks
reference/        PPI ground-truth matrices and the notebooks that rebuild them
config/           setup.yaml, the training configuration template
envs/             environment.yml (float) and environment.lock.yml (exact)
fasta/            human proteome FASTA backing the ESM-C embeddings
model/  data/     created by the downloads; not in git
```

### Training

Edit `config/setup.yaml` — paths, `num_bins`, model size, optimiser — then:

```bash
python -m omicsfm.train --config config/setup.yaml
```

Runs log to Weights & Biases if you are logged in.

### Reference matrices

`reference/*.npz` are the PPI ground truths used to score attention, tracked in git
(17 MB, sparse uint16) so scoring runs against a fixed snapshot. `reference/notebooks/`
rebuilds them from source, but these databases change continuously, so a rebuild will
not reproduce the published numbers — the builders refuse to overwrite an existing
`.npz` for that reason.

### Experiments and dataset construction

Not in this repository. Each experiment is a self-contained bundle on Zenodo with its
own environment file, inputs and one command that reproduces its table or figure. The
transcriptomics corpus-building pipeline (ARCHS4 and CELLxGENE download,
normalisation, gene mapping, splitting) ships the same way.

## Citation

Manuscript in preparation. Licensed MIT.
