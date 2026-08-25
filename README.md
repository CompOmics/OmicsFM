# OmicsFM

A transformer foundation model for different omics modalities: proteomics and
transcriptomics. It is trained by masked modelling on expression bins given the
proteins or genes detected in a sample, predict the abundance of the ones held out.
Two things fall out of that:

- **Sample embeddings** (`compute_sst`) — one vector per proteome or transcriptome,
  which clusters by tissue without ever being told the tissue.
- **Attention maps** (`attention_map`) — the transformer is the only place where
  information moves *between* proteins, so its attention weights rank which proteins
  the model treats as related, recovering known interactions (CORUM, STRING,
  Reactome, BioPlex).
- **Protein identity embeddings** (`visualize_proteins`) — the learned-identity
  checkpoints assign each protein a trainable vector shaped only by expression
  context; frozen, these embeddings predict mean gene essentiality beyond what
  protein sequence alone provides.

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

## Tutorials

Start in `tutorials/`; each notebook is executed, so the expected outputs are
visible without running anything.

- **`attention_network_from_psms.ipynb`** — from a mass-spectrometry search result
  to a protein–protein attention network: download search-engine output from PRIDE,
  read the PSMs with [`psm_utils`](https://psm-utils.readthedocs.io), build abundance
  profiles, extract zero-shot attention, and validate the top pairs against nine
  interaction/pathway databases. Works with any UniProt protein set thanks to the
  ESM-C checkpoint.
- **`SST_embedding.ipynb`** — whole-sample (SST) embeddings on the full proteomics
  corpus: why proteomes deeper than the 1024-protein context window need multiple
  passes, how averaging re-sampled passes stabilises the embedding, and the
  tissue-coloured corpus UMAP. Caches the computed SSTs so UMAP settings can be
  iterated in seconds.
- **`protein_embeddings.ipynb`** — the learned protein identity embeddings as an
  interactive UMAP (hover for protein and gene, coloured by corpus detection
  frequency), and how to reuse them for downstream prediction such as gene
  essentiality (manuscript Fig. 5D–F).
Both run on the GPU in minutes to a couple of hours (the SST corpus pass
is the long one and is cached afterwards); reduce `n_epochs` / `N_PASSES` on
CPU-only machines.

## Repository layout

```
omicsfm/          the package
  api.py            compute_sst, predict, visualize_sst, attention_map
  attention.py      attention extraction, PPI ground truth, enrichment curves
  data.py           ExpressionDataset: .h5ad -> binned tensors
  architecture.py   the transformer
  train.py          training loop
  esmc_utils.py     ESM-C sequence embeddings
  hub.py            artifact resolution and download
tutorials/        executed tutorial notebooks (see Tutorials above)
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

### Reproducing the manuscript

All results of the OmicsFM manuscript can be reproduced from four public resources.
Use the first public release of this repository, **v1.0.0** — the version the
manuscript's experiments were run with:

```bash
git clone --branch v1.0.0 https://github.com/CompOmics/OmicsFM.git
```

- **Source code** — this repository:
  [github.com/CompOmics/OmicsFM](https://github.com/CompOmics/OmicsFM).
- **Pretrained checkpoints** (all three modalities) — Hugging Face:
  [rednaSander/omicsfm](https://huggingface.co/rednaSander/omicsfm).
- **Training corpora** — bulk and single-cell transcriptomics:
  [rednaSander/omicsfm-data](https://huggingface.co/datasets/rednaSander/omicsfm-data);
  the reprocessed proteomics corpus will be made available in the same repository
  upon publication.
- **Zenodo deposits**:
  - tissue-specific attention networks (30 human tissues, proteomics and
    transcriptomics): [10.5281/zenodo.22069867](https://doi.org/10.5281/zenodo.22069867)
  - dataset-construction pipelines (run-level metadata-annotation pipeline with its
    resulting annotations, and the transcriptomics corpus builders):
    [10.5281/zenodo.22071865](https://doi.org/10.5281/zenodo.22071865)
  - code, inputs and outputs of all manuscript experiments — each a self-contained
    bundle with its own README, environment and run scripts:
    [10.5281/zenodo.22072026](https://doi.org/10.5281/zenodo.22072026)

All public datasets used are referenced in Supplementary Data 1 under their
original accessions.

## Citation

Manuscript in preparation. Licensed MIT.
