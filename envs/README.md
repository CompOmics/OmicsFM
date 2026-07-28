# Conda environments

Five environments cover the experiments. They are separate because the model
stacks pin incompatible torch and Python versions.

| Environment | Python | torch | Used by |
|---|---|---|---|
| `prot_gpt` | 3.12 | 2.10.0+cu128 | OmicsFM training and inference, all figures, table assembly |
| `scprint` | 3.11 | 2.7.0+cu128 | scPRINT, BenGRN scoring, scDataLoader |
| `scGPT` | 3.10 | 2.11.0+cu128 | scGPT comparator |
| `deepsem` | 3.10 | 2.11.0+cu128 | DeepSEM baseline |
| `genie3` | 3.10 | — | GENIE3 baseline |

## Create

```bash
conda env create -f envs/prot_gpt.yml
conda env create -f envs/scprint.yml
conda env create -f envs/scGPT.yml
conda env create -f envs/deepsem.yml
conda env create -f envs/genie3.yml
```

Conda supplies only `python` and `pip`; every other package is pinned through
pip. That keeps the files solvable on Linux even though they were captured on
Windows, where conda build strings would not transfer.

## CUDA

Four of the five pin a `+cu128` torch build and carry

```
--extra-index-url https://download.pytorch.org/whl/cu128
```

so pip can find those wheels. For a different CUDA version, change both the
index URL and the `torch` / `torchvision` / `torchaudio` pins to match. On
Linux, pip pulls the `nvidia-*` runtime wheels automatically as torch
dependencies; they are deliberately not pinned here because the Windows and
Linux sets differ.

The environments were captured on an RTX 5090 (compute capability 12.0), which
requires CUDA 12.8 or newer — that is why the pins are `cu128` rather than
something older.

## If an environment fails to solve

The files pin the full transitive closure, which is faithful but strict. When a
single package has no Linux wheel at the pinned version, relax that one line to
an unpinned name rather than regenerating the whole file:

```yaml
      - some-package          # was some-package==1.2.3
```

`scprint.yml` (276 packages) and `prot_gpt.yml` (189) are the most likely to
need this; the three baseline environments are small.

## Regenerating

These were generated from working environments rather than written by hand. To
refresh them after changing an environment, export with conda providing only
python and pip, and pin the rest from `pip list --format=freeze`, dropping
Windows-only distributions such as `pywin32`.
