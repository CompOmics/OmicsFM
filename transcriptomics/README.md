# Transcriptomics

Two self-contained pipelines that build expression datasets for pretraining ProtGPT.
Both emit the **same `expression.h5ad` contract** (sparse CSR X + obs + var), so the
downstream binning/splitting and the proteomics protein space line up across modalities.

| Arm | Source | Scale | Status |
|-----|--------|-------|--------|
| [`single_cell/`](single_cell/) | [CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) | millions of cells, 27 tissues | working |
| [`bulk/`](bulk/) | [ARCHS4](https://archs4.org/) human bulk RNA-seq | ~1M+ samples, many tissues | skeleton |

Each arm has its own `pipeline_config.yaml`, `build_*_dataset.py`, and `README.md`.
There is intentionally **no shared `common/`** yet — the two pipelines differ in source
I/O and grouping (`dataset_id` vs `series_id`); shared code will be factored out once the
bulk arm is implemented and the genuine overlap is clear.

**Shared metadata vocabulary:** the bulk arm harmonizes ARCHS4's free-text metadata to
the same controlled vocabulary used by the proteomics pipeline
([`../agentic-metadata`](../agentic-metadata)), so tissue/disease labels are aligned
across the proteomics, single-cell, and bulk foundation models.

See each arm's README for usage.
