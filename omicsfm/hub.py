"""Resolve and fetch OmicsFM checkpoints and datasets from the HuggingFace Hub.

Nothing here has to be run before using OmicsFM. Every accessor is lazy:

    from omicsfm.hub import get_checkpoint
    ckpt = get_checkpoint("sc_transcriptomics")   # 39 MB on first call, free after

Artifacts are looked up locally first, so a machine that already has the files —
or an air-gapped cluster with them staged — never downloads anything:

1. ``$OMICSFM_HOME`` if set, otherwise the repository this package lives in
2. the local HuggingFace cache
3. a download from the Hub

``omicsfm download`` is a prefetch convenience for offline and HPC use, not a
prerequisite.

The transcriptomics datasets are public. The checkpoints and the proteomics
data are private until publication, so those need authentication:
``huggingface-cli login``, or set ``HF_TOKEN``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Three repositories, split by what may be published. Override any of them if
# the artifacts move, e.g. to an organisation account before submission.
MODEL_REPO = os.environ.get("OMICSFM_MODEL_REPO", "rednaSander/omicsfm")
DATA_REPO = os.environ.get("OMICSFM_DATA_REPO", "rednaSander/omicsfm-data")
PROTEOMICS_REPO = os.environ.get(
    "OMICSFM_PROTEOMICS_REPO", "rednaSander/omicsfm-data-proteomics")

REPOS = {
    "model": (MODEL_REPO, "model"),
    "data": (DATA_REPO, "dataset"),
    "proteomics": (PROTEOMICS_REPO, "dataset"),
}

# Pin the Hub revision so published numbers stay tied to exact weights even if
# newer artifacts are uploaded later. Override for development.
REVISION = os.environ.get("OMICSFM_REVISION", "main")


@dataclass(frozen=True)
class Artifact:
    """One downloadable file."""

    key: str
    repo: str            # a key of REPOS
    remote: str          # path within the Hub repository
    local: str           # path within the OmicsFM tree
    megabytes: float
    tier: str            # models | test | valid | train
    description: str


# Published name -> (directory under model/, size in MB).
#
# The Hub names describe what a user chooses between: the input modality, and
# whether ESM-C sequence embeddings are used. The local directories keep their
# original training-run names so existing experiment code is unaffected.
CHECKPOINTS = {
    "proteomics":                ("prot_flashlfq_diann",       39.0),
    "proteomics_esmc":           ("prot_flashlfq_diann_esmc",  111.4),
    "bulk_transcriptomics":      ("bulk_trans",                39.0),
    "bulk_transcriptomics_esmc": ("bulk_trans_esmc",           111.4),
    "sc_transcriptomics":        ("trans_prot_mapped",         39.0),
    "sc_transcriptomics_esmc":   ("trans_prot_mapped_esmc",    111.4),
}

# Hub dataset name -> (repo key, local directory, {split: MB}).
DATASETS = {
    "proteomics_uniprot": (
        "proteomics", "data/flashlfq_diann",
        {"train": 651.7, "valid": 20.1, "test": 34.1}),
    "bulk_transcriptomics_uniprot": (
        "data", "data/bulk_trans/protein_mapped",
        {"train": 20305.0, "valid": 1105.9, "test": 1054.7}),
    "bulk_transcriptomics_hgnc": (
        "data", "data/bulk_trans/all_transcripts",
        {"train": 38676.5, "valid": 2119.7, "test": 2017.3}),
    "sc_transcriptomics_uniprot": (
        "data", "data/sc_trans/protein_mapped",
        {"train": 11386.9, "valid": 1269.8, "test": 2927.2}),
    "sc_transcriptomics_ensembl": (
        "data", "data/sc_trans/all_transcripts",
        {"train": 11284.5, "valid": 1239.0, "test": 3127.7}),
}

ARTIFACTS: dict[str, Artifact] = {}

for _name, (_dir, _mb) in CHECKPOINTS.items():
    ARTIFACTS[_name] = Artifact(
        key=_name, repo="model",
        remote=f"{_name}/best_model.ckpt",
        local=f"model/{_dir}/best_model.ckpt",
        megabytes=_mb, tier="models",
        description=f"OmicsFM checkpoint: {_name}",
    )

# Needed by the ESM-C checkpoints. The FASTA it derives from is tracked in git.
ARTIFACTS["esmc_cache"] = Artifact(
    key="esmc_cache", repo="model",
    remote="esmc_human_cache.pt",
    local="data/esmc_human_cache.pt",
    megabytes=94.7, tier="models",
    description="Precomputed ESM-C sequence embeddings",
)

for _name, (_repo, _dir, _splits) in DATASETS.items():
    for _split, _mb in _splits.items():
        ARTIFACTS[f"{_name}/{_split}"] = Artifact(
            key=f"{_name}/{_split}", repo=_repo,
            remote=f"{_name}/{_split}.h5ad",
            local=f"{_dir}/{_split}.h5ad",
            megabytes=_mb, tier=_split if _split != "test" else "test",
            description=f"{_name} {_split} split",
        )

TIERS = ("models", "test", "valid", "train")

# The PPI ground-truth matrices are deliberately absent: they are small enough
# to be tracked in git under reference/, so they need no download at all.


def omicsfm_home() -> Path:
    """Root of the local model/ and data/ tree."""
    override = os.environ.get("OMICSFM_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def local_path(artifact: Artifact) -> Path:
    return omicsfm_home() / artifact.local


def is_present(artifact: Artifact) -> bool:
    return local_path(artifact).exists()


def fetch(artifact: Artifact, *, revision: str | None = None,
          token: str | None = None) -> Path:
    """Return a local path to the artifact, downloading it only if missing."""
    target = local_path(artifact)
    if target.exists():
        return target

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("huggingface_hub is required: pip install huggingface_hub") from exc

    repo_id, repo_type = REPOS[artifact.repo]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id, repo_type=repo_type, filename=artifact.remote,
            revision=revision or REVISION,
            token=token or os.environ.get("HF_TOKEN"),
        )
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise RuntimeError(
            f"Cannot access {repo_id!r} on the HuggingFace Hub.\n"
            "The checkpoints and the proteomics data are private until\n"
            "publication. Authenticate with an account that has access:\n"
            "  huggingface-cli login      (or set HF_TOKEN)\n"
            f"Alternatively point OMICSFM_HOME at a tree containing {artifact.local}."
        ) from exc

    import shutil
    shutil.copyfile(downloaded, target)
    return target


def get_checkpoint(name: str, **kwargs) -> Path:
    """Local path to an OmicsFM checkpoint, downloading it on first use."""
    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; expected one of {sorted(CHECKPOINTS)}")
    return fetch(ARTIFACTS[name], **kwargs)


def get_dataset(name: str, split: str = "test", **kwargs) -> Path:
    """Local path to a dataset split, downloading it on first use."""
    key = f"{name}/{split}"
    if key not in ARTIFACTS:
        raise KeyError(f"unknown dataset {key!r}; expected one of {sorted(DATASETS)}"
                       f" with split in train/valid/test")
    return fetch(ARTIFACTS[key], **kwargs)


def get_artifact(key: str, **kwargs) -> Path:
    """Local path to any declared artifact, downloading it on first use."""
    if key not in ARTIFACTS:
        raise KeyError(f"unknown artifact {key!r}")
    return fetch(ARTIFACTS[key], **kwargs)


def plan(tiers=("models",)) -> list[Artifact]:
    """Artifacts in the given tiers that are not present locally."""
    return [a for a in ARTIFACTS.values() if a.tier in tiers and not is_present(a)]


def summarise(tiers=TIERS) -> str:
    """Table of what each tier costs and what is already here."""
    lines = [f"  {'tier':8} {'files':>6} {'present':>8} {'to fetch':>11}",
             "  " + "-" * 36]
    for tier in tiers:
        items = [a for a in ARTIFACTS.values() if a.tier == tier]
        missing = [a for a in items if not is_present(a)]
        mb = sum(a.megabytes for a in missing)
        size = f"{mb/1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
        lines.append(f"  {tier:8} {len(items):6} {len(items)-len(missing):8} {size:>11}")
    return "\n".join(lines)
