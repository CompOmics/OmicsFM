"""Upload the OmicsFM checkpoints to a private HuggingFace repository.

    python tools/upload_models.py                 # dry run: show the plan
    python tools/upload_models.py --create        # create the repo (private)
    python tools/upload_models.py --upload        # upload the checkpoints

Names on the Hub come from omicsfm.hub.CHECKPOINTS, so the module that
downloads them and the script that uploads them cannot disagree.

Requires authentication: huggingface-cli login, or set HF_TOKEN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omicsfm.hub import ARTIFACTS, CHECKPOINTS, MODEL_REPO, local_path  # noqa: E402

CARD = """---
license: cc-by-4.0
tags:
  - biology
  - proteomics
  - transcriptomics
  - single-cell
library_name: omicsfm
---

# OmicsFM

A foundation model over paired proteomic and transcriptomic expression.
Attention between measured features yields a protein-protein association
network without task-specific training.

## Checkpoints

| name | modality | feature ID embedding | parameters |
|---|---|---|---|
| `proteomics` | proteomics | learned | 4.94 M |
| `proteomics_esmc` | proteomics | ESM-C | 5.75 M |
| `bulk_transcriptomics` | bulk transcriptomics | learned | 4.94 M |
| `bulk_transcriptomics_esmc` | bulk transcriptomics | ESM-C | 5.75 M |
| `sc_transcriptomics` | single-cell transcriptomics | learned | 4.94 M |
| `sc_transcriptomics_esmc` | single-cell transcriptomics | ESM-C | 5.75 M |

All are mapped onto a shared UniProt accession vocabulary.

**Feature ID embedding.** The `learned` variants train an embedding table over
the vocabulary from scratch. The `ESM-C` variants instead look each feature up
in a frozen table of precomputed ESM-C sequence embeddings and pass it through
a small trained projection, so identity is grounded in protein sequence rather
than learned from co-expression alone.

**Parameter counts** are the trained model excluding the feature ID embedding,
which scales with vocabulary rather than model capacity and would otherwise
dominate. The transformer is identical across all six at 4.94 M parameters; the
ESM-C variants add 0.81 M for the projection. Excluded on top of that: the
5.19 M learned embedding table (`learned` variants) and the 23.35 M frozen
ESM-C lookup (`ESM-C` variants), which is not trained at all.

## Use

```python
from omicsfm.hub import get_checkpoint
from omicsfm.api import attention_map

ckpt = get_checkpoint("sc_transcriptomics")   # downloaded and cached on first use
network = attention_map(str(ckpt), data_path=...)
```

The ESM-C variants additionally need the canonical human proteome FASTA and
the precomputed embedding cache; see the repository README.

## Citation

Publication in preparation.
"""


def plan() -> list[tuple[str, Path, float]]:
    rows = []
    for name in CHECKPOINTS:
        artifact = ARTIFACTS[name]
        rows.append((name, local_path(artifact), artifact.megabytes))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=MODEL_REPO)
    parser.add_argument("--create", action="store_true", help="create the private repo")
    parser.add_argument("--upload", action="store_true", help="upload the checkpoints")
    parser.add_argument("--public", action="store_true",
                        help="create public instead of private (not the default)")
    args = parser.parse_args()

    rows = plan()
    missing = [(n, p) for n, p, _ in rows if not p.exists()]
    print(f"repository: {args.repo}   ({'public' if args.public else 'PRIVATE'})\n")
    print(f"  {'hub name':28} {'size':>8}  local file")
    for name, path, mb in rows:
        mark = "  " if path.exists() else "!!"
        print(f"{mark}{name:28} {mb:7.1f}M  {path.relative_to(REPO_ROOT)}")
    total = sum(mb for _, _, mb in rows)
    print(f"\n  {len(rows)} checkpoints, {total/1024:.2f} GB")

    if missing:
        sys.exit(f"\n{len(missing)} checkpoint(s) not found locally; aborting.")

    if not (args.create or args.upload):
        print("\n  dry run; pass --create and/or --upload to act")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("not authenticated: run 'huggingface-cli login' or set HF_TOKEN")
    print(f"\n  authenticated as {who}")

    if args.create:
        api.create_repo(repo_id=args.repo, repo_type="model",
                        private=not args.public, exist_ok=True)
        print(f"  repo ready: https://huggingface.co/{args.repo}")
        api.upload_file(path_or_fileobj=CARD.encode("utf-8"),
                        path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
        print("  model card uploaded")

    if args.upload:
        for name, path, mb in rows:
            target = f"{name}/best_model.ckpt"
            print(f"  uploading {name} ({mb:.0f} MB) -> {target}", flush=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=target,
                            repo_id=args.repo, repo_type="model")
        print(f"\n  done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
