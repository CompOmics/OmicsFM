"""
ESM-C embedding utilities.

Compute ESM-C protein embeddings from FASTA sequences and build
lookup tensors for use in ProtGPT.
"""

import logging
import os

import torch
from Bio import SeqIO
from tqdm import tqdm

log = logging.getLogger(__name__)

MAX_SEQ_LEN = 2048


def load_fasta_sequences(fasta_path: str) -> dict[str, str]:
    """Parse a UniProt FASTA file and return {accession: sequence}."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        acc = record.id.split("|")[1]
        sequences[acc] = str(record.seq)
    return sequences


def compute_esmc_embeddings(
    proteins: list[str],
    sequences: dict[str, str],
    model_name: str = "esmc_600m",
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Run ESM-C inference and return {accession: mean-pooled embedding}."""
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    device = torch.device(device)
    client = ESMC.from_pretrained(model_name, device=device)
    log.info(f"Loaded {model_name} on {device}")

    embeddings = {}
    failed = []

    for acc in tqdm(proteins, desc="Computing ESM-C embeddings"):
        seq = sequences[acc][:MAX_SEQ_LEN]
        try:
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
            emb = output.embeddings.detach().cpu()
            if emb.ndim == 3:
                emb = emb[0].mean(dim=0)
            elif emb.ndim == 2:
                emb = emb.mean(dim=0)
            embeddings[acc] = emb
        except Exception as e:
            failed.append(acc)
            if len(failed) <= 5:
                log.warning(f"Failed: {acc} ({len(seq)} aa): {e}")

    log.info(f"Embedded {len(embeddings)}/{len(proteins)} proteins ({len(failed)} failed)")
    return embeddings


def _build_lookup_tensor(
    protein_names: list[str], embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build (N+1, emb_dim) lookup tensor from per-protein embeddings."""
    missing = [acc for acc in protein_names if acc not in embeddings]
    if missing:
        preview = missing[:10]
        more = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)}/{len(protein_names)} proteins have no ESM-C embedding. "
            f"First 10: {preview}{more}"
        )
    emb_dim = next(iter(embeddings.values())).shape[0]
    lookup = torch.zeros(len(protein_names) + 1, emb_dim)
    for i, acc in enumerate(protein_names):
        lookup[i + 1] = embeddings[acc]
    return lookup


def build_esmc_lookup(
    protein_names: list[str],
    fasta_path: str,
    model_name: str = "esmc_600m",
    cache_path: str | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build an ESM-C lookup tensor for the given proteins.

    If cache_path is provided and exists, loads per-protein embeddings from it
    (avoiding ESM-C inference). Otherwise computes from FASTA and saves to
    cache_path for future reuse.

    Args:
        protein_names: Sorted list of protein accessions (determines index order).
        fasta_path: Path to UniProt FASTA file with sequences.
        model_name: ESM-C model variant.
        cache_path: Optional path to a .pt cache file with per-protein embeddings.

    Returns:
        Tensor of shape (len(protein_names) + 1, emb_dim) with padding at index 0.

    Raises:
        ValueError: If any protein in protein_names is not found in the FASTA.
    """
    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        log.info(f"Loading ESM-C embeddings from cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        cached_embs = data["embeddings"]  # {acc: tensor}

        missing_from_cache = [p for p in protein_names if p not in cached_embs]
        if missing_from_cache:
            preview = missing_from_cache[:10]
            more = (f" ... (+{len(missing_from_cache) - 10} more)"
                    if len(missing_from_cache) > 10 else "")
            raise ValueError(
                f"{len(missing_from_cache)}/{len(protein_names)} proteins missing from "
                f"ESM-C cache ({cache_path}). First 10: {preview}{more}. "
                f"Delete the cache to recompute, or pre-build it with "
                f"`python -m protgpt.esmc_utils --fasta {fasta_path} --save {cache_path}`."
            )
        lookup = _build_lookup_tensor(protein_names, cached_embs)
        log.info(f"Built ESM-C lookup from cache: {lookup.shape}")
        return lookup

    # Compute from FASTA
    log.info(f"Loading sequences from {fasta_path}...")
    sequences = load_fasta_sequences(fasta_path)
    log.info(f"  {len(sequences)} sequences loaded")

    missing = [p for p in protein_names if p not in sequences]
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"{len(missing)}/{len(protein_names)} proteins not found in FASTA ({fasta_path}). "
            f"First 10: {preview}"
        )

    embeddings = compute_esmc_embeddings(protein_names, sequences, model_name, device=device)

    # Save to cache for future reuse
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        emb_dim = next(iter(embeddings.values())).shape[0]
        torch.save({"embeddings": embeddings, "emb_dim": emb_dim}, cache_path)
        log.info(f"Saved ESM-C cache to {cache_path}")

    lookup = _build_lookup_tensor(protein_names, embeddings)
    log.info(f"Built ESM-C lookup: {lookup.shape}")
    return lookup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compute ESM-C embeddings for all proteins in a FASTA file.",
    )
    parser.add_argument("--fasta", required=True, help="Path to UniProt FASTA file")
    parser.add_argument("--save", required=True, help="Output path for the .pt cache file")
    parser.add_argument("--model", default="esmc_600m", help="ESM-C model name (default: esmc_600m)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for ESM-C inference (default: cuda if available, else cpu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    log.info(f"Loading sequences from {args.fasta}...")
    sequences = load_fasta_sequences(args.fasta)
    log.info(f"  {len(sequences)} sequences loaded")

    proteins = sorted(sequences.keys())
    embeddings = compute_esmc_embeddings(proteins, sequences, args.model, device=args.device)

    emb_dim = next(iter(embeddings.values())).shape[0]
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"embeddings": embeddings, "emb_dim": emb_dim}, args.save)
    log.info(f"Saved {len(embeddings)} embeddings (dim={emb_dim}) to {args.save}")
