"""
ProtGPT API — inference, evaluation, and visualization.

Functions:
    predict             : Run the model on a dataset; returns embeddings, predictions, and metadata.
    visualize_proteins  : Reduce & plot the learned protein embeddings (from the embedding layer).
    visualize_proteomes : Reduce & plot the learned sample-summary (SST) embeddings from predict() output.
"""

import logging
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from protgpt.architecture import ProtGPT
from protgpt.data import ExpressionCollator, ExpressionDataset

log = logging.getLogger(__name__)

# helpers
def _load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load a raw checkpoint dict, stripping torch.compile's '_orig_mod.' prefix."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if any(k.startswith("_orig_mod.") for k in ckpt.get("model", {})):
        ckpt["model"] = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    return ckpt


def _build_model_cfg(cfg: dict) -> dict:
    """Extract model config dict from a checkpoint config."""
    mcfg = cfg["model"]
    model_cfg = {
        "num_features": None,  # set by caller
        "num_bins":     cfg["data"]["num_bins"],
        "d_model":      mcfg["d_model"],
        "n_heads":      mcfg["n_heads"],
        "n_layers":     mcfg["n_layers"],
        "d_ff":         mcfg["d_ff"],
        "dropout":      mcfg["dropout"],
        "use_esmc":     mcfg.get("use_esmc", False),
    }
    pg_cfg = mcfg.get("protein_groups", {})
    if pg_cfg.get("enabled", False):
        model_cfg["protein_groups"] = pg_cfg
    return model_cfg


def _load_model(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> tuple:
    """Load a ProtGPT model from a training checkpoint (same-dataset inference).

    For inference on a different dataset, use build_model_for_dataset() instead.

    Args:
        checkpoint_path: path to the .ckpt file saved during training.
        device: torch device string.

    Returns:
        (model, full_config_dict)
    """
    ckpt = _load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    model_cfg = _build_model_cfg(cfg)

    esmc_lookup = None
    if model_cfg["use_esmc"]:
        esmc_lookup = ckpt["model"]["esmc_embedding.weight"]
        model_cfg["num_features"] = esmc_lookup.shape[0] - 1
    else:
        model_cfg["num_features"] = ckpt["model"]["feature_emb.weight"].shape[0] - 1

    model = ProtGPT(model_cfg, esmc_lookup=esmc_lookup)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    log.info(f"Loaded model from {checkpoint_path} "
             f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")
    return model, cfg


def build_model_for_dataset(
    checkpoint_path: str,
    dataset: ExpressionDataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fasta_path: str | None = None,
    esmc_cache: str | None = None,
) -> tuple:
    """Load a ProtGPT model and build ESM-C embeddings for a (possibly new) dataset.

    This computes ESM-C embeddings from FASTA for the dataset's proteins,
    builds the lookup tensor in memory, and loads the trained weights.
    The esmc_proj MLP transfers to any protein set since it projects from
    the same ESM-C embedding space.

    Args:
        checkpoint_path: path to the .ckpt file.
        dataset: a loaded ExpressionDataset (provides protein names).
        device: torch device string.
        fasta_path: path to FASTA file. Defaults to the one stored in checkpoint config.
        esmc_cache: path to precomputed ESM-C embeddings (.pt). Defaults to checkpoint config.

    Returns:
        (model, full_config_dict)
    """
    ckpt = _load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    model_cfg = _build_model_cfg(cfg)
    use_esmc = model_cfg["use_esmc"]

    feature_names = list(dataset.feature_names)
    model_cfg["num_features"] = len(feature_names)

    esmc_lookup = None
    if use_esmc:
        from protgpt.esmc_utils import build_esmc_lookup
        fasta = fasta_path or cfg["model"]["fasta_path"]
        cache = esmc_cache or cfg["model"].get("esmc_cache")
        esmc_lookup = build_esmc_lookup(
            feature_names, fasta, cfg["model"].get("esmc_model", "esmc_600m"),
            cache_path=cache,
        )

    model = ProtGPT(model_cfg, esmc_lookup=esmc_lookup)

    # Load trained weights; skip esmc_embedding.weight if shape differs
    state = ckpt["model"]
    if use_esmc and state["esmc_embedding.weight"].shape != esmc_lookup.shape:
        state = {k: v for k, v in state.items() if k != "esmc_embedding.weight"}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    log.info(f"Loaded model from {checkpoint_path} "
             f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")
    return model, cfg


def _reduce(embeddings: np.ndarray, method: str = "umap", n_components: int = 2, **kwargs) -> np.ndarray:
    """Dimensionality reduction wrapper.
    
    Args:
        embeddings: (N, D) array.
        method: one of 'umap', 'tsne', 'pca'.
        n_components: target dimensions (default 2).

    Returns:
        (N, n_components) array.
    """
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, **kwargs)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        reducer = UMAP(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'pca', 'tsne', 'umap'.")
    return reducer.fit_transform(embeddings)



# 1.  predict
def predict(
    checkpoint_path: str,
    data_path: str,
    mode: str = "eval",
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    fasta_path: str | None = None,
    esmc_cache: str | None = None,
) -> dict:
    """
    Run the trained model on a dataset and collect all outputs.

    Args:
        checkpoint_path: path to the .ckpt file.
        data_path: path to a parquet file with the same schema as training data.
        mode: 'eval'      — use context/target split (context_ratio from config) so
                             predictions and losses can be evaluated.
              'inference'  — all detected proteins become context (context_ratio=1.0).
                             No targets, but SST embeddings capture the full proteome.
        device: torch device string (e.g. 'cuda', 'cuda:0', 'cpu').
        batch_size: batch size for the dataloader.
        num_workers: number of dataloader workers.
        fasta_path: path to FASTA file for ESM-C embeddings. Defaults to checkpoint config.
        esmc_cache: path to precomputed ESM-C embeddings (.pt). Defaults to checkpoint config.

    Returns:
        dict with keys:
            sst_emb            np.ndarray  (N, d_model)    — SST embedding per sample
            pred_ctx           list[np.ndarray]            — ctx-head predictions per sample (eval mode only)
            pred_sst           list[np.ndarray]            — sst-head predictions per sample (eval mode only)
            true_bins          list[np.ndarray]            — ground-truth bins per sample (eval mode only)
            target_feature_ids list[np.ndarray]            — target protein IDs per sample (eval mode only)
            sample_ids         list[str]                   — sample identifiers (same order as rows)
            config             dict                        — full config dict from checkpoint
    """
    assert mode in ("eval", "inference"), f"mode must be 'eval' or 'inference', got '{mode}'"

    # load dataset first so we know which proteins are needed
    ckpt = _load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    detect_groups = dcfg.get("detect_groups", dcfg.get("has_protein_groups", False))
    pg_cfg = mcfg.get("protein_groups", {})
    pg_enabled = pg_cfg.get("enabled", False)
    abs_cfg = mcfg.get("absent_species", {})
    abs_enabled = abs_cfg.get("enabled", False)

    log.info(f"Loading dataset from {data_path} ...")
    max_group_size = pg_cfg.get("max_group_size", 1) if pg_enabled else 1
    ds = ExpressionDataset(data_path, num_bins=dcfg["num_bins"],
                           detect_groups=detect_groups, max_group_size=max_group_size)
    sample_ids = list(ds.sample_ids)

    # build model with ESM-C embeddings for this dataset's proteins
    model, cfg = build_model_for_dataset(checkpoint_path, ds, device, fasta_path, esmc_cache)

    # collator: in inference mode use all detected proteins as context
    feature_context_ratio = mcfg["feature_context_ratio"] if mode == "eval" else 1.0
    group_context_ratio = pg_cfg.get("group_context_ratio", 1.0) if mode == "eval" else 1.0
    absent_context_ratio = abs_cfg.get("absent_context_ratio", 1.0) if mode == "eval" else 1.0
    collator = ExpressionCollator(
        num_features=ds.num_features(),
        feature_context_ratio=feature_context_ratio,
        group_context_ratio=group_context_ratio,
        input_features=mcfg["input_features"],
        protein_groups_enabled=pg_enabled,
        input_groups=pg_cfg.get("input_groups", 0) if pg_enabled else 0,
        absent_species_enabled=abs_enabled,
        input_absent=abs_cfg.get("input_absent", 0),
        absent_context_ratio=absent_context_ratio,
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collator,
    )

    # run forward in eval mode
    all_sst = []
    all_pred_ctx = []
    all_pred_sst = []
    all_true_bins = []
    all_target_ids = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)

            all_sst.append(out["sst_emb"].cpu().numpy())

            if mode == "eval":
                for pc, pp, tb, tid in zip(
                    out["pred_ctx"], out["pred_sst"],
                    out["true_bins"], out["target_feature_ids"],
                ):
                    all_pred_ctx.append(pc.cpu().numpy())
                    all_pred_sst.append(pp.cpu().numpy())
                    all_true_bins.append(tb.cpu().numpy())
                    all_target_ids.append(tid.cpu().numpy())

    result = {
        "sst_emb":            np.concatenate(all_sst, axis=0),  # (N, d_model)
        "sample_ids":         sample_ids,

    }

    if mode == "eval":
        result["pred_ctx"]           = all_pred_ctx
        result["pred_sst"]           = all_pred_sst
        result["true_bins"]          = all_true_bins
        result["target_feature_ids"] = all_target_ids

        # compute aggregate metrics for convenience
        all_pc = np.concatenate(all_pred_ctx)
        all_pp = np.concatenate(all_pred_sst)
        all_tb = np.concatenate(all_true_bins)
        result["metrics"] = {
            "ctx_mse": float(np.mean((all_pc - all_tb) ** 2)),
            "sst_mse": float(np.mean((all_pp - all_tb) ** 2)),
            "ctx_mae": float(np.mean(np.abs(all_pc - all_tb))),
            "sst_mae": float(np.mean(np.abs(all_pp - all_tb))),
            "n_predictions": len(all_tb),
        }
        log.info(f"Eval metrics: ctx_mse={result['metrics']['ctx_mse']:.4f}, "
                 f"sst_mse={result['metrics']['sst_mse']:.4f}")

    log.info(f"Done. SST embeddings shape: {result['sst_emb'].shape}")
    return result



# 2.  visualize_proteins
def visualize_proteins(
    model: ProtGPT | str,
    dataset: ExpressionDataset,
    metadata: pd.DataFrame | None = None,
    color_by: str | None = None,
    method: str = "umap",
    figsize: tuple = (10, 8),
    save_path: str | None = None,
    **reducer_kwargs,
) -> pd.DataFrame:
    """
    Visualize the learned protein embeddings from the model's embedding layer.

    Uses the ExpressionDataset to know which proteins exist and retrieve their
    learned embeddings from the model.  Metadata is joined via a 'protein_id'
    column (or index) that should match the protein names in the dataset.

    Args:
        model: a loaded ProtGPT model, or path to a .ckpt file.
        dataset: a loaded ExpressionDataset (provides protein names and indices).
        metadata: optional DataFrame with a 'protein_id' column (or index named
                  'protein_id') matching the protein names in the dataset, plus
                  the column specified by color_by.  If None, plots without coloring.
        color_by: column name in metadata to colour the scatter plot by.
                  Required when metadata is provided.
        method: dimensionality reduction method ('umap', 'tsne', 'pca').
        figsize: matplotlib figure size.
        save_path: if provided, save the figure to this path.
        **reducer_kwargs: forwarded to the reducer (e.g. n_neighbors for UMAP).

    Returns:
        DataFrame with columns: protein_id, dim_1, dim_2, and optionally <color_by>.
    """
    if isinstance(model, str):
        model, _ = _load_model(model)

    # protein names from the dataset (var_names / accessions, index order)
    feature_names = list(dataset.feature_names)  # length = num_features

    # extract embeddings: indices 1..num_features in the embedding layer
    # (index 0 is the padding token, real proteins are 1-indexed)
    device = next(model.parameters()).device
    indices = torch.arange(1, len(feature_names) + 1, device=device)
    emb_weights = model.feature_emb(indices).detach().cpu().numpy()  # (num_features, d_model)

    # reduce
    coords = _reduce(emb_weights, method=method, **reducer_kwargs)

    # build result dataframe
    result = pd.DataFrame({
        "protein_id": feature_names,
        "dim_1": coords[:, 0],
        "dim_2": coords[:, 1],
    })

    # merge metadata if provided
    if metadata is not None and color_by is not None:
        # support metadata with protein_id as either a column or the index
        if "protein_id" not in metadata.columns:
            metadata = metadata.rename_axis("protein_id").reset_index()
        result = result.merge(metadata[["protein_id", color_by]], on="protein_id", how="left")

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    if color_by is not None and color_by in result.columns:
        for cat in result[color_by].dropna().unique():
            subset = result[result[color_by] == cat]
            ax.scatter(subset["dim_1"], subset["dim_2"], label=str(cat), s=10, alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", markerscale=2)
    else:
        ax.scatter(result["dim_1"], result["dim_2"], s=10, alpha=0.7)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    title_suffix = f" colored by {color_by}" if color_by and color_by in result.columns else ""
    ax.set_title(f"Protein embeddings ({method.upper()}){title_suffix}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Figure saved to {save_path}")
    plt.show()

    return result

# 3.  visualize_proteomes
def visualize_proteomes(
    predictions: dict,
    metadata: pd.DataFrame | None = None,
    color_by: str | None = None,
    method: str = "umap",
    figsize: tuple = (10, 8),
    save_path: str | None = None,
    **reducer_kwargs,
) -> pd.DataFrame:
    """
    Visualize the learned sample-summary (SST) embeddings produced by predict().

    Args:
        predictions: output dict from predict() — must contain 'sst_emb' and 'sample_ids'.
        metadata: optional DataFrame with a 'sample_id' column (or index) matching
                  the sample IDs from the dataset.  If None, plots without coloring.
        color_by: column name in metadata to colour the scatter plot by.
                  Required when metadata is provided.
        method: dimensionality reduction method ('umap', 'tsne', 'pca').
        figsize: matplotlib figure size.
        save_path: if provided, save the figure to this path.
        **reducer_kwargs: forwarded to the reducer.

    Returns:
        DataFrame with columns: sample_id, dim_1, dim_2, and optionally <color_by>.
    """
    sst_emb = predictions["sst_emb"]         # (N, d_model)
    sample_ids = predictions["sample_ids"]    # list of str

    # reduce
    coords = _reduce(sst_emb, method=method, **reducer_kwargs)

    # build result dataframe
    result = pd.DataFrame({
        "sample_id": sample_ids,
        "dim_1": coords[:, 0],
        "dim_2": coords[:, 1],
    })

    # merge metadata if provided
    if metadata is not None and color_by is not None:
        if "sample_id" not in metadata.columns:
            metadata = metadata.rename_axis("sample_id").reset_index()
        result = result.merge(metadata[["sample_id", color_by]], on="sample_id", how="left")

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    if color_by is not None and color_by in result.columns:
        for cat in result[color_by].dropna().unique():
            subset = result[result[color_by] == cat]
            ax.scatter(subset["dim_1"], subset["dim_2"], label=str(cat), s=10, alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", markerscale=2)
    else:
        ax.scatter(result["dim_1"], result["dim_2"], s=10, alpha=0.7)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    title_suffix = f" colored by {color_by}" if color_by and color_by in result.columns else ""
    ax.set_title(f"Sample-summary (SST) embeddings ({method.upper()}){title_suffix}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Figure saved to {save_path}")
    plt.show()

    return result
