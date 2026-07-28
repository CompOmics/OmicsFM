"""
ProtGPT API — inference, evaluation, and visualization.

Functions:
    compute_sst          : Sample Summary Token (SST) embedding per sample — whole-sample representation.
    predict              : Predict held-out expression bins (masked-modeling eval); predictions + metrics.
    visualize_proteins   : Reduce & plot the learned protein embeddings (learned-embedding models only).
    visualize_sst        : Reduce & plot the SST embeddings from compute_sst() output.
    eval_attention_heads : Score every (layer, head) by PPI enrichment to find the best head.
    attention_map        : Accumulate the protein×protein attention map (mean or one head).
    enrichment_curves    : Enrichment curves of an attention map vs PPI ground-truth databases.
    load_ppi_ground_truth: Load a bundled PPI ground-truth matrix (corum/string/kegg/reactome/gocc).
"""

import logging
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from omicsfm.architecture import ProtGPT
from omicsfm.data import ExpressionCollator, ExpressionDataset

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
        from omicsfm.esmc_utils import build_esmc_lookup
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
    """Dimensionality reduction wrapper with sensible, reproducible defaults.

    Defaults (override via kwargs):
        pca  : random_state=0
        umap : n_neighbors=75, min_dist=0.3, random_state=42
        tsne : random_state=42

    Args:
        embeddings: (N, D) array.
        method: one of 'umap', 'tsne', 'pca'.
        n_components: target dimensions (default 2).

    Returns:
        (N, n_components) array.
    """
    method = method.lower()
    if method == "pca":
        params = {"random_state": 0, **kwargs}
        reducer = PCA(n_components=n_components, **params)
    elif method == "tsne":
        params = {"random_state": 42, **kwargs}
        reducer = TSNE(n_components=n_components, **params)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        params = {"n_neighbors": 75, "min_dist": 0.3, "random_state": 42, **kwargs}
        reducer = UMAP(n_components=n_components, **params)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'pca', 'tsne', 'umap'.")
    return reducer.fit_transform(embeddings)


def _scatter_2d(df: pd.DataFrame, x: str, y: str, color_by: str | None,
                hover: list[str] | None, title: str, width: int, height: int,
                save_path: str | None,
                color_order: list[str] | None = None,
                palette: list[str] | None = None):
    """Interactive 2-D scatter via plotly express (shared by the visualize_* fns).

    Categorical `color_by` defaults to a frequency-ordered legend and a large
    qualitative palette (scales past 20 categories); pass `color_order` to fix the
    legend order and `palette` to supply your own colour sequence. Numeric
    `color_by` uses a continuous scale. Returns the plotly Figure.
    """
    import plotly.express as px

    kwargs = dict(x=x, y=y, title=title, width=width, height=height,
                  hover_data=hover or None)
    if color_by is not None:
        kwargs["color"] = color_by
        is_categorical = not pd.api.types.is_numeric_dtype(df[color_by])
        if is_categorical:
            order = color_order or df[color_by].astype(str).value_counts().index.tolist()
            pal = palette or (px.colors.qualitative.Dark24 + px.colors.qualitative.Light24)
            df = df.copy()
            df[color_by] = df[color_by].astype(str)
            kwargs["category_orders"] = {color_by: [str(c) for c in order]}
            kwargs["color_discrete_sequence"] = list(pal)[:len(order)]

    fig = px.scatter(df, **kwargs)
    fig.update_traces(marker=dict(size=4, opacity=0.7, line=dict(width=0)))
    fig.update_layout(legend=dict(itemsizing="constant"),
                      plot_bgcolor="white", paper_bgcolor="white")
    if save_path:
        fig.write_html(save_path) if save_path.endswith(".html") else fig.write_image(save_path)
        log.info(f"Figure saved to {save_path}")
    fig.show()
    return fig



# shared engine
def _run(checkpoint_path: str, data_path: str, *, eval_mode: bool, device: str,
         batch_size: int, num_workers: int, fasta_path: str | None,
         esmc_cache: str | None) -> dict:
    """Load checkpoint + dataset + model and run the forward pass over the data.

    eval_mode=False → every detected feature is context (the SST summarizes the whole
                      sample); no targets are held out.
    eval_mode=True  → context/target split (ratios from the training config), so the
                      held-out target bins are predicted.

    Returns a dict with 'sst_emb' and 'sample_ids' always, plus the per-sample
    predictions and aggregate 'metrics' when eval_mode. compute_sst() and predict()
    shape their public output from this.
    """
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

    # context ratios: full context (1.0) for SST embeddings, config split for evaluation
    collator = ExpressionCollator(
        num_features=ds.num_features(),
        feature_context_ratio=mcfg["feature_context_ratio"] if eval_mode else 1.0,
        group_context_ratio=pg_cfg.get("group_context_ratio", 1.0) if eval_mode else 1.0,
        input_features=mcfg["input_features"],
        protein_groups_enabled=pg_enabled,
        input_groups=pg_cfg.get("input_groups", 0) if pg_enabled else 0,
        absent_species_enabled=abs_enabled,
        input_absent=abs_cfg.get("input_absent", 0),
        absent_context_ratio=abs_cfg.get("absent_context_ratio", 1.0) if eval_mode else 1.0,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, collate_fn=collator)

    all_sst, all_pred_ctx, all_pred_sst, all_true_bins, all_target_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Running model"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            all_sst.append(out["sst_emb"].cpu().numpy())
            if eval_mode:
                for pc, pp, tb, tid in zip(out["pred_ctx"], out["pred_sst"],
                                           out["true_bins"], out["target_feature_ids"]):
                    all_pred_ctx.append(pc.cpu().numpy())
                    all_pred_sst.append(pp.cpu().numpy())
                    all_true_bins.append(tb.cpu().numpy())
                    all_target_ids.append(tid.cpu().numpy())

    res = {"sst_emb": np.concatenate(all_sst, axis=0), "sample_ids": sample_ids}
    if eval_mode:
        all_pc = np.concatenate(all_pred_ctx)
        all_pp = np.concatenate(all_pred_sst)
        all_tb = np.concatenate(all_true_bins)
        res.update(
            pred_ctx=all_pred_ctx, pred_sst=all_pred_sst,
            true_bins=all_true_bins, target_feature_ids=all_target_ids,
            metrics={
                "ctx_mse": float(np.mean((all_pc - all_tb) ** 2)),
                "sst_mse": float(np.mean((all_pp - all_tb) ** 2)),
                "ctx_mae": float(np.mean(np.abs(all_pc - all_tb))),
                "sst_mae": float(np.mean(np.abs(all_pp - all_tb))),
                "n_predictions": len(all_tb),
            },
        )
    return res


# 1.  compute_sst  — whole-sample embeddings
def compute_sst(
    checkpoint_path: str,
    data_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    fasta_path: str | None = None,
    esmc_cache: str | None = None,
) -> dict:
    """Compute the Sample Summary Token (SST) embedding for every sample.

    All detected features are used as context, so each SST embedding summarizes the
    full sample (proteome/transcriptome). Feed the result straight to visualize_sst().

    Args:
        checkpoint_path: path to the .ckpt file.
        data_path: path to a `.h5ad` dataset, or an in-memory AnnData (same protein/gene space as training).
        device: torch device string ('cuda', 'cuda:0', 'cpu').
        batch_size / num_workers: dataloader settings.
        fasta_path / esmc_cache: override the checkpoint config's ESM-C FASTA / cache.

    Returns:
        dict with:
            sst_emb     np.ndarray (N, d_model) — one embedding per sample
            sample_ids  list[str]               — row-aligned sample identifiers
    """
    res = _run(checkpoint_path, data_path, eval_mode=False, device=device,
               batch_size=batch_size, num_workers=num_workers,
               fasta_path=fasta_path, esmc_cache=esmc_cache)
    log.info(f"Done. SST embeddings: {res['sst_emb'].shape}")
    return {"sst_emb": res["sst_emb"], "sample_ids": res["sample_ids"]}


# 2.  predict  — masked-bin prediction / evaluation
def predict(
    checkpoint_path: str,
    data_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    fasta_path: str | None = None,
    esmc_cache: str | None = None,
) -> dict:
    """Predict held-out expression bins (the model's masked-modeling task).

    A fraction of each sample's features is held out as targets (context/target split
    from the training config) and their expression bins are predicted by both heads.
    Use compute_sst() instead if you want whole-sample embeddings.

    Args:
        checkpoint_path: path to the .ckpt file.
        data_path: path to a `.h5ad` dataset, or an in-memory AnnData (same protein/gene space as training).
        device: torch device string ('cuda', 'cuda:0', 'cpu').
        batch_size / num_workers: dataloader settings.
        fasta_path / esmc_cache: override the checkpoint config's ESM-C FASTA / cache.

    Returns:
        dict with:
            sample_ids         list[str]        — sample identifiers
            pred_ctx           list[np.ndarray] — ctx-head predicted bins, per sample
            pred_sst           list[np.ndarray] — sst-head predicted bins, per sample
            true_bins          list[np.ndarray] — ground-truth bins, per sample
            target_feature_ids list[np.ndarray] — target feature ids, per sample
            metrics            dict             — ctx_mse, sst_mse, ctx_mae, sst_mae, n_predictions
    """
    res = _run(checkpoint_path, data_path, eval_mode=True, device=device,
               batch_size=batch_size, num_workers=num_workers,
               fasta_path=fasta_path, esmc_cache=esmc_cache)
    m = res["metrics"]
    log.info(f"Eval metrics: ctx_mse={m['ctx_mse']:.4f}, sst_mse={m['sst_mse']:.4f}")
    return {
        "sample_ids":         res["sample_ids"],
        "pred_ctx":           res["pred_ctx"],
        "pred_sst":           res["pred_sst"],
        "true_bins":          res["true_bins"],
        "target_feature_ids": res["target_feature_ids"],
        "metrics":            res["metrics"],
    }



def _merge_metadata(df, key, metadata, cols):
    """Left-join the requested metadata columns onto df by `key` (column or index)."""
    if metadata is None or not cols:
        return df
    md = metadata if key in metadata.columns else metadata.rename_axis(key).reset_index()
    keep = [key] + [c for c in cols if c in md.columns and c != key]
    return df.merge(md[keep].drop_duplicates(key), on=key, how="left")


# 3.  visualize_proteins
def visualize_proteins(
    model: ProtGPT | str,
    dataset: ExpressionDataset,
    metadata: pd.DataFrame | None = None,
    color_by: str | None = None,
    hover: list[str] | None = None,
    method: str = "umap",
    title: str | None = None,
    width: int = 1100,
    height: int = 700,
    save_path: str | None = None,
    **reducer_kwargs,
):
    """Interactive (plotly) view of the learned protein embeddings (learned-embedding models only).

    Args:
        model: a loaded ProtGPT model, or path to a .ckpt file.
        dataset: a loaded ExpressionDataset (provides protein names / index order).
        metadata: optional DataFrame keyed by 'protein_id' (column or index) with annotation columns.
        color_by: metadata column to colour by (categorical → discrete palette, numeric → continuous).
        hover: extra metadata columns to show on hover.
        method: 'umap' | 'tsne' | 'pca' (defaults from _reduce).
        title / width / height: plot title and size.
        save_path: '.html' → interactive file, other extensions → static image (needs kaleido).
        **reducer_kwargs: forwarded to the reducer (e.g. n_neighbors).

    Returns:
        plotly Figure.
    """
    if isinstance(model, str):
        model, _ = _load_model(model)

    feature_names = list(dataset.feature_names)
    device = next(model.parameters()).device
    indices = torch.arange(1, len(feature_names) + 1, device=device)  # 0 = padding
    emb_weights = model.feature_emb(indices).detach().cpu().numpy()

    coords = _reduce(emb_weights, method=method, **reducer_kwargs)
    m = method.upper()
    x, y = f"{m}_1", f"{m}_2"
    df = pd.DataFrame({"protein_id": feature_names, x: coords[:, 0], y: coords[:, 1]})

    wanted = list(dict.fromkeys(([color_by] if color_by else []) + (hover or [])))
    df = _merge_metadata(df, "protein_id", metadata, wanted)

    hover_cols = ["protein_id"] + [c for c in (hover or []) if c in df.columns]
    color = color_by if (color_by and color_by in df.columns) else None
    return _scatter_2d(df, x, y, color, hover_cols,
                       title or f"Protein embeddings ({m})", width, height, save_path)


# 4.  visualize_sst
def visualize_sst(
    sst: dict,
    metadata: pd.DataFrame | None = None,
    color_by: str | None = None,
    hover: list[str] | None = None,
    method: str = "umap",
    title: str | None = None,
    width: int = 1100,
    height: int = 700,
    save_path: str | None = None,
    color_order: list[str] | None = None,
    palette: list[str] | None = None,
    **reducer_kwargs,
):
    """Interactive (plotly) view of the Sample Summary Token (SST) embeddings from compute_sst().

    Args:
        sst: output dict from compute_sst() — must contain 'sst_emb' and 'sample_ids'.
        metadata: optional DataFrame keyed by 'sample_id' (column or index), e.g. ExpressionDataset.obs.
        color_by: metadata column to colour by (categorical → discrete palette, numeric → continuous).
        color_order: explicit legend order for a categorical color_by (defaults to frequency order).
        palette: explicit colour sequence aligned to color_order (defaults to Dark24+Light24).
        hover: extra metadata columns to show on hover (e.g. ['tissue', 'acquisition', 'split']).
        method: 'umap' | 'tsne' | 'pca' (defaults from _reduce).
        title / width / height: plot title and size.
        save_path: '.html' → interactive file, other extensions → static image (needs kaleido).
        **reducer_kwargs: forwarded to the reducer (e.g. n_neighbors=75, min_dist=0.3).

    Returns:
        plotly Figure.
    """
    sst_emb = sst["sst_emb"]
    sample_ids = list(sst["sample_ids"])

    coords = _reduce(sst_emb, method=method, **reducer_kwargs)
    m = method.upper()
    x, y = f"{m}_1", f"{m}_2"
    df = pd.DataFrame({"sample_id": sample_ids, x: coords[:, 0], y: coords[:, 1]})

    wanted = list(dict.fromkeys(([color_by] if color_by else []) + (hover or [])))
    df = _merge_metadata(df, "sample_id", metadata, wanted)

    hover_cols = ["sample_id"] + [c for c in (hover or []) if c in df.columns]
    color = color_by if (color_by and color_by in df.columns) else None
    return _scatter_2d(df, x, y, color, hover_cols,
                       title or f"SST embeddings ({m})", width, height, save_path,
                       color_order=color_order, palette=palette)


# ── attention-based protein–protein analysis (re-exported from omicsfm.attention) ──
# Imported at the bottom so these stay available as omicsfm.api.<name>;
# omicsfm.attention defers its import of this module to avoid the cycle.
from omicsfm.attention import (  # noqa: E402
    attention_map,
    eval_attention_heads,
    enrichment_curves,
    load_ppi_ground_truth,
)
