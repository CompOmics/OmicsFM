"""
Protein–protein relationships from attention maps.

The transformer layers are the only place information moves between proteins, so
accumulating attention weights between detected proteins over many forward passes
gives a protein×protein association score — a candidate ranking for protein–protein
relationships.

Most heads are noise and only some carry useful structure, and which ones isn't
known a priori. The workflow is therefore two-stage:

  1. eval_attention_heads(...) — score EVERY (layer, head) against a ground-truth
     PPI set (enrichment AUC), restricted to the model∩GT protein universe so all
     heads of one layer fit in memory. Returns a per-head table + heatmap; you read
     off the best head (e.g. layer 1, head 3).
  2. attention_map(..., head=(1, 3)) — full inference accumulating only that one
     head's protein×protein map (or head=None for the all-heads mean).

Attention is captured by temporarily replacing TransformerBlock.forward with an
explicit-softmax version (the training path keeps the fused SDPA kernel untouched).
"""

import contextlib
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from protgpt.architecture import TransformerBlock
from protgpt.api import _load_checkpoint, build_model_for_dataset
from protgpt.data import ExpressionDataset, ExpressionCollator

log = logging.getLogger(__name__)

# Ground-truth PPI databases ship inside the package (resolved relative to this
# file so it works from any working directory). For deployment these can instead
# be fetched from Zenodo into this directory.
GT_DIR_DEFAULT = str(Path(__file__).resolve().parent / "ppi_ground_truth")
DATABASES = ("corum", "gocc", "kegg", "reactome", "string")


# ── ground truth ────────────────────────────────────────────────────────────

def load_ppi_ground_truth(db: str, gt_dir: str = GT_DIR_DEFAULT):
    """Load a PPI ground-truth matrix + its protein order.

    Args:
        db: one of the matrices in gt_dir, e.g. 'corum', 'string', 'kegg',
            'reactome', 'gocc'.
        gt_dir: directory with '<db>_ground_truth_matrix.npz' (key 'matrix') and
            'ground_truth_proteins.tsv' (columns idx, protein).

    Returns:
        (matrix (G, G) float32 with 1=interacting / 0=not / NaN=unknown,
         proteins list[str] of length G).
    """
    import pandas as pd

    mat = np.load(os.path.join(gt_dir, f"{db}_ground_truth_matrix.npz"))["matrix"]
    prot = pd.read_csv(os.path.join(gt_dir, "ground_truth_proteins.tsv"), sep="\t")
    proteins = prot.sort_values("idx")["protein"].tolist()
    assert mat.shape[0] == len(proteins), (mat.shape, len(proteins))
    return mat.astype(np.float32), proteins


# ── attention capture (monkeypatch TransformerBlock.forward) ─────────────────

@contextlib.contextmanager
def _capture_attention(model, spec):
    """Temporarily patch TransformerBlock.forward to record attention into a buffer.

    spec controls what is stored per layer (keeps per-batch memory small):
        "mean_per_layer"   → head-mean (B, S, S) for every layer
        ("layer", L)       → full per-head (B, H, S, S) only for layer L
        ("head", L, H)     → single head (B, S, S) only for layer L, head H

    Yields a list `buf`; the caller clears it each batch and reads what it asked for.
    """
    for i, blk in enumerate(model.layers):
        blk._cap_idx = i  # tag layer index so the patched forward knows where it is

    buf: list = []
    orig = TransformerBlock.forward

    def patched(self, x, attn_mask=None):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale          # (B, H, S, S)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)                                # (B, H, S, S)

        idx = self._cap_idx
        if spec == "mean_per_layer":
            buf.append(attn.mean(dim=1).detach())                       # (B, S, S)
        elif spec[0] == "layer" and idx == spec[1]:
            buf.append(attn.detach())                                   # (B, H, S, S)
        elif spec[0] == "head" and idx == spec[1]:
            buf.append(attn[:, spec[2]].detach())                       # (B, S, S)

        h = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x

    TransformerBlock.forward = patched
    try:
        yield buf
    finally:
        TransformerBlock.forward = orig
        for blk in model.layers:
            if hasattr(blk, "_cap_idx"):
                del blk._cap_idx


# ── shared model/data setup ──────────────────────────────────────────────────

def _build(checkpoint_path, data_path, device, batch_size, num_workers,
           fasta_path, esmc_cache, shuffle):
    """Load checkpoint config + dataset + model + an all-context dataloader."""
    ckpt = _load_checkpoint(checkpoint_path)
    cfg, mcfg, dcfg = ckpt["config"], ckpt["config"]["model"], ckpt["config"]["data"]
    detect_groups = dcfg.get("detect_groups", dcfg.get("has_protein_groups", False))
    pg_cfg = mcfg.get("protein_groups", {})
    pg_enabled = pg_cfg.get("enabled", False)
    max_group_size = pg_cfg.get("max_group_size", 1) if pg_enabled else 1

    ds = ExpressionDataset(data_path, num_bins=dcfg["num_bins"],
                           detect_groups=detect_groups, max_group_size=max_group_size)
    model, _ = build_model_for_dataset(checkpoint_path, ds, device, fasta_path, esmc_cache)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    collator = ExpressionCollator(  # all detected proteins are context (no targets)
        num_features=ds.num_features(), feature_context_ratio=1.0,
        group_context_ratio=1.0, input_features=mcfg["input_features"],
        protein_groups_enabled=pg_enabled,
        input_groups=pg_cfg.get("input_groups", 0) if pg_enabled else 0,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=True, collate_fn=collator)
    return ds, model, dl, mcfg["n_layers"], mcfg["n_heads"]


def _accumulate_pairs(attn_bs, role_b, feat_b, uni_idx, n_uni, summ, cnt):
    """Scatter-add one sample's context×context sub-map into (n_uni, n_uni) sum/count.

    attn_bs: (S, S) attention for this sample; uni_idx maps a 0-based model protein
    index to its row/col in the accumulator (or -1 to skip).
    """
    pos_mask = role_b == 1
    pos_mask[0] = False                                   # drop the SST/PST token at position 0
    pos = pos_mask.nonzero(as_tuple=True)[0]
    if pos.numel() < 2:
        return
    upos = uni_idx[feat_b[pos] - 1]                       # (L,) universe row, -1 if outside
    keep = upos >= 0
    pos, upos = pos[keep], upos[keep]
    L = pos.numel()
    if L < 2:
        return
    sub = attn_bs[pos][:, pos].reshape(-1).float()        # (L*L,)
    ii = upos.repeat_interleave(L)
    jj = upos.repeat(L)
    flat = ii * n_uni + jj
    summ.view(-1).index_add_(0, flat, sub)
    cnt.view(-1).index_add_(0, flat, torch.ones_like(sub))


def _finalize(summ, cnt, symmetrize):
    """sum/count → per-pair mean; optional i↔j symmetrize; NaN diagonal."""
    import warnings
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices = never co-occurred
        m = summ.cpu().numpy() / cnt.cpu().numpy().astype(np.float32)
        if symmetrize:
            m = np.nanmean(np.stack([m, m.T], 0), 0).astype(np.float32)
    np.fill_diagonal(m, np.nan)
    return m


# ── 1. single-map accumulation (mean over heads, or one chosen head) ─────────

def attention_map(checkpoint_path, data_path, head=None, proteins=None,
                  n_epochs=1, batch_size=32, num_workers=0, device="cuda",
                  fasta_path=None, esmc_cache=None, symmetrize=True, accum_device=None):
    """Accumulate one protein×protein attention map over a dataset.

    Args:
        head: None → mean over all layers & heads (one map). (layer, head) → that
            single head only (0-based; also accepts a 'l1h3' string).
        proteins: optional list of accessions to restrict the universe (else full P).
        n_epochs: passes over the data (>1 re-subsamples the input window for coverage).
        symmetrize: if True (default) the two directions are averaged into one
            undirected map (attn[i, j] == attn[j, i]). Set False to keep the map
            DIRECTIONAL: attention is inherently directed, so attn[i, j] is how much
            protein i (query) attends to protein j (key) — which can carry biological
            meaning (e.g. an asymmetric regulator→target signal). The co-occurrence
            count is direction-independent, so all asymmetry comes from attention
            itself. The diagonal is NaN either way.
        accum_device: where the (P,P) accumulators live (default = device).

    Returns:
        dict: {'attn': (P', P') float32, 'count': (P', P'), 'proteins': [accessions]}.
        When symmetrize=False, rows are the "attending-from" protein and columns the
        "attended-to" protein, so attn[i, j] need not equal attn[j, i].
    """
    head = _parse_head(head)
    accum_device = accum_device or device
    ds, model, dl, n_layers, n_heads = _build(
        checkpoint_path, data_path, device, batch_size, num_workers,
        fasta_path, esmc_cache, shuffle=n_epochs > 1)

    feature_names = list(ds.feature_names)
    uni_idx, uni_names, n_uni = _universe(feature_names, proteins, device)

    summ = torch.zeros(n_uni, n_uni, dtype=torch.float32, device=accum_device)
    cnt = torch.zeros(n_uni, n_uni, dtype=torch.float32, device=accum_device)
    spec = "mean_per_layer" if head is None else ("head", head[0], head[1])

    with _capture_attention(model, spec) as buf, torch.no_grad():
        for ep in range(n_epochs):
            for batch in tqdm(dl, desc=f"attention {ep + 1}/{n_epochs}"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                buf.clear()
                model(batch)
                attn = (torch.stack(buf, 0).mean(0) if head is None else buf[0])  # (B, S, S)
                role, feat = batch["role"], batch["feature_ids"][:, :, 0]
                for b in range(role.shape[0]):
                    _accumulate_pairs(attn[b].to(accum_device), role[b].to(accum_device),
                                      feat[b].to(accum_device), uni_idx, n_uni, summ, cnt)

    return {"attn": _finalize(summ, cnt, symmetrize),
            "count": cnt.cpu().numpy().astype(np.int64),
            "proteins": uni_names}


# ── 2. per-head benchmarking against ground truth ────────────────────────────

def eval_attention_heads(checkpoint_path, data_path, db="combined",
                         gt_dir=GT_DIR_DEFAULT, k=100, n_epochs=1,
                         batch_size=32, num_workers=0, device="cuda",
                         fasta_path=None, esmc_cache=None, accum_device=None,
                         plot=True):
    """Score every (layer, head) by how well its attention recovers known PPIs.

    Holding all (layer, head) maps at once is infeasible (~48 × P² ≈ 78 GB at P≈20k),
    so this makes ONE data pass per layer, holding only that layer's `n_heads`
    accumulators (~13 GB) plus a single shared co-occurrence count. Each head's map
    is scored by enrichment AUC@k against the ground truth, restricted to the
    model∩ground-truth protein universe.

    Args:
        db: "combined" (all of DATABASES, scored per-db and averaged), a single db
            name ("corum", "string", ...), or a list of db names.
        k: top-k for the enrichment curve.
        plot: if True, return a plotly heatmap (of the average when multiple dbs).

    Returns:
        dict: {'auc': {db: (n_layers, n_heads)}, 'auc_avg': (n_layers, n_heads),
               'best': (layer, head) by average, 'databases': [...],
               'n_proteins': int, 'figure': plotly Figure or None}.
    """
    accum_device = accum_device or device
    dbs = list(DATABASES) if db == "combined" else ([db] if isinstance(db, str) else list(db))

    ds, model, dl, n_layers, n_heads = _build(
        checkpoint_path, data_path, device, batch_size, num_workers,
        fasta_path, esmc_cache, shuffle=n_epochs > 1)
    feature_names = list(ds.feature_names)

    # universe = model ∩ ground truth (all dbs share one protein order)
    _, gt_prot = load_ppi_ground_truth(dbs[0], gt_dir)
    gt_idx_of = {p: i for i, p in enumerate(gt_prot)}
    model_pos = np.where([p in gt_idx_of for p in feature_names])[0]
    gt_sub_idx = np.array([gt_idx_of[feature_names[i]] for i in model_pos])
    n_uni = len(model_pos)
    log.info(f"model ∩ ground-truth: {n_uni} proteins; databases={dbs}")

    gt_common = {}
    for d in dbs:
        mat, _ = load_ppi_ground_truth(d, gt_dir)
        gt_common[d] = mat[np.ix_(gt_sub_idx, gt_sub_idx)]
        del mat

    uni_idx = torch.full((len(feature_names),), -1, dtype=torch.long, device=accum_device)
    uni_idx[torch.as_tensor(model_pos, device=accum_device)] = torch.arange(n_uni, device=accum_device)

    auc = {d: np.full((n_layers, n_heads), np.nan, dtype=np.float32) for d in dbs}
    for L in range(n_layers):
        # one count matrix shared by all heads (co-occurrence is head-independent)
        summ = torch.zeros(n_heads, n_uni, n_uni, dtype=torch.float32, device=accum_device)
        cnt = torch.zeros(n_uni, n_uni, dtype=torch.float32, device=accum_device)
        with _capture_attention(model, ("layer", L)) as buf, torch.no_grad():
            for ep in range(n_epochs):
                pass_tag = f" pass {ep + 1}/{n_epochs}" if n_epochs > 1 else ""
                for batch in tqdm(dl, desc=f"layer {L + 1}/{n_layers}{pass_tag}"):
                    batch = {kk: v.to(device, non_blocking=True) for kk, v in batch.items()}
                    buf.clear()
                    model(batch)
                    _scatter_layer(buf[0], batch["role"], batch["feature_ids"][:, :, 0],
                                   uni_idx, n_uni, summ, cnt, accum_device)
        for h in range(n_heads):
            m = _finalize(summ[h], cnt, symmetrize=True)
            for d in dbs:
                auc[d][L, h] = _enrichment_auc(m, gt_common[d], k)
        log.info(f"layer {L}: " + "  ".join(f"{d} best@head={np.nanmax(auc[d][L]):.1f}" for d in dbs))
        del summ, cnt

    auc_avg = np.nanmean(np.stack([auc[d] for d in dbs], 0), 0)
    best = tuple(int(x) for x in np.unravel_index(np.nanargmax(auc_avg), auc_avg.shape))
    fig = _auc_heatmap(auc_avg, "average" if len(dbs) > 1 else dbs[0]) if plot else None
    return {"auc": auc, "auc_avg": auc_avg, "best": best, "databases": dbs,
            "n_proteins": n_uni, "figure": fig}


def _scatter_layer(attn, role, feat, uni_idx, n_uni, summ, cnt, dev):
    """Accumulate one batch's context×context attention for ALL heads of a layer.

    attn: (B, H, S, S). The universe rows + pair indices are computed once per sample
    (shared across heads), the count is added once, and each head's sub-map is
    scattered into summ[h]. Everything stays on `dev` (no per-sample host transfers).
    """
    role = role.to(dev)
    feat = feat.to(dev)
    ctx = role == 1
    ctx[:, 0] = False                                       # drop the SST token
    urow = torch.where(ctx, uni_idx[(feat - 1).clamp(min=0)], torch.full_like(feat, -1))
    H = summ.shape[0]
    for b in range(role.shape[0]):
        sel = (urow[b] >= 0).nonzero(as_tuple=True)[0]
        if sel.numel() < 2:
            continue
        u = urow[b, sel]
        flat = (u.view(-1, 1) * n_uni + u.view(1, -1)).reshape(-1)   # (L*L,) shared by all heads
        cnt.view(-1).index_add_(0, flat, torch.ones(flat.numel(), device=dev))
        sub = attn[b].to(dev)[:, sel][:, :, sel].reshape(H, -1).float()   # (H, L*L)
        for h in range(H):
            summ[h].view(-1).index_add_(0, flat, sub[h])


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_head(head):
    if head is None or isinstance(head, tuple):
        return head
    if isinstance(head, str):  # 'l1h3'
        import re
        m = re.fullmatch(r"[lL](\d+)[hH](\d+)", head)
        if not m:
            raise ValueError(f"head string must look like 'l1h3', got {head!r}")
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f"head must be None, (layer, head), or 'l#h#', got {head!r}")


def _universe(feature_names, proteins, device):
    if proteins is None:
        n = len(feature_names)
        return (torch.arange(n, device=device), list(feature_names), n)
    keep = set(proteins)
    pos = [i for i, p in enumerate(feature_names) if p in keep]
    uni_idx = torch.full((len(feature_names),), -1, dtype=torch.long, device=device)
    uni_idx[torch.as_tensor(pos, device=device)] = torch.arange(len(pos), device=device)
    return uni_idx, [feature_names[i] for i in pos], len(pos)


def _enrichment_curve(score, gt_sub, k):
    """Mean enrichment@1..k of `score`'s top-k vs binary ground truth gt_sub.

    Enrichment@k = (precision@k among labeled top-k) / (per-row positive rate).
    Returns (curve (k,), n_used) where n_used is the number of contributing rows
    (targets that have at least one positive and one labeled partner).
    """
    s = np.where(np.isnan(score), -np.inf, score).astype(np.float32, copy=False)
    np.fill_diagonal(s, -np.inf)
    k = min(k, s.shape[1] - 1)
    part = np.argpartition(-s, k, axis=1)[:, :k]
    part_scores = np.take_along_axis(s, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    top = np.take_along_axis(part, order, axis=1)             # (N, k)

    rows = np.arange(top.shape[0])[:, None]
    labels = gt_sub[rows, top]
    row_pos = np.nansum(gt_sub == 1, axis=1).astype(np.float64)
    row_lab = np.nansum(~np.isnan(gt_sub), axis=1).astype(np.float64)
    pos = (labels == 1).astype(np.float64)
    lab = (~np.isnan(labels)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline = row_pos / row_lab
        enr = (np.cumsum(pos, axis=1) / np.cumsum(lab, axis=1)) / baseline[:, None]
    valid = (row_pos > 0) & (row_lab > 0)
    if valid.sum() == 0:
        return np.full(k, np.nan), 0
    return np.nanmean(enr[valid], axis=0), int(valid.sum())


def _enrichment_auc(score, gt_sub, k):
    """AUC (∑ over k) of the enrichment curve — the per-head benchmark scalar."""
    curve, _ = _enrichment_curve(score, gt_sub, k)
    return float(np.nansum(curve))


def enrichment_curves(attn_map, db="combined", k=100, gt_dir=GT_DIR_DEFAULT, plot=True):
    """Enrichment curves of an accumulated attention map vs PPI ground truth.

    Args:
        attn_map: output of attention_map() — must contain 'attn' (P×P) and 'proteins'.
        db: "combined" (all of DATABASES), a single db name, or a list of names.
        k: top-k for the curves.
        plot: if True, also return a plotly figure with one curve per database.

    Returns:
        dict: {'curves': {db: (k,) enrichment@1..k}, 'auc': {db: float},
               'n_used': {db: int}, 'figure': plotly Figure or None}.
    """
    score = np.asarray(attn_map["attn"])
    proteins = list(attn_map["proteins"])
    dbs = list(DATABASES) if db == "combined" else ([db] if isinstance(db, str) else list(db))
    p_idx = {p: i for i, p in enumerate(proteins)}

    curves, aucs, n_used = {}, {}, {}
    for d in dbs:
        gt_mat, gt_prot = load_ppi_ground_truth(d, gt_dir)
        gt_of = {p: i for i, p in enumerate(gt_prot)}
        common = [p for p in proteins if p in gt_of]
        mi = np.array([p_idx[p] for p in common])
        gi = np.array([gt_of[p] for p in common])
        curve, n = _enrichment_curve(score[np.ix_(mi, mi)], gt_mat[np.ix_(gi, gi)], k)
        curves[d], aucs[d], n_used[d] = curve, float(np.nansum(curve)), n
        log.info(f"  {d:9s} enr@1={curve[0]:.2f}  enr@{len(curve)}={curve[-1]:.2f}  "
                 f"AUC={aucs[d]:.1f}  (n={n})")

    fig = _curve_plot(curves) if plot else None
    return {"curves": curves, "auc": aucs, "n_used": n_used, "figure": fig}


def _curve_plot(curves):
    import plotly.graph_objects as go
    fig = go.Figure()
    for d, c in curves.items():
        fig.add_scatter(x=list(range(1, len(c) + 1)), y=c, mode="lines",
                        name=f"{d} (AUC={np.nansum(c):.0f})")
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="random")
    fig.update_layout(title="PPI enrichment vs top-k", xaxis_title="top-k",
                      yaxis_title="enrichment (precision / baseline)",
                      plot_bgcolor="white", width=900, height=600,
                      legend=dict(itemsizing="constant"))
    fig.show()
    return fig


def _auc_heatmap(auc, db):
    import plotly.express as px
    n_layers, n_heads = auc.shape
    fig = px.imshow(auc, text_auto=".0f", aspect="auto", color_continuous_scale="Viridis",
                    labels=dict(x="head", y="layer", color=f"AUC@k ({db})"),
                    x=[f"h{h}" for h in range(n_heads)], y=[f"l{l}" for l in range(n_layers)],
                    title=f"Per-head PPI enrichment AUC vs {db}")
    fig.show()
    return fig
