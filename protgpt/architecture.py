import torch
import torch.nn as nn
import torch.nn.functional as F


# Protein Group Encoder
class ProteinGroupEncoder(nn.Module):
    """Encode a set of protein embeddings into a single group embedding.

    Uses a CLS-token pattern: a learned group-size embedding is prepended to the
    member sequence, self-attention runs over all positions, and the CLS output
    becomes the group representation.

    For mean mode, simply computes the masked mean of member embeddings.
    """

    def __init__(self, d_model: int, max_group_size: int, mode: str = "attention",
                 n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.max_group_size = max_group_size
        self.mode = mode

        if mode == "attention":
            # group-size embedding: index by actual group size (0 unused, 1..max_group_size)
            self.size_embedding = nn.Embedding(max_group_size + 1, d_model)
            nn.init.normal_(self.size_embedding.weight, std=0.02)

            # self-attention layers (pre-norm style)
            self.attn_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(n_layers):
                self.attn_layers.append(
                    nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
                )
                self.norms.append(nn.LayerNorm(d_model))
            self.final_norm = nn.LayerNorm(d_model)

    def forward(self, embeddings: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, max_group_size, d_model) — member embeddings (padding positions are zero)
            group_sizes: (N,) — actual number of members per group (1 for individuals)

        Returns:
            (N, d_model) — one embedding per group/protein
        """
        if self.mode == "mean":
            return self._mean_pool(embeddings, group_sizes)
        return self._attention_pool(embeddings, group_sizes)

    def _mean_pool(self, embeddings, group_sizes):
        """Masked mean of member embeddings."""
        N, G, D = embeddings.shape
        # mask: (N, G, 1) — True for real members
        mask = torch.arange(G, device=embeddings.device).unsqueeze(0) < group_sizes.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (N, G, 1)
        summed = (embeddings * mask).sum(dim=1)  # (N, D)
        return summed / group_sizes.unsqueeze(1).float().clamp(min=1)

    def _attention_pool(self, embeddings, group_sizes):
        """CLS-token self-attention pooling."""
        N, G, D = embeddings.shape
        device = embeddings.device

        # prepend group-size CLS token: (N, 1+G, D)
        cls_emb = self.size_embedding(group_sizes)  # (N, D)
        seq = torch.cat([cls_emb.unsqueeze(1), embeddings], dim=1)  # (N, 1+G, D)

        # build key_padding_mask: True = ignore. shape (N, 1+G)
        # position 0 (CLS) is always valid; positions 1..G are valid if < group_size
        member_positions = torch.arange(G, device=device).unsqueeze(0)  # (1, G)
        member_mask = member_positions >= group_sizes.unsqueeze(1)  # (N, G) True=padding
        cls_mask = torch.zeros(N, 1, dtype=torch.bool, device=device)  # CLS always valid
        key_padding_mask = torch.cat([cls_mask, member_mask], dim=1)  # (N, 1+G)

        # self-attention layers (pre-norm)
        x = seq
        for norm, attn in zip(self.norms, self.attn_layers):
            h = norm(x)
            h, _ = attn(h, h, h, key_padding_mask=key_padding_mask)
            x = x + h

        x = self.final_norm(x)

        # return CLS position output
        return x[:, 0, :]  # (N, D)


# Transformer Block
class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head attention + FFN.

    Uses F.scaled_dot_product_attention directly so PyTorch can dispatch
    to FlashAttention / memory-efficient kernels with a 4-D bool mask.
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config['d_model']
        n_heads = config['n_heads']
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = config['dropout']
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, config['d_ff']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_ff'], d_model),
            nn.Dropout(config['dropout']),
        )

    def forward(self, x, attn_mask=None):
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)             # each (B, S, n_heads, head_dim)
        q = q.transpose(1, 2)                    # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                        # (B, n_heads, S, head_dim)
        h = h.transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ProtGPT Model
class ProtGPT(nn.Module):
    """
    Transformer that learns to predict masked protein expression bins.

    Input: dict from ExpressionCollator with role, feature_ids, bin_values tensors.
    Flow:
        1. Embed tokens: feature_emb + bin_emb (targets get no bin signal).
        2. Override position 0 with the Proteome Summary Token (PST).
        3. Run through transformer with custom attention mask:
           - PST + context attend to PST + context.
           - Each target attends to PST + context only (not other targets).
        4. Two prediction heads:
           a) ctx_head: predict target bins from their output representations.
           b) pst_head: predict target bins using PST output ⊕ protein identity.
        5. Loss = MSE(ctx_head) + MSE(pst_head).
    """

    def __init__(self, config: dict, esmc_lookup: torch.Tensor = None):
        super().__init__()
        self.config = config
        self.use_esmc = config.get('use_esmc', False)

        if self.use_esmc and esmc_lookup is not None:
            # frozen ESM-C embeddings + trainable projection to d_model
            esmc_dim = esmc_lookup.shape[1]
            self.esmc_embedding = nn.Embedding.from_pretrained(esmc_lookup, freeze=True)
            #NOTE: play around with how the projection is done
            # 1. self.esmc_proj = nn.Linear(esmc_dim, config['d_model'])  # simple linear projection
            # 2. mlp projection
            self.esmc_proj = nn.Sequential(
                nn.Linear(esmc_dim, esmc_dim // 2),
                nn.GELU(),
                nn.Linear(esmc_dim // 2, config['d_model']),
            )
        else:
            # learned embeddings (index 0 = padding token, real proteins are 1..num_features)
            self.feature_emb = nn.Embedding(config['num_features'] + 1, config['d_model'])

        # bin embedding: 0 = not detected (absent proteins in context), 1..num_bins = expression bins
        self.bin_emb = nn.Embedding(config['num_bins'] + 1, config['d_model'])

        # PST: learnable initial vector (same for every sample, scale matches embedding init)
        self.pst_token = nn.Parameter(torch.randn(config['d_model']) * 0.02)

        # transformer stack
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layers'])
        ])
        self.final_norm = nn.LayerNorm(config['d_model'])

        # head 1: full-context prediction (target repr → bin)
        self.ctx_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.GELU(),
            nn.Linear(config['d_model'], 1),
        )

        # head 2: PST-only prediction (PST repr ⊕ protein emb → bin)
        self.pst_head = nn.Sequential(
            nn.Linear(config['d_model'] * 2, config['d_model']),
            nn.GELU(),
            nn.Linear(config['d_model'], 1),
        )

        # NOTE: for later application we could add more heads that predict other properties from the PST or target representations
        # one could be predicting if a protein is present in the proteome or not
        # or predicting which of 2 proteins is highest expressed, etc.
        # this multi-task self-supervised setup could encourage the model to learn more general and useful representations of the proteome

        # protein group encoder (optional)
        pg_cfg = config.get('protein_groups', {})
        self.pg_enabled = pg_cfg.get('enabled', False)
        if self.pg_enabled:
            attn_cfg = pg_cfg.get('attention', {})
            self.group_encoder = ProteinGroupEncoder(
                d_model=config['d_model'],
                max_group_size=pg_cfg['max_group_size'],
                mode=pg_cfg.get('encoder', 'mean'),
                n_heads=attn_cfg.get('n_heads', 4),
                n_layers=attn_cfg.get('n_encoder_layers', 1),
            )

        self._init_weights()

    def _get_feature_emb(self, feature_ids):
        """Return protein embeddings, using either learned or ESM-C embeddings."""
        if self.use_esmc and hasattr(self, 'esmc_embedding'):
            return self.esmc_proj(self.esmc_embedding(feature_ids))
        return self.feature_emb(feature_ids)

    # weight init
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                # skip frozen pretrained embeddings (e.g. ESM-C lookup)
                if not m.weight.requires_grad:
                    continue
                nn.init.normal_(m.weight, std=0.02)

    # attention mask
    @staticmethod
    def _build_attention_mask(role, n_heads):
        """
        Build batched attention mask from role tensor.

        Attention rules (mask[query, key]):
          1. Context queries (role=1) attend to context keys only (role=1).
          2. Target queries  (role=2) attend to context keys only (role=1).
             → targets cannot see other targets or padding.
          3. Padding queries (role=0) attend to PST (position 0) only.
             → prevents NaN in softmax; padding outputs are never used.
          In short: only context positions (including PST) are valid keys,
          and only non-padding positions are real queries.

        Args:
            role: (batch, input) with 0=padding, 1=context/PST, 2=target
            n_heads: number of attention heads

        Returns:
            (batch, 1, input, input) bool mask for F.scaled_dot_product_attention.
            True = attend, False = block.
        """
        is_ctx_key = (role == 1).unsqueeze(1)       # (batch, 1, input)
        is_valid_query = (role >= 1).unsqueeze(2)   # (batch, input, 1)
        allowed = is_valid_query & is_ctx_key        # (batch, input, input)

        # Patch: let padding rows attend to PST (position 0) so softmax
        # gets at least one finite value and doesn't produce NaN → rule 3
        allowed[:, :, 0] |= (role == 0)

        # (batch, 1, input, input) — broadcasts over heads, no copy
        return allowed.unsqueeze(1)


    def forward(self, batch):
        """
        Batched forward with pre-collated sequences.

        Args:
            batch: dict from ExpressionCollator with keys:
                role         (batch, input): 0=padding, 1=context/PST, 2=target
                feature_ids  (batch, input): protein indices
                bin_values   (batch, input): true bin values

        Returns:
            dict with keys:
                transformer_out    (batch, seq_len, d_model): full transformer output
                pst_emb            (batch, d_model): proteome summary token embedding per sample
                pred_ctx           list of (num_targets,) tensors: ctx_head predictions per sample
                pred_pst           list of (num_targets,) tensors: pst_head predictions per sample
                true_bins          list of (num_targets,) tensors: ground-truth bins per sample
                target_feature_ids list of (num_targets,) tensors: protein IDs for each target
                role               (batch, seq_len): passthrough of the role tensor
        """
        role        = batch["role"]          # (batch, input)
        feature_ids = batch["feature_ids"]   # (batch, input, G)
        bin_values  = batch["bin_values"]    # (batch, input)
        group_sizes = batch.get("group_sizes")  # (batch, input) or None

        batch_size, _ = role.shape

        # protein identity embeddings
        if self.pg_enabled and group_sizes is not None:
            # protein groups: route individuals through direct lookup, groups through encoder
            B, S, G = feature_ids.shape
            d = self.config['d_model']
            flat_ids = feature_ids.reshape(B * S, G)
            flat_sizes = group_sizes.reshape(B * S)

            is_pg = flat_sizes > 1
            pg_idx = is_pg.nonzero(as_tuple=True)[0]
            indiv_idx = (~is_pg).nonzero(as_tuple=True)[0]

            prot_emb = torch.zeros(B * S, d, device=feature_ids.device)

            # .to(prot_emb.dtype): under autocast the ESM-C projection / group encoder
            # may return bf16, but prot_emb is float32 — index_put requires a dtype match.
            if indiv_idx.numel() > 0:
                prot_emb[indiv_idx] = self._get_feature_emb(flat_ids[indiv_idx, 0]).to(prot_emb.dtype)

            if pg_idx.numel() > 0:
                pg_emb = self._get_feature_emb(flat_ids[pg_idx])
                pg_out = self.group_encoder(pg_emb, flat_sizes[pg_idx])
                prot_emb[pg_idx] = pg_out.to(prot_emb.dtype)

            prot_emb = prot_emb.reshape(B, S, d)
        else:
            # no groups: all positions are individual, use first column of feature_ids
            prot_emb = self._get_feature_emb(feature_ids[:, :, 0])

        # bin embeddings: targets get no bin signal (zeroed out), only protein identity
        bin_emb = self.bin_emb(bin_values)
        bin_emb = bin_emb * (role != 2).unsqueeze(-1)  # zero out target bins
        token_emb = prot_emb + bin_emb  # (batch, input, d)
        token_emb[:, 0, :] = self.pst_token  # override position 0 with PST

        # attention mask from role tensor (vectorized)
        attn_mask = self._build_attention_mask(role, self.config['n_heads'])

        # transformer (fully batched)
        x = token_emb
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.final_norm(x)  # (batch, input, d)

        # extract PST embeddings (position 0 for every sample)
        pst_emb = x[:, 0, :]  # (batch, d)

        # vectorized target extraction across the whole batch
        tgt_mask = (role == 2)  # (B, S)
        batch_idx, seq_idx = tgt_mask.nonzero(as_tuple=True)

        if batch_idx.numel() > 0:
            tgt_repr = x[batch_idx, seq_idx]           # (N_total, d)
            pred_ctx = self.ctx_head(tgt_repr).squeeze(-1)  # (N_total,)

            pst_expanded = pst_emb[batch_idx]           # (N_total, d)
            tgt_prot = prot_emb[batch_idx, seq_idx]     # (N_total, d)
            pst_input = torch.cat([pst_expanded, tgt_prot], dim=-1)
            pred_pst = self.pst_head(pst_input).squeeze(-1)  # (N_total,)

            true_bins_flat = bin_values[batch_idx, seq_idx].float()
            tgt_ids_flat = feature_ids[batch_idx, seq_idx]

            # split back into per-sample lists
            counts = tgt_mask.sum(dim=1)  # (B,)
            all_pred_ctx = list(pred_ctx.split(counts.tolist()))
            all_pred_pst = list(pred_pst.split(counts.tolist()))
            all_true_bins = list(true_bins_flat.split(counts.tolist()))
            all_target_ids = list(tgt_ids_flat.split(counts.tolist()))
        else:
            empty_f = torch.tensor([], device=x.device)
            empty_l = torch.tensor([], device=x.device, dtype=torch.long)
            all_pred_ctx = [empty_f] * batch_size
            all_pred_pst = [empty_f] * batch_size
            all_true_bins = [empty_f] * batch_size
            all_target_ids = [empty_l] * batch_size

        return {
            "transformer_out":    x,
            "pst_emb":            pst_emb,
            "pred_ctx":           all_pred_ctx,
            "pred_pst":           all_pred_pst,
            "true_bins":          all_true_bins,
            "target_feature_ids": all_target_ids,
            "role":               role,
        }

    def compute_loss(self, batch):
        """
        Run forward pass and compute training losses.

        Per-sample MSE is averaged across samples (each sample weighted equally,
        regardless of its target count).

        Args:
            batch: dict from ExpressionCollator (same as forward()).

        Returns:
            (total_loss, ctx_loss, pst_loss)
        """
        role = batch["role"]
        device = role.device
        tgt_mask = (role == 2)  # (B, S)
        counts = tgt_mask.sum(dim=1)  # (B,)

        # skip full forward if no targets at all
        has_targets = counts > 0
        if not has_targets.any():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        out = self.forward(batch)

        # extract flat predictions for samples that have targets
        batch_idx, seq_idx = tgt_mask.nonzero(as_tuple=True)
        pred_ctx_flat = torch.cat(out["pred_ctx"])  # (N_total,)
        pred_pst_flat = torch.cat(out["pred_pst"])  # (N_total,)
        true_flat = torch.cat(out["true_bins"])      # (N_total,)

        # per-target squared errors
        se_ctx = (pred_ctx_flat - true_flat).square()
        se_pst = (pred_pst_flat - true_flat).square()

        # per-sample MSE via scatter_add, then average across samples
        valid_counts = counts[has_targets].float()
        n_valid = valid_counts.numel()

        sample_se_ctx = torch.zeros(role.shape[0], device=device)
        sample_se_pst = torch.zeros(role.shape[0], device=device)
        sample_se_ctx.scatter_add_(0, batch_idx, se_ctx)
        sample_se_pst.scatter_add_(0, batch_idx, se_pst)

        avg_ctx = (sample_se_ctx[has_targets] / valid_counts).sum() / n_valid
        avg_pst = (sample_se_pst[has_targets] / valid_counts).sum() / n_valid
        return avg_ctx + avg_pst, avg_ctx, avg_pst
