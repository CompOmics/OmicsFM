"""
Training loop for ProtGPT.
Usage: python -m protgpt.train --config config/setup.yaml
"""

PROJECT_NAME = 'protGPT'

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import yaml
import argparse
import logging
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from protgpt.data import ExpressionDataset, ExpressionCollator
from protgpt.architecture import ProtGPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# helpers
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(path: str, num_bins: int, detect_groups: bool = False,
                 max_group_size: int = 1, build_workers: int = 1) -> ExpressionDataset:
    """Load a unified .h5ad source as an ExpressionDataset (builds its cache if needed)."""
    log.info(f"Loading {path} ...")
    ds = ExpressionDataset(path, num_bins=num_bins, detect_groups=detect_groups,
                           max_group_size=max_group_size, build_workers=build_workers)
    log.info(f"  → {len(ds)} samples, {ds.num_features()} features ({ds.modality})\n")
    return ds


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run one pass over a dataloader, return avg losses."""
    model.eval()
    total_loss, ctx_loss, sst_loss, n = 0.0, 0.0, 0.0, 0
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, lc, lp = model.compute_loss(batch)
        num_samples = batch["role"].shape[0]
        total_loss += loss.item() * num_samples
        ctx_loss += lc.item() * num_samples
        sst_loss += lp.item() * num_samples
        n += num_samples
    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, ctx_loss / n, sst_loss / n


# main
def train(config_path: str):
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    ocfg = cfg["output"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"Device: {device}")

    # data — unified .h5ad source for both modalities (see protgpt/convert.py)
    pg_cfg = mcfg.get("protein_groups", {})
    pg_enabled = pg_cfg.get("enabled", False)
    abs_cfg = mcfg.get("absent_species", {})
    abs_enabled = abs_cfg.get("enabled", False)

    detect_groups = dcfg.get("detect_groups", False)
    # max_group_size>1 only when groups are modeled; otherwise grouped proteins are dropped
    max_group_size = pg_cfg.get("max_group_size", 1) if pg_enabled else 1
    # parallel cache build (one-time); does not affect the cache contents
    build_workers = dcfg.get("cache_build_workers", 1)

    train_ds = load_dataset(dcfg["train_path"], dcfg["num_bins"], detect_groups, max_group_size, build_workers)
    valid_ds = load_dataset(dcfg["valid_path"], dcfg["num_bins"], detect_groups, max_group_size, build_workers)
    test_ds  = load_dataset(dcfg["test_path"],  dcfg["num_bins"], detect_groups, max_group_size, build_workers)
    feature_names = train_ds.feature_names

    if train_ds.modality == "transcriptomics" and pg_enabled:
        log.warning("Protein groups not supported for transcriptomics — disabling")
        pg_enabled = False

    collator = ExpressionCollator(
        num_features=train_ds.num_features(),
        feature_context_ratio=mcfg["feature_context_ratio"],
        group_context_ratio=pg_cfg.get("group_context_ratio", 1.0),
        input_features=mcfg["input_features"],
        protein_groups_enabled=pg_enabled,
        input_groups=pg_cfg.get("input_groups", 0) if pg_enabled else 0,
        absent_species_enabled=abs_enabled,
        input_absent=abs_cfg.get("input_absent", 0),
        absent_context_ratio=abs_cfg.get("absent_context_ratio", 1.0),
    )
    dl_kwargs = dict(batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
                     pin_memory=True, collate_fn=collator, persistent_workers=tcfg["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    valid_dl = DataLoader(valid_ds, shuffle=False, **dl_kwargs)
    test_dl  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    # model
    model_cfg = {
        "num_features": train_ds.num_features(),
        "num_bins": dcfg["num_bins"],
        "d_model": mcfg["d_model"],
        "n_heads": mcfg["n_heads"],
        "n_layers": mcfg["n_layers"],
        "d_ff": mcfg["d_ff"],
        "dropout": mcfg["dropout"],
        "use_esmc": mcfg.get("use_esmc", False),
    }
    if pg_enabled:
        model_cfg["protein_groups"] = pg_cfg

    # build ESM-C lookup table from FASTA if enabled
    esmc_lookup = None
    if model_cfg["use_esmc"]:
        from protgpt.esmc_utils import build_esmc_lookup
        esmc_lookup = build_esmc_lookup(
            feature_names,
            mcfg["fasta_path"],
            mcfg.get("esmc_model", "esmc_600m"),
            cache_path=mcfg.get("esmc_cache"),
            device=device,
        )

    model = ProtGPT(model_cfg, esmc_lookup=esmc_lookup).to(device)

    # transfer learning: load pretrained weights
    transfer_cfg = cfg.get("transfer", {})
    transfer_ckpt = transfer_cfg.get("checkpoint_path")
    if transfer_ckpt:
        log.info(f"Loading pretrained weights from {transfer_ckpt} ...")
        ckpt = torch.load(transfer_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        # Strip torch.compile prefix if present
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # esmc_embedding is frozen and rebuilt from the current FASTA — never transfer it.
        # The checkpoint's copy is tied to its own protein list + index ordering.
        if "esmc_embedding.weight" in state_dict:
            log.debug("  Skipping esmc_embedding.weight (rebuilt from FASTA)")
            state_dict.pop("esmc_embedding.weight")

        # Learned protein embedding: remap rows by accession if protein list changed.
        if "feature_emb.weight" in state_dict:
            old_names = ckpt["feature_names"]
            old_w = state_dict["feature_emb.weight"]
            cur_w = model.feature_emb.weight.data
            if old_names != feature_names:
                old_idx = {name: i for i, name in enumerate(old_names)}
                # row 0 is padding in both → keep as-is in cur_w
                new_w = cur_w.clone()
                new_w[0] = old_w[0]
                fresh_init = []
                for i, name in enumerate(feature_names, start=1):
                    j = old_idx.get(name)
                    if j is not None:
                        new_w[i] = old_w[j + 1]  # +1 skips pad row in old
                    else:
                        fresh_init.append(name)
                dropped = [n for n in old_names if n not in set(feature_names)]
                state_dict["feature_emb.weight"] = new_w
                matched = len(feature_names) - len(fresh_init)
                log.info(f"  Remapped feature_emb.weight by accession: "
                         f"{matched}/{len(feature_names)} rows transferred")
                if fresh_init:
                    log.info(f"  Fresh init: {len(fresh_init)} proteins new to this run "
                             f"(not in checkpoint)")
                    preview = fresh_init[:10]
                    more = f" ... (+{len(fresh_init) - 10} more)" if len(fresh_init) > 10 else ""
                    log.debug(f"  Fresh init proteins: {preview}{more}")
                if dropped:
                    log.info(f"  Dropped: {len(dropped)} proteins in checkpoint but not "
                             f"in this run")
                    preview = dropped[:10]
                    more = f" ... (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
                    log.debug(f"  Dropped proteins: {preview}{more}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.debug(f"  Missing keys (not loaded): {missing}")
        if unexpected:
            log.debug(f"  Unexpected keys (ignored): {unexpected}")
        log.info(f"  Loaded pretrained weights from epoch {ckpt.get('epoch', '?')}, "
                 f"val_loss={ckpt.get('val_loss', '?')}")

        # Optionally freeze transformer layers
        freeze_epochs = transfer_cfg.get("freeze_epochs", 0)
        if transfer_cfg.get("freeze_transformer", False) and freeze_epochs > 0:
            log.info(f"  Freezing transformer layers for {freeze_epochs} epochs")
            for name, param in model.named_parameters():
                if "layers." in name:  # transformer blocks live in self.layers
                    param.requires_grad = False

    model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters (trainable): {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])

    # output dir
    save_dir = Path(ocfg["save_dir"]) / ocfg["experiment_name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.ckpt"

    # save config for reproducibility
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # wandb
    wandb.init(
        entity="compomics",
        project=PROJECT_NAME,
        name=ocfg["experiment_name"],
        config=cfg,
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # training loop
    best_val_loss = float("inf")
    patience_counter = 0
    val_freq = tcfg["val_frequency"]
    n_batches = len(train_dl)

    # Single validation cadence
    #   0 < val_freq < 1  → every `val_every` batches within an epoch (no end-of-epoch run)
    #   val_freq >= 1     → at the end of every `val_epochs` epochs
    #   val_freq <= 0     → at the end of every epoch
    if 0 < val_freq < 1:
        val_every = max(1, int(n_batches * val_freq))
        val_epochs = None
    else:
        val_every = None
        val_epochs = max(1, int(val_freq)) if val_freq and val_freq >= 1 else 1
    cadence = (f"every {val_every} batches" if val_every
               else f"every {val_epochs} epoch(s)")
    log.info(f"Training: {tcfg['max_epochs']} epochs, {n_batches} batches/epoch, "
             f"batch_size={tcfg['batch_size']}, validate {cadence}")

    def run_validation(tag=""):
        """Evaluate on valid, log, update best/patience, save best.

        The single place validation happens. Returns True when early-stopping
        patience is exhausted.
        """
        nonlocal best_val_loss, patience_counter
        vl, vc, vp = evaluate(model, valid_dl, device)
        log.info(f"  {tag}val_loss={vl:.4f} (ctx={vc:.4f} sst={vp:.4f})")
        wandb.log({"val/loss": vl, "val/ctx_loss": vc, "val/sst_loss": vp,
                   "global_step": global_step})
        if vl < best_val_loss - tcfg["early_stopping_delta"]:
            best_val_loss = vl
            patience_counter = 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val_loss": vl, "feature_names": feature_names}, best_path)
            log.info(f"  ✓ New best model saved (val_loss={vl:.4f})")
        else:
            patience_counter += 1
        model.train()
        return patience_counter >= tcfg["early_stopping_patience"]

    freeze_epochs = transfer_cfg.get("freeze_epochs", 0) if transfer_ckpt else 0

    global_step = 0
    for epoch in range(1, tcfg["max_epochs"] + 1):
        # Unfreeze transformer after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            log.info(f"Unfreezing transformer layers at epoch {epoch}")
            for name, param in model.named_parameters():
                if "layers." in name:  # only the transformer blocks we froze; keep ESM-C frozen
                    param.requires_grad = True
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"  Trainable parameters: {n_params:,}")

        model.train()
        epoch_loss, epoch_ctx, epoch_sst, epoch_n = 0.0, 0.0, 0.0, 0
        t0 = time.time()
        should_stop = False

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{tcfg['max_epochs']}", leave=True)
        for batch_idx, batch in enumerate(pbar, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, lc, lp = model.compute_loss(batch)
            loss.backward()
            if tcfg["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["gradient_clip"])
            optimizer.step()

            num_samples = batch["role"].shape[0]
            epoch_loss += loss.item() * num_samples
            epoch_ctx += lc.item() * num_samples
            epoch_sst += lp.item() * num_samples
            epoch_n += num_samples

            # update progress bar with running averages
            pbar.set_postfix(
                loss=f"{epoch_loss / epoch_n:.4f}",
                ctx=f"{epoch_ctx / epoch_n:.4f}",
                sst=f"{epoch_sst / epoch_n:.4f}",
            )

            # log step-level metrics to wandb
            wandb.log({
                "train/step_loss": loss.item(),
                "train/step_ctx_loss": lc.item(),
                "train/step_sst_loss": lp.item(),
                "global_step": global_step,
            })

            global_step += 1

            # intra-epoch validation (fractional val_frequency only)
            if val_every and batch_idx % val_every == 0:
                if run_validation(f"[epoch {epoch} batch {batch_idx}/{n_batches}] "):
                    should_stop = True
                    break

        # epoch summary (always logged)
        dt = time.time() - t0
        tl = epoch_loss / max(epoch_n, 1)
        tc = epoch_ctx / max(epoch_n, 1)
        tp = epoch_sst / max(epoch_n, 1)
        log.info(f"Epoch {epoch}/{tcfg['max_epochs']} ({dt:.0f}s) — train_loss={tl:.4f} (ctx={tc:.4f} sst={tp:.4f})")
        wandb.log({
            "train/epoch_loss": tl,
            "train/epoch_ctx_loss": tc,
            "train/epoch_sst_loss": tp,
            "epoch": epoch,
        })

        # epoch-based validation (only when not using the intra-epoch cadence)
        if not should_stop and val_epochs is not None and epoch % val_epochs == 0:
            should_stop = run_validation()

        if should_stop:
            log.info(f"Early stopping at epoch {epoch} (patience={tcfg['early_stopping_patience']})")
            break

    # test evaluation — load the best checkpoint if validation ever saved one
    if best_path.exists():
        log.info("Loading best model for test evaluation ...")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        log.warning("No best checkpoint saved (validation never ran?) — testing the final model")
    tl, tc, tp = evaluate(model, test_dl, device)
    log.info(f"Test loss={tl:.4f} (ctx={tc:.4f} sst={tp:.4f})")
    wandb.log({"test/loss": tl, "test/ctx_loss": tc, "test/sst_loss": tp})
    wandb.finish()
    log.info("Done.")


# entry point 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/setup.yaml")
    args = parser.parse_args()
    train(args.config)

