"""
Harmonize ARCHS4 bulk metadata in place against the proteomics controlled vocabulary.

SEPARATE from build_bulk_dataset.py by design: the vocabulary / normalizer is tuned
iteratively (grow tissues.txt, add NER, Cellosaurus, ...), and this step must be
re-runnable WITHOUT rebuilding the ~45 GB matrix. It therefore reads ONLY the `obs`
group of the h5ad (never loads X), adds `<field>_harmonized` columns, writes `obs` back
in place (X and var untouched), and emits `normalization_map.tsv` (original -> normalized
per field, with similarity + status) for manual inspection.

Reuses the SapBERT normalizer in ../agentic-metadata.

Usage:
    # Harmonize in place:
    python transcriptomics/bulk/normalize_metadata.py --config transcriptomics/bulk/pipeline_config.yaml
    # Tune-only: write the QC map but DON'T modify the h5ad (iterate on the vocab fast):
    python transcriptomics/bulk/normalize_metadata.py --config transcriptomics/bulk/pipeline_config.yaml --dry-run
    # Point at a specific h5ad:
    python transcriptomics/bulk/normalize_metadata.py --config ... --expression path/to/expression.h5ad
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import h5py
from anndata.io import read_elem, write_elem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root


def _add_file_log(output_dir, name):
    """Also write logs to <output_dir>/logs/<name>.log so logs live with the bulk artifacts."""
    logdir = Path(output_dir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)


def load_normalizer(meta_cfg: dict):
    """Build the agentic-metadata SapBERT normalizer from the config's normalization block."""
    am_root = ROOT.parent / "agentic-metadata"
    if str(am_root) not in sys.path:
        sys.path.insert(0, str(am_root))
    from agentic_metadata.normalization.normalizer import MultiOntologyNormalizer
    from agentic_metadata.normalization.config import NormalizationConfig

    ncfg_dict = dict(meta_cfg.get("normalization", {}))
    for k in ("ontology_dir", "cache_dir"):
        if k in ncfg_dict and not Path(ncfg_dict[k]).is_absolute():
            ncfg_dict[k] = str((ROOT / ncfg_dict[k]).resolve())
    normalizer = MultiOntologyNormalizer(NormalizationConfig.from_dict(ncfg_dict))
    normalizer.load_all()
    return normalizer


# Curated map: ARCHS4 characteristics_ch1 key (lowercased) -> our target field.
# The submitter's own key tells us the entity type, so a tissue value never reaches the
# disease vocabulary (root fix for the Skin->dermatitis / Breast->breast cancer leakage).
FIELD_KEYS = {
    "tissue": {"tissue", "tissue type", "tissuetype", "tissue source", "source tissue",
               "organ", "tissue/cell type", "tissue/celltype", "anatomical site", "anatomic site"},
    "disease": {"disease", "disease state", "disease status", "diagnosis",
                "patient diagnosis", "condition", "disease/condition"},
    "cell_line": {"cell line", "cell_line", "cellline"},
}
# Fields allowed to fall back to free-text source_name_ch1 when no structured key exists.
# tissue = higher recall; disease stays key-only = high precision (no cross-category leak).
_SOURCE_FALLBACK = {"tissue"}


def parse_characteristics(s):
    """Parse ARCHS4 characteristics_ch1 'k: v,k: v' into {key_lower: value} (first wins)."""
    out = {}
    for seg in str(s).split(","):
        if ":" in seg:
            k, v = seg.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k and v and k not in out:
                out[k] = v
    return out


def _field_value(parsed, field, source_name):
    """Value to normalize for `field`: structured characteristics key first, then (for
    whitelisted fields only) the free-text source_name. Returns (value, origin)."""
    for k, v in parsed.items():
        if k in FIELD_KEYS.get(field, ()):
            return v, "key"
    if field in _SOURCE_FALLBACK and source_name:
        return source_name, "source"
    return None, None


def load_overrides(cfg):
    """Load manual overrides {field: {raw_lower: forced_term}}.

    Path resolution: cfg['metadata']['manual_overrides_file'] if set, else the
    standalone manual_overrides.yaml sitting next to this script (easy to find + edit).
    """
    p = cfg.get("metadata", {}).get("manual_overrides_file")
    path = Path(p) if (p and Path(p).is_absolute()) else \
        (Path(p) if p else Path(__file__).resolve().parent / "manual_overrides.yaml")
    if not path.exists():
        log.info(f"No manual overrides file at {path}")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    norm = {field: {str(k).strip().lower(): v for k, v in (m or {}).items()}
            for field, m in data.items()}
    log.info(f"Loaded manual overrides from {path}: "
             + ", ".join(f"{f}={len(m)}" for f, m in norm.items()))
    return norm


def harmonize(obs, normalizer, fields: dict, tracker, overrides=None):
    """Route each value to the right vocab via characteristics_ch1 keys, then normalize.

    A value reaches the disease vocabulary only if the submitter labeled it with a disease
    key (disease/diagnosis/...), so bare tissue names can no longer leak into disease.
    Manual overrides (manual_overrides.yaml) are applied BEFORE SapBERT and logged with
    status "override" so every forced mapping is visible in the QC map.
    """
    overrides = overrides or {}
    obs = obs.drop(columns=[c for c in obs.columns if c.endswith("_harmonized")], errors="ignore")
    chars = (obs["characteristics_ch1"] if "characteristics_ch1" in obs.columns
             else pd.Series([""] * len(obs), index=obs.index)).fillna("").astype(str)
    snames = (obs["source_name_ch1"] if "source_name_ch1" in obs.columns
              else pd.Series([""] * len(obs), index=obs.index)).fillna("").astype(str)
    parsed_list = [parse_characteristics(c) for c in chars]

    for entity in fields:
        ov = overrides.get(entity, {})
        cache, out_vals, n_key, n_ov = {}, [], 0, 0
        for parsed, sname in zip(parsed_list, snames):
            val, origin = _field_value(parsed, entity, sname)
            if not val:
                out_vals.append("")
                continue
            n_key += origin == "key"
            if val.strip().lower() in ov:                 # manual override (pre-SapBERT)
                norm = ov[val.strip().lower()]
                tracker.record(entity, val, norm, 1.0, "override")
                out_vals.append(norm)
                n_ov += 1
                continue
            if val not in cache:  # cached: one SapBERT (GPU) embed per unique value
                res = normalizer.normalize(val, entity_type=entity)
                norm = res.ontology_name if res.is_normalized else ""
                cache[val] = (norm or "", float(res.similarity),
                              "mapped" if res.is_normalized else "unmapped")
            norm, sim, status = cache[val]
            tracker.record(entity, val, norm, sim, status)  # per-sample -> count = frequency
            out_vals.append(norm)
        obs[f"{entity}_harmonized"] = out_vals
        log.info(f"  {entity}: {sum(bool(v) for v in out_vals)}/{len(out_vals)} mapped | "
                 f"{n_key} from a structured '{entity}' key | {n_ov} via manual override | "
                 f"{len(cache)} unique SapBERT values")
    return obs


def preview_obs_from_source(cfg: dict, n: int):
    """Build an obs-like frame from N random ARCHS4 source metadata rows (no matrix needed).

    Used for --preview: iterate on the vocabulary against real ARCHS4 free-text strings
    before/without building the expression matrix.
    """
    import archs4py as a4  # local import: only needed for preview
    F = str(ROOT / cfg["archs4_h5"])
    sc_thresh = float(cfg.get("qc", {}).get("max_singlecell_probability", 0.5))
    geo = np.asarray(a4.meta.field(F, "geo_accession"))
    idx = np.arange(len(geo))
    try:
        scp = np.asarray(a4.meta.field(F, "singlecellprobability"), dtype=float)
        idx = idx[scp < sc_thresh]
    except Exception:
        pass
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    pick = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
    cols = {}
    for fld in ("geo_accession", "series_id", "source_name_ch1", "characteristics_ch1", "title"):
        try:
            cols[fld] = np.asarray(a4.meta.field(F, fld))[pick]
        except Exception as e:
            log.warning(f"field {fld!r} unavailable: {e}")
    log.info(f"Preview: {len(pick)} random candidate samples (of {len(idx)} after sc-filter)")
    return pd.DataFrame(cols)


def main():
    parser = argparse.ArgumentParser(description="Harmonize bulk h5ad metadata in place (separate from build)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expression", default=None, help="Path to expression.h5ad (overrides config-derived)")
    parser.add_argument("--dry-run", action="store_true", help="Write normalization_map.tsv but do NOT modify the h5ad")
    parser.add_argument("--preview", type=int, default=None, metavar="N",
                        help="Map N random ARCHS4 source-metadata rows directly (no matrix needed); "
                             "writes only normalization_map_preview.tsv. For tuning the vocab.")
    parser.add_argument("--from-h5ad", action="store_true",
                        help="Read raw metadata from the h5ad obs instead of the pristine "
                             "metadata_raw.parquet snapshot (default is the snapshot).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    meta_cfg = cfg.get("metadata", {})
    if not meta_cfg.get("enabled", False):
        log.warning("metadata.enabled is false in config; nothing to do.")
        return

    out_dir = ROOT / cfg["output_dir"]
    _add_file_log(out_dir, "normalize")

    # --- preview mode: harmonize real ARCHS4 metadata directly, write the map only ---
    if args.preview:
        obs = preview_obs_from_source(cfg, args.preview)
        normalizer = load_normalizer(meta_cfg)
        from agentic_metadata.normalization.tracker import NormalizationTracker  # noqa: E402
        tracker = NormalizationTracker()
        harmonize(obs, normalizer, meta_cfg.get("normalization", {}).get("fields", {}), tracker,
                  load_overrides(cfg))
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "normalization_map_preview.tsv"
        tracker.write_report(report_path)
        log.info(f"Wrote PREVIEW normalization map ({args.preview} samples) -> {report_path}")
        return

    h5ad = Path(args.expression) if args.expression else out_dir / "all_transcripts" / "expression.h5ad"
    if not h5ad.exists():
        raise FileNotFoundError(f"{h5ad} not found — run build_bulk_dataset.py first.")

    # Source of RAW metadata: prefer the pristine snapshot so we always start from the
    # original (never compound on a previously-harmonized obs). Falls back to the h5ad obs.
    raw_snapshot = out_dir / "metadata_raw.parquet"
    with h5py.File(h5ad, "r") as f:
        n_samples = f["X"].shape[0]
    if raw_snapshot.exists() and not args.from_h5ad:
        log.info(f"Reading ORIGINAL raw metadata from {raw_snapshot} (pristine snapshot)")
        obs = pd.read_parquet(raw_snapshot)
        if len(obs) != n_samples:
            raise ValueError(f"snapshot rows ({len(obs)}) != h5ad samples ({n_samples}); "
                             f"rebuild or pass --from-h5ad")
    else:
        log.info(f"Reading obs from {h5ad} (X not loaded)")
        with h5py.File(h5ad, "r") as f:
            obs = read_elem(f["obs"])
    log.info(f"obs: {obs.shape}  columns={list(obs.columns)}")

    normalizer = load_normalizer(meta_cfg)  # also puts ../agentic-metadata on sys.path
    from agentic_metadata.normalization.tracker import NormalizationTracker  # noqa: E402
    tracker = NormalizationTracker()
    obs = harmonize(obs, normalizer, meta_cfg.get("normalization", {}).get("fields", {}), tracker,
                    load_overrides(cfg))

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "normalization_map.tsv"
    tracker.write_report(report_path)
    log.info(f"Wrote normalization QC report -> {report_path}")

    if args.dry_run:
        log.info("--dry-run: h5ad NOT modified. Inspect the map, tune the vocab, re-run.")
        return

    # Replace ONLY the obs group; X and var are untouched.
    log.info(f"Writing harmonized obs back into {h5ad} (X untouched)")
    with h5py.File(h5ad, "a") as f:
        del f["obs"]
        write_elem(f, "obs", obs)
    log.info("Done.")


if __name__ == "__main__":
    main()
