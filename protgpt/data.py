"""
Unified data layer for ProtGPT.

Both modalities (proteomics, transcriptomics) share one on-disk **source** format —
a `.h5ad` file with raw (unbinned) values, protein/gene accessions in `var_names`,
and per-sample metadata in `obs` (see docs/unified_data_format_plan.md and
protgpt/convert.py).

`ExpressionDataset` reads an h5ad source and, on first use, builds a compact,
**binned**, ragged CSR **cache** (flat `.npy` arrays) keyed by the binning/group
config. The cache is memory-mapped (or loaded fully into RAM when it fits), so the
training hot-loop never touches the h5ad and never holds the full matrix in RAM.

`ExpressionCollator` assembles batches: a `[SST | individuals | groups | absent]`
sequence with per-type subsampling. Protein groups are stored ragged and padded to
the **batch-local** max group size; absent species are sampled on the fly.
"""

import hashlib
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import anndata as ad

log = logging.getLogger(__name__)

CACHE_VERSION = 2  # bumped: feature-axis rename changed cache file names
_BUILD_BLOCK = 8192  # rows per block when streaming the h5ad during cache build


# ── helpers ──────────────────────────────────────────────────────────────────

def _index_dtype_for(n_var: int):
    """Smallest unsigned int dtype that indexes n_var columns."""
    return np.uint16 if n_var <= np.iinfo(np.uint16).max else np.uint32


def _segmented_bins(vals: np.ndarray, lengths: np.ndarray, num_bins: int) -> np.ndarray:
    """Vectorized rank-based binning of many samples at once.

    `vals` is the concatenation of every sample's detected values in a block;
    `lengths` gives each sample's count. Returns a uint8 array (same length as
    `vals`) of bins in 1..num_bins, where each sample is binned independently by
    within-sample value rank.

    Equivalent to running, per sample,
        order = argsort(v, kind="stable"); ranks[order] = arange(1, n+1)
        bins  = ceil(ranks / n * num_bins)
    but as a single segmented sort over the whole block. A stable lexsort by
    (sample, value) reproduces the per-sample stable argsort exactly (ties keep
    original order), so the output matches per-row normalization but faster.
    """
    n = len(vals)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    n_rows = len(lengths)
    rows = np.repeat(np.arange(n_rows), lengths)          # sample id per element
    starts = np.zeros(n_rows, dtype=np.int64)             # block-local row offsets
    np.cumsum(lengths[:-1], out=starts[1:])

    order = np.lexsort((vals, rows))                      # stable: by sample, then value
    within = np.arange(n) - starts[rows[order]]           # 0-based rank within its sample
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = within + 1                             # scatter back to original order

    row_len = lengths[rows].astype(np.float64)
    return np.ceil(ranks / row_len * num_bins).astype(np.uint8)


def _split_row_groups(ids: np.ndarray, bins: np.ndarray, vals: np.ndarray,
                      max_group_size: int):
    """Split one sample's detected proteins into individuals + groups.

    `ids` (1-based) and `bins` are precomputed; `vals` are the raw abundances used
    for group detection. Returns (ind_ids, ind_bins, groups) where groups is a list
    of (member_ids, group_bin). Group detection collapses proteins with identical
    RAW values; groups larger than max_group_size are dropped (legacy `counts <= G`).
    """
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    count_per_protein = counts[inv]
    is_ind = count_per_protein == 1
    ind_ids, ind_bins = ids[is_ind], bins[is_ind]

    groups = []
    if max_group_size >= 2:
        valid_vals = np.nonzero((counts >= 2) & (counts <= max_group_size))[0]
        for u in valid_vals:
            mask = inv == u
            members = ids[mask]
            group_bin = bins[mask][0]  # group shares one value → first member's bin
            groups.append((members, int(group_bin)))
    return ind_ids, ind_bins, groups


# ── dataset ──────────────────────────────────────────────────────────────────

class ExpressionDataset(Dataset):
    """Unified dataset over a `.h5ad` source for proteomics or transcriptomics.

    Args:
        h5ad_path: path to the source `.h5ad` (raw values, var_names = accessions).
        num_bins: number of expression bins (rank-based, per sample).
        detect_groups: collapse proteins with identical raw abundance into groups
            (proteomics only). max_group_size bounds / filters group size.
        max_group_size: groups larger than this are dropped; 1 disables grouping.
        cache_dir: where to put the derived cache (default: alongside the h5ad).
        ram_fraction: load the cache fully into RAM if it fits within this fraction
            of currently-available RAM; otherwise memory-map it.
    """

    def __init__(self, h5ad_path, num_bins=10, detect_groups=False, max_group_size=1,
                 cache_dir=None, ram_fraction=0.5):
        self.h5ad_path = str(h5ad_path)
        self.num_bins = num_bins
        self.detect_groups = detect_groups
        self.max_group_size = max_group_size if detect_groups else 1
        self.ram_fraction = ram_fraction

        self._read_source()  

        cache_dir = cache_dir or self._default_cache_dir()
        if not self._try_load_cache(cache_dir):
            self._build_cache(cache_dir)
            assert self._try_load_cache(cache_dir), "cache build did not produce a loadable cache"

    # -- source ---------------------------------------------------------------

    def _read_source(self):
        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        self.feature_names = list(adata.var_names)   # accessions, index order
        self.sample_ids = list(adata.obs_names)
        self.obs = adata.obs.copy()
        self.var = adata.var.copy()
        self.modality = adata.uns.get("modality", "unknown")
        self._n_features = len(self.feature_names)
        self._n_samples = adata.n_obs
        if adata.isbacked:
            adata.file.close()

    def num_features(self):
        return self._n_features

    def __len__(self):
        return self._n_samples

    # -- cache keying ---------------------------------------------------------

    def _var_names_hash(self):
        "serves as the cache fingerprint to evaluate if the same h5ad source was used to make it"
        h = hashlib.sha1()
        for name in self.feature_names:
            h.update(name.encode())
            h.update(b"\0")
        return h.hexdigest()[:16]

    def _cache_tag(self):
        return f"bins{self.num_bins}_grp{int(self.detect_groups)}x{self.max_group_size}"

    def _default_cache_dir(self):
        stem = os.path.splitext(os.path.basename(self.h5ad_path))[0]
        parent = os.path.dirname(self.h5ad_path) or "."
        return os.path.join(parent, f"cache_{stem}_{self._cache_tag()}")

    def _expected_meta(self):
        return {
            "cache_version": CACHE_VERSION,
            "n_samples": self._n_samples,
            "n_features": self._n_features,
            "num_bins": self.num_bins,
            "detect_groups": self.detect_groups,
            "max_group_size": self.max_group_size,
            "var_names_hash": self._var_names_hash(),
        }

    # -- cache load -----------------------------------------------------------

    def _try_load_cache(self, cache_dir):
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if meta != self._expected_meta():
            log.info(f"  Cache mismatch at {cache_dir}, will rebuild")
            return False

        total_bytes = sum(
            os.path.getsize(os.path.join(cache_dir, fn))
            for fn in os.listdir(cache_dir) if fn.endswith(".npy")
        )
        mmap = self._should_mmap(total_bytes)
        mode = "r" if mmap else None

        def load(name):
            return np.load(os.path.join(cache_dir, name), mmap_mode=mode)

        self.ind_feature_ids = load("ind_feature_ids.npy")
        self.ind_bins = load("ind_bins.npy")
        self.ind_offsets = np.asarray(load("ind_offsets.npy"))  # offsets always in RAM

        if self.detect_groups:
            self.grp_members = load("grp_members.npy")
            self.grp_member_offsets = np.asarray(load("grp_member_offsets.npy"))
            self.grp_bins = load("grp_bins.npy")
            self.grp_offsets = np.asarray(load("grp_offsets.npy"))

        log.info(f"  Loaded cache ({'mmap' if mmap else 'RAM'}, "
                 f"{total_bytes / 1e6:.0f} MB) from {cache_dir}")
        return True

    def _should_mmap(self, total_bytes):
        try:
            import psutil
            available = psutil.virtual_memory().available
        except Exception:
            return total_bytes > 2 * 1024**3  # no psutil → mmap anything over 2 GB
        return total_bytes > self.ram_fraction * available

    # -- cache build (streaming over the h5ad) --------------------------------

    def _build_cache(self, cache_dir):
        import h5py

        log.info(f"Building cache for {self.h5ad_path} ({self._cache_tag()}) ...")
        os.makedirs(cache_dir, exist_ok=True)
        idx_dtype = _index_dtype_for(self._n_features)

        ind_ids_parts, ind_bin_parts, ind_lens = [], [], []
        grp_member_parts, grp_size_parts, grp_bin_parts, grp_counts = [], [], [], []

        with h5py.File(self.h5ad_path, "r") as f:
            Xg = f["X"]
            assert Xg.attrs.get("encoding-type") == "csr_matrix", "X must be CSR"
            indptr = Xg["indptr"][:]
            data_ds, idx_ds = Xg["data"], Xg["indices"]
            n = self._n_samples

            for r0 in tqdm(range(0, n, _BUILD_BLOCK), desc="Building cache"):
                r1 = min(r0 + _BUILD_BLOCK, n)
                s, e = int(indptr[r0]), int(indptr[r1])
                block_data = data_ds[s:e]
                block_idx = idx_ds[s:e]
                lengths = (indptr[r0 + 1:r1 + 1] - indptr[r0:r1]).astype(np.int64)

                # binning is vectorized across the whole block (both modalities)
                bins_block = _segmented_bins(block_data, lengths, self.num_bins)
                ids_block = (block_idx.astype(np.int64) + 1).astype(idx_dtype, copy=False)

                if not self.detect_groups:
                    # every detected protein is an individual — no per-row work
                    ind_ids_parts.append(ids_block)
                    ind_bin_parts.append(bins_block)
                    ind_lens.extend(lengths.tolist())
                    continue

                # groups: per-row formation (proteomics only), reusing block bins/ids
                local_starts = np.zeros(len(lengths) + 1, dtype=np.int64)
                np.cumsum(lengths, out=local_starts[1:])
                for j in range(len(lengths)):
                    a, b = int(local_starts[j]), int(local_starts[j + 1])
                    ind_ids, ind_bins, groups = _split_row_groups(
                        ids_block[a:b], bins_block[a:b], block_data[a:b], self.max_group_size,
                    )
                    ind_ids_parts.append(ind_ids)
                    ind_bin_parts.append(ind_bins)
                    ind_lens.append(len(ind_ids))
                    grp_counts.append(len(groups))
                    for members, gbin in groups:
                        grp_member_parts.append(members.astype(idx_dtype, copy=False))
                        grp_size_parts.append(len(members))
                        grp_bin_parts.append(gbin)

        self._save_packed(cache_dir, idx_dtype, ind_ids_parts, ind_bin_parts, ind_lens,
                          grp_member_parts, grp_size_parts, grp_bin_parts, grp_counts)

        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump(self._expected_meta(), f, indent=2)
        log.info(f"  Saved cache to {cache_dir}")

    def _save_packed(self, cache_dir, idx_dtype, ind_ids_parts, ind_bin_parts, ind_lens,
                     grp_member_parts, grp_size_parts, grp_bin_parts, grp_counts):
        def save(name, arr):
            np.save(os.path.join(cache_dir, name), arr)

        ind_ids = (np.concatenate(ind_ids_parts) if ind_ids_parts
                   else np.zeros(0, idx_dtype))
        ind_bins = (np.concatenate(ind_bin_parts) if ind_bin_parts
                    else np.zeros(0, np.uint8))
        ind_offsets = np.zeros(len(ind_lens) + 1, dtype=np.int64)
        np.cumsum(np.asarray(ind_lens, dtype=np.int64), out=ind_offsets[1:])
        save("ind_feature_ids.npy", ind_ids.astype(idx_dtype, copy=False))
        save("ind_bins.npy", ind_bins.astype(np.uint8, copy=False))
        save("ind_offsets.npy", ind_offsets)

        if self.detect_groups:
            grp_members = (np.concatenate(grp_member_parts) if grp_member_parts
                           else np.zeros(0, idx_dtype))
            grp_member_offsets = np.zeros(len(grp_size_parts) + 1, dtype=np.int64)
            np.cumsum(np.asarray(grp_size_parts, dtype=np.int64), out=grp_member_offsets[1:])
            grp_bins = np.asarray(grp_bin_parts, dtype=np.uint8)
            grp_offsets = np.zeros(len(grp_counts) + 1, dtype=np.int64)
            np.cumsum(np.asarray(grp_counts, dtype=np.int64), out=grp_offsets[1:])
            save("grp_members.npy", grp_members.astype(idx_dtype, copy=False))
            save("grp_member_offsets.npy", grp_member_offsets)
            save("grp_bins.npy", grp_bins)
            save("grp_offsets.npy", grp_offsets)

    # -- item access ----------------------------------------------------------

    def __getitem__(self, idx):
        """Return one sample's data by slicing the flat cache arrays.

        Mental model — "parallel arrays + offset rulers":
          * Values live in flat, positionally-aligned arrays
          * Offset arrays hold no data — they are rulers marking where each thing's
            slice begins.

        Individuals need one ruler (sample -> its individuals: ind_offsets).
        Groups need two stacked rulers, since there are two levels of "many":
          grp_offsets         : sample -> its groups
          grp_member_offsets  : group  -> its members
        ``grp_members`` is every group's members concatenated end-to-end across all
        groups; ``grp_sizes`` (consecutive member-offset diffs) tells the collator
        where one group stops and the next begins; ``grp_bins`` is one bin per group.
        """
        a, b = int(self.ind_offsets[idx]), int(self.ind_offsets[idx + 1])
        item = {
            "ind_ids": torch.as_tensor(np.asarray(self.ind_feature_ids[a:b], dtype=np.int64)),
            "ind_bins": torch.as_tensor(np.asarray(self.ind_bins[a:b], dtype=np.int64)),
        }
        if self.detect_groups:
            g0, g1 = int(self.grp_offsets[idx]), int(self.grp_offsets[idx + 1]) 
            if g1 > g0: # does this sample have any groups?
                m0 = int(self.grp_member_offsets[g0])
                m1 = int(self.grp_member_offsets[g1])
                sizes = (self.grp_member_offsets[g0 + 1:g1 + 1]
                         - self.grp_member_offsets[g0:g1])
                item["grp_members"] = torch.as_tensor(
                    np.asarray(self.grp_members[m0:m1], dtype=np.int64))
                item["grp_sizes"] = torch.as_tensor(np.asarray(sizes, dtype=np.int64))
                item["grp_bins"] = torch.as_tensor(
                    np.asarray(self.grp_bins[g0:g1], dtype=np.int64))
            else:
                item["grp_members"] = torch.zeros(0, dtype=torch.long)
                item["grp_sizes"] = torch.zeros(0, dtype=torch.long)
                item["grp_bins"] = torch.zeros(0, dtype=torch.long)
        return item


# ── collator ───────────────────────────────────────────────────────────────

class ExpressionCollator:
    """Collate samples into the model's batched sequence tensors.

    Sequence layout: [SST, individuals…PAD, groups…PAD, absent…PAD].
    Individuals / groups / absent each have their own budget and context ratio.
    Groups are padded to the **batch-local** max group size; absent species are
    sampled on the fly from proteins not detected in the sample.

    Returns dict:
        role         (B, S):    0=padding, 1=context/SST, 2=target
        feature_ids  (B, S, G): protein ids per position (G = batch-local max group size)
        bin_values   (B, S):    expression bins
        group_sizes  (B, S):    members per position (only when protein_groups_enabled)
    """

    def __init__(self, num_features: int, feature_context_ratio: float = 0.85,
                 group_context_ratio: float = 1.0, input_features: int = 1024,
                 protein_groups_enabled: bool = False, input_groups: int = 0,
                 absent_species_enabled: bool = False, input_absent: int = 0,
                 absent_context_ratio: float = 1.0):
        self.num_features = num_features
        self.feature_context_ratio = feature_context_ratio
        self.group_context_ratio = group_context_ratio
        self.input_features = input_features
        self.protein_groups_enabled = protein_groups_enabled
        self.input_groups = input_groups if protein_groups_enabled else 0
        self.absent_species_enabled = absent_species_enabled
        self.input_absent = input_absent if absent_species_enabled else 0
        self.absent_context_ratio = absent_context_ratio

    def __call__(self, batch):
        B = len(batch)

        # 1) per-sample subsampling (done first so we know the batch-local group width)
        prepared = [self._prepare_sample(s) for s in batch]
        G = 1
        for p in prepared:
            if p["g_sizes"].numel() > 0:
                G = max(G, int(p["g_sizes"].max()))
        S = 1 + self.input_features + self.input_groups + self.input_absent

        role = torch.zeros(B, S, dtype=torch.long)
        feature_ids = torch.zeros(B, S, G, dtype=torch.long)
        bin_values = torch.zeros(B, S, dtype=torch.long)
        group_sizes = torch.zeros(B, S, dtype=torch.long)

        grp_seg = 1 + self.input_features
        abs_seg = grp_seg + self.input_groups

        for i, p in enumerate(prepared):
            role[i, 0] = 1  # SST

            # individuals (width 1)
            n_ind = len(p["ind_ids"])
            if n_ind > 0:
                pos = torch.arange(1, 1 + n_ind)
                n_ctx = max(1, int(n_ind * self.feature_context_ratio))
                role[i, pos[:n_ctx]] = 1
                role[i, pos[n_ctx:]] = 2
                feature_ids[i, pos, 0] = p["ind_ids"]
                bin_values[i, pos] = p["ind_bins"]
                group_sizes[i, pos] = 1

            # groups (members scattered via precomputed slot/col indices — no loop)
            n_grp = p["g_sizes"].numel()
            if n_grp > 0:
                pos = torch.arange(grp_seg, grp_seg + n_grp)
                n_ctx = max(1, int(n_grp * self.group_context_ratio))
                role[i, pos[:n_ctx]] = 1
                role[i, pos[n_ctx:]] = 2
                bin_values[i, pos] = p["g_bins"]
                group_sizes[i, pos] = p["g_sizes"]
                feature_ids[i, grp_seg + p["g_seg"], p["g_col"]] = p["g_members"]

            # absent (width 1, bin stays 0 = not detected)
            n_abs = len(p["abs_ids"])
            if n_abs > 0:
                pos = torch.arange(abs_seg, abs_seg + n_abs)
                n_ctx = int(n_abs * self.absent_context_ratio)
                if n_ctx > 0:
                    role[i, pos[:n_ctx]] = 1
                role[i, pos[n_ctx:]] = 2
                feature_ids[i, pos, 0] = p["abs_ids"]
                group_sizes[i, pos] = 1

        result = {"role": role, "feature_ids": feature_ids, "bin_values": bin_values}
        if self.protein_groups_enabled:
            result["group_sizes"] = group_sizes
        return result

    def _prepare_sample(self, sample):
        """Subsample one sample's individuals / groups and draw absent species."""
        ind_ids, ind_bins = sample["ind_ids"], sample["ind_bins"]
        n_ind = len(ind_ids)
        if n_ind > self.input_features:
            sel = torch.randperm(n_ind)[:self.input_features]
            ind_ids, ind_bins = ind_ids[sel], ind_bins[sel]
        else:
            perm = torch.randperm(n_ind)
            ind_ids, ind_bins = ind_ids[perm], ind_bins[perm]

        # groups: select up to input_groups whole groups and gather their members
        # vectorized (no per-group Python loop). g_seg/g_col give each member's
        # (slot, column) so __call__ can scatter them in one indexed assignment.
        g_members = torch.zeros(0, dtype=torch.long)
        g_sizes = torch.zeros(0, dtype=torch.long)
        g_bins = torch.zeros(0, dtype=torch.long)
        g_seg = torch.zeros(0, dtype=torch.long)
        g_col = torch.zeros(0, dtype=torch.long)
        if self.input_groups > 0 and "grp_sizes" in sample and sample["grp_sizes"].numel() > 0:
            sizes = sample["grp_sizes"]
            members_flat = sample["grp_members"]
            n_grp = sizes.numel()
            order = torch.randperm(n_grp)[:self.input_groups]
            off = torch.zeros(n_grp + 1, dtype=torch.long)
            torch.cumsum(sizes, dim=0, out=off[1:])
            sel_sizes = sizes[order]
            sel_starts = off[order]                                  # start of each group in members_flat
            k = order.numel()
            g_seg = torch.repeat_interleave(torch.arange(k), sel_sizes)   # slot (group) per member
            sel_off = torch.zeros(k + 1, dtype=torch.long)
            torch.cumsum(sel_sizes, dim=0, out=sel_off[1:])
            g_col = torch.arange(int(sel_off[-1])) - sel_off[g_seg]       # column within group
            g_members = members_flat[sel_starts[g_seg] + g_col]          # gathered members
            g_sizes = sel_sizes
            g_bins = sample["grp_bins"][order]

        abs_ids = torch.zeros(0, dtype=torch.long)
        if self.input_absent > 0:
            detected = ind_ids if g_members.numel() == 0 else torch.cat([ind_ids, g_members])
            abs_ids = self._sample_absent(detected)

        return {
            "ind_ids": ind_ids, "ind_bins": ind_bins,
            "g_members": g_members, "g_sizes": g_sizes, "g_bins": g_bins,
            "g_seg": g_seg, "g_col": g_col, "abs_ids": abs_ids,
        }

    def _sample_absent(self, detected_ids):
        """Sample `input_absent` protein ids (1-based) not detected in this sample."""
        k = self.input_absent
        detected = set(detected_ids.tolist())
        # rejection sampling, repeatedly sample random ids util we have k that where not detected
        # cap the number of tries to avoid infinite loops
        out, tries = [], 0
        max_tries = k * 20 + 100
        while len(out) < k and tries < max_tries:
            cand = int(torch.randint(1, self.num_features + 1, (1,)).item())
            if cand not in detected:
                out.append(cand)
                detected.add(cand)
            tries += 1
        return torch.tensor(out, dtype=torch.long)
