# loading data into object, normalization, transposition to have samples as columns and proteins as rows
import json
import os
import numpy as np
import logging
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


class ProteomeDataset(Dataset):
    def __init__(self, data, num_bins=10, detect_groups=False, max_group_size=1,
                 group_cache_path=None, include_absent=False):
        #NOTE: expected data is: columns: pxd_accession, run_name, protein_1, protein_2, ..., rows = samples
        self.data = data
        self.detect_groups = detect_groups
        self.max_group_size = max_group_size
        self.include_absent = include_absent
        if detect_groups:
            self._detect_protein_groups()  # before binning — uses raw abundance values!
        self.normalize_dataset(bins=num_bins)
        self.transpose_data()
        self._load_or_precompute_positions(group_cache_path)

    def __getitem__(self, idx):
        """Return precomputed sample data for the collator.

        Always returns (exp_prot_ids, exp_bins) for detected/expressed proteins.
        When include_absent=True, also returns absent_ids.
        """
        if self.include_absent:
            return self.exp_prot_ids[idx], self.exp_bins[idx], self.absent_ids[idx]
        return self.exp_prot_ids[idx], self.exp_bins[idx]

    def __len__(self):
        return len(self.data.columns)

    def num_proteins(self):
        return len(self.data.index)

    # ── precomputation ──────────────────────────────────────────────────

    def _load_or_precompute_positions(self, cache_path=None):
        """Load precomputed positions from cache or compute and save them."""
        n_samples = len(self.data.columns)

        if cache_path and os.path.exists(cache_path):
            logging.info(f"Loading precomputed positions from cache: {cache_path}")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (cache.get("n_samples") == n_samples
                    and cache.get("max_group_size") == self.max_group_size
                    and cache.get("n_proteins") == self.num_proteins()
                    and cache.get("detect_groups") == self.detect_groups
                    and cache.get("include_absent") == self.include_absent):
                self.exp_prot_ids = cache["exp_prot_ids"]
                self.exp_bins = cache["exp_bins"]
                if self.include_absent:
                    self.absent_ids = cache["absent_ids"]
                logging.info(f"  Loaded {n_samples} cached samples")
                return
            else:
                logging.info(f"  Cache mismatch, recomputing...")

        self._precompute_positions()

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            save_dict = {
                "exp_prot_ids": self.exp_prot_ids,
                "exp_bins": self.exp_bins,
                "n_samples": n_samples,
                "n_proteins": self.num_proteins(),
                "max_group_size": self.max_group_size,
                "detect_groups": self.detect_groups,
                "include_absent": self.include_absent,
            }
            if self.include_absent:
                save_dict["absent_ids"] = self.absent_ids
            torch.save(save_dict, cache_path)
            logging.info(f"  Saved position cache to {cache_path}")

    def _precompute_positions(self):
        """Precompute per-sample position tensors.

        For each sample, produces:
          prot_ids: (n_positions, max_group_size) — detected protein IDs (1-based), 0=padding
          bins:     (n_positions,) — expression bins (1..num_bins)
          absent_ids: (n_absent,) — protein IDs with zero expression (only when include_absent)

        When detect_groups=True: proteins with identical values are collapsed into groups.
        When detect_groups=False: every detected protein gets its own position (size 1).
        """
        G = self.max_group_size
        n_samples = len(self.data.columns)
        self.exp_prot_ids = []
        self.exp_bins = []
        if self.include_absent:
            self.absent_ids = []

        for idx in tqdm(range(n_samples), desc="Precomputing proteome metadata"):
            sample_bins = torch.tensor(self.data.iloc[:, idx].values, dtype=torch.long)

            # collect absent protein IDs before any filtering
            if self.include_absent:
                absent_mask = sample_bins == 0
                self.absent_ids.append(absent_mask.nonzero(as_tuple=False).squeeze(-1) + 1)  # 1-based

            if self.detect_groups:
                prot_ids, bins = self._precompute_with_groups(idx, sample_bins, G)
            else:
                prot_ids, bins = self._precompute_no_groups(sample_bins, G)

            self.exp_prot_ids.append(prot_ids)
            self.exp_bins.append(bins)

        logging.info(f"Precomputed positions for {n_samples} samples\n")

    def _precompute_no_groups(self, sample_bins, G):
        """Each detected protein becomes its own position (size 1)."""
        detected_mask = sample_bins > 0
        detected_indices = detected_mask.nonzero(as_tuple=True)[0]
        n_det = len(detected_indices)

        if n_det == 0:
            return torch.zeros(0, G, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        prot_ids = torch.zeros(n_det, G, dtype=torch.long)
        prot_ids[:, 0] = detected_indices + 1  # 1-based
        bins = sample_bins[detected_indices]
        return prot_ids, bins

    def _precompute_with_groups(self, idx, sample_bins, G):
        """Collapse proteins with identical values into groups."""
        gids = torch.tensor(self.group_ids.iloc[:, idx].values, dtype=torch.long)

        nonzero_mask = gids > 0
        nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
        nonzero_gids = gids[nonzero_indices]

        if nonzero_gids.numel() == 0:
            return torch.zeros(0, G, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        unique_gids, inverse, counts = torch.unique(
            nonzero_gids, return_inverse=True, return_counts=True
        )

        # filter out groups exceeding max_group_size
        valid = counts <= G
        if valid.sum() == 0:
            return torch.zeros(0, G, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        # keep only proteins belonging to valid groups
        valid_per_protein = valid[inverse]
        filtered_indices = nonzero_indices[valid_per_protein]
        filtered_inverse = inverse[valid_per_protein]

        # renumber groups contiguously
        valid_old_ids = valid.nonzero(as_tuple=True)[0]
        id_remap = torch.zeros(len(unique_gids), dtype=torch.long)
        id_remap[valid_old_ids] = torch.arange(len(valid_old_ids))
        new_inverse = id_remap[filtered_inverse]
        new_counts = counts[valid]
        n_positions = len(new_counts)

        # sort by group to cluster members
        sorted_order = new_inverse.argsort()
        sorted_prot_indices = filtered_indices[sorted_order] + 1  # 1-based
        sorted_bins = sample_bins[filtered_indices[sorted_order]]

        # build padded position tensors
        offsets = torch.zeros(n_positions + 1, dtype=torch.long)
        offsets[1:] = new_counts.cumsum(0)

        prot_ids = torch.zeros(n_positions, G, dtype=torch.long)
        bins = torch.zeros(n_positions, dtype=torch.long)

        for p in range(n_positions):
            s = offsets[p].item()
            n = new_counts[p].item()
            prot_ids[p, :n] = sorted_prot_indices[s:s + n]
            bins[p] = sorted_bins[s]

        return prot_ids, bins

    # ── group detection ─────────────────────────────────────────────────

    def _detect_protein_groups(self):
        """Detect protein groups by finding proteins with identical non-zero abundance values per sample.

        Must be called BEFORE normalization since binning can merge unrelated proteins.
        """
        proteome_cols = self.data.select_dtypes(include="number").columns
        vals = self.data[proteome_cols].values

        group_ids = np.zeros(vals.shape, dtype=int)

        for i in range(vals.shape[0]):
            row = vals[i]
            nonzero_mask = (row > 0) & ~np.isnan(row)
            if not nonzero_mask.any():
                continue

            nonzero_vals = row[nonzero_mask]
            nonzero_indices = np.where(nonzero_mask)[0]

            val_to_gid = {}
            gid = 1
            for j, v in enumerate(nonzero_vals):
                if v not in val_to_gid:
                    val_to_gid[v] = gid
                    gid += 1
                group_ids[i, nonzero_indices[j]] = val_to_gid[v]

        self.group_ids = pd.DataFrame(group_ids, index=self.data.index, columns=proteome_cols)

        # log statistics
        n_samples = vals.shape[0]
        total_detected = 0
        total_in_groups = 0
        total_groups = 0
        max_group_size_seen = 0
        for i in range(n_samples):
            gids_row = group_ids[i]
            nonzero_gids = gids_row[gids_row > 0]
            total_detected += len(nonzero_gids)
            _, counts = np.unique(nonzero_gids, return_counts=True)
            multi = counts > 1
            total_in_groups += counts[multi].sum()
            total_groups += multi.sum()
            if len(counts) > 0:
                max_group_size_seen = max(max_group_size_seen, counts.max())

        avg_detected = total_detected / n_samples
        avg_in_groups = total_in_groups / n_samples
        avg_groups = total_groups / n_samples
        pct = total_in_groups / total_detected * 100 if total_detected > 0 else 0
        logging.info(f"Protein groups (per sample avg): {avg_in_groups:.0f}/{avg_detected:.0f} detected proteins "
                     f"belong to a group ({pct:.1f}%), {avg_groups:.0f} groups, max group size={max_group_size_seen}")

    # ── normalization & transposition ───────────────────────────────────

    def normalize_dataset(self, bins=7):
        proteome_cols = self.data.select_dtypes(include="number").columns
        vals = self.data[proteome_cols].values.copy()

        detected = (vals > 0) & ~np.isnan(vals)
        result = np.zeros(vals.shape, dtype=int)

        for i in range(vals.shape[0]):
            mask = detected[i]
            if mask.any():
                row_vals = vals[i, mask]
                n_detected = mask.sum()
                order = np.argsort(row_vals)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, n_detected + 1)
                result[i, mask] = np.ceil(ranks / n_detected * bins).astype(int)

        self.data[proteome_cols] = result

        n_detected_avg = detected.sum(axis=1).mean()
        logging.info(f"Avg detected proteins per sample: {n_detected_avg:.0f}/{len(proteome_cols)}")
        logging.info(f"Binned dataset: 0=not detected, 1-{bins}=expression bins per sample.")

    def transpose_data(self):
        id_cols = self.data.select_dtypes(exclude="number").columns.tolist()
        sample_ids = self.data[id_cols].astype(str).agg('_'.join, axis=1)
        self.data = self.data.drop(columns=id_cols)
        sorted_cols = sorted(self.data.columns)
        self.data = self.data[sorted_cols]
        self.data.index = sample_ids
        self.data.index.name = 'sample_id'
        self.data = self.data.transpose()

        if self.detect_groups:
            self.group_ids = self.group_ids[sorted_cols]
            self.group_ids.index = sample_ids
            self.group_ids.index.name = 'sample_id'
            self.group_ids = self.group_ids.transpose()


class TranscriptomicsDataset(Dataset):
    """Dataset for single-cell transcriptomics data stored as sparse CSR matrix.

    Loads expression.npz + metadata + protein_columns from build_sc_dataset.py output.
    Row range selects the split (train/valid/test) from the ordered matrix.

    Cache is stored as memory-mapped numpy arrays with compact dtypes (int16 for
    protein IDs, uint8 for bins) in a directory alongside the .npz file. This avoids
    loading the full dataset into RAM — only pages touched by the current batch are
    resident in memory.

    Produces the same (exp_prot_ids, exp_bins[, absent_ids]) interface as ProteomeDataset.
    """

    _CACHE_VERSION = 2  # bump when cache format changes

    def __init__(self, expression_path, protein_columns_path,
                 num_bins=10, cache_path=None, include_absent=False):
        self.include_absent = include_absent
        self.num_bins = num_bins

        # Load protein column names
        with open(protein_columns_path) as f:
            self.protein_columns = json.load(f)
        self._n_proteins = len(self.protein_columns)

        # Try loading memory-mapped cache first (avoids loading the sparse matrix)
        if cache_path and self._try_load_mmap_cache(cache_path):
            return

        # Cache miss — load sparse matrix and build cache
        from scipy.sparse import load_npz
        logging.info(f"Loading {expression_path}...")
        self.X = load_npz(expression_path)
        self._n_samples = self.X.shape[0]
        logging.info(f"  Loaded: {self._n_samples:,} cells × {self.X.shape[1]:,} proteins, "
                     f"nnz={self.X.nnz:,}")

        # Normalize (rank-based binning per row on sparse data)
        self._normalize_sparse(num_bins)

        # Build packed arrays and save as mmap cache
        self._build_and_save_mmap_cache(cache_path)

    def __getitem__(self, idx):
        """Return per-sample data sliced from mmap'd arrays, converted to torch."""
        start = self._offsets[idx]
        end = self._offsets[idx + 1]
        prot_ids = torch.as_tensor(self._all_prot_ids[start:end].astype(np.int64)).unsqueeze(1)
        bins = torch.as_tensor(self._all_bins[start:end].astype(np.int64))

        if self.include_absent:
            abs_start = self._abs_offsets[idx]
            abs_end = self._abs_offsets[idx + 1]
            absent = torch.as_tensor(self._all_absent_ids[abs_start:abs_end].astype(np.int64))
            return prot_ids, bins, absent
        return prot_ids, bins

    def __len__(self):
        return self._n_samples

    def num_proteins(self):
        return self._n_proteins

    def _normalize_sparse(self, num_bins):
        """Rank-based binning per row, operating on sparse CSR. Stores result as list of (indices, bins)."""
        logging.info(f"Binning {self.X.shape[0]:,} cells into {num_bins} expression bins...")
        self._binned_rows = []

        for i in tqdm(range(self.X.shape[0]), desc="Binning cells"):
            row = self.X.getrow(i)
            if row.nnz == 0:
                self._binned_rows.append((np.array([], dtype=np.int16), np.array([], dtype=np.uint8)))
                continue

            col_indices = row.indices
            values = row.data

            order = np.argsort(values)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(values) + 1)
            bins = np.ceil(ranks / len(values) * num_bins).astype(np.uint8)

            self._binned_rows.append((col_indices.astype(np.int16), bins))

        n_proteins = self.X.shape[1]
        del self.X
        self.X = None

        avg_detected = np.mean([len(r[0]) for r in self._binned_rows])
        logging.info(f"  Avg detected proteins per cell: {avg_detected:.0f}/{n_proteins}")

    def _try_load_mmap_cache(self, cache_path):
        """Attempt to load memory-mapped cache directory. Returns True on success."""
        cache_dir = self._cache_dir(cache_path)
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            return False

        with open(meta_path) as f:
            meta = json.load(f)

        if (meta.get("version") != self._CACHE_VERSION
                or meta.get("n_proteins") != self._n_proteins
                or meta.get("include_absent") != self.include_absent
                or meta.get("num_bins") != self.num_bins):
            logging.info(f"  Cache mismatch at {cache_dir}, recomputing...")
            return False

        self._n_samples = meta["n_samples"]

        # Offsets are small (~32MB for 4M samples) — load into RAM for fast indexing
        self._offsets = np.load(os.path.join(cache_dir, "offsets.npy"))
        # Packed arrays are memory-mapped — only accessed pages enter RAM
        self._all_prot_ids = np.load(os.path.join(cache_dir, "prot_ids.npy"), mmap_mode="r")
        self._all_bins = np.load(os.path.join(cache_dir, "bins.npy"), mmap_mode="r")

        if self.include_absent:
            self._abs_offsets = np.load(os.path.join(cache_dir, "abs_offsets.npy"))
            self._all_absent_ids = np.load(os.path.join(cache_dir, "abs_ids.npy"), mmap_mode="r")

        total = self._offsets[-1]
        logging.info(f"  Loaded mmap cache: {self._n_samples:,} samples, "
                     f"{total:,} entries ({cache_dir})")
        return True

    def _build_and_save_mmap_cache(self, cache_path):
        """Build packed arrays from binned rows and save as memory-mapped .npy files."""
        n_samples = len(self._binned_rows)

        # First pass: compute total sizes for pre-allocation
        det_sizes = np.array([len(r[0]) for r in self._binned_rows], dtype=np.int64)
        offsets = np.zeros(n_samples + 1, dtype=np.int64)
        np.cumsum(det_sizes, out=offsets[1:])
        total_detected = offsets[-1]

        if self.include_absent:
            abs_sizes = np.array([self._n_proteins - s for s in det_sizes], dtype=np.int64)
            abs_offsets = np.zeros(n_samples + 1, dtype=np.int64)
            np.cumsum(abs_sizes, out=abs_offsets[1:])
            total_absent = abs_offsets[-1]

        # Pre-allocate packed arrays with compact dtypes
        all_prot_ids = np.empty(total_detected, dtype=np.int16)
        all_bins = np.empty(total_detected, dtype=np.uint8)
        if self.include_absent:
            all_absent_ids = np.empty(total_absent, dtype=np.int16)

        # Second pass: fill packed arrays
        for idx in tqdm(range(n_samples), desc="Packing SC positions"):
            col_indices, bins = self._binned_rows[idx]
            start = offsets[idx]
            end = offsets[idx + 1]
            all_prot_ids[start:end] = col_indices + 1   # 1-based IDs
            all_bins[start:end] = bins

            if self.include_absent:
                detected = set(col_indices.tolist())
                absent = np.array([p + 1 for p in range(self._n_proteins) if p not in detected], dtype=np.int16)
                abs_start = abs_offsets[idx]
                all_absent_ids[abs_start:abs_start + len(absent)] = absent

        self._binned_rows = None
        logging.info(f"Packed {n_samples:,} samples → {total_detected:,} detected entries")

        # Save to mmap cache directory
        if cache_path:
            cache_dir = self._cache_dir(cache_path)
            os.makedirs(cache_dir, exist_ok=True)

            np.save(os.path.join(cache_dir, "prot_ids.npy"), all_prot_ids)
            np.save(os.path.join(cache_dir, "bins.npy"), all_bins)
            np.save(os.path.join(cache_dir, "offsets.npy"), offsets)
            if self.include_absent:
                np.save(os.path.join(cache_dir, "abs_ids.npy"), all_absent_ids)
                np.save(os.path.join(cache_dir, "abs_offsets.npy"), abs_offsets)

            meta = {
                "version": self._CACHE_VERSION,
                "n_samples": n_samples,
                "n_proteins": self._n_proteins,
                "include_absent": self.include_absent,
                "num_bins": self.num_bins,
            }
            with open(os.path.join(cache_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
            logging.info(f"  Saved mmap cache to {cache_dir}")

            # Re-open as mmap so training uses the memory-mapped path
            self._offsets = offsets
            self._all_prot_ids = np.load(os.path.join(cache_dir, "prot_ids.npy"), mmap_mode="r")
            self._all_bins = np.load(os.path.join(cache_dir, "bins.npy"), mmap_mode="r")
            if self.include_absent:
                self._abs_offsets = abs_offsets
                self._all_absent_ids = np.load(os.path.join(cache_dir, "abs_ids.npy"), mmap_mode="r")
        else:
            # No cache path — keep in RAM (still benefits from compact dtypes)
            self._offsets = offsets
            self._all_prot_ids = all_prot_ids
            self._all_bins = all_bins
            if self.include_absent:
                self._abs_offsets = abs_offsets
                self._all_absent_ids = all_absent_ids

        self._n_samples = n_samples

    @staticmethod
    def _cache_dir(cache_path):
        """Convert legacy .pt cache path to mmap directory path."""
        if cache_path.endswith(".pt"):
            return cache_path[:-3]  # train_cache.pt → train_cache
        return cache_path


class ProteomeCollator:
    """Collate proteome samples into batched tensors for the transformer.

    Receives precomputed (prot_ids, bins) per sample, subsamples into fixed-size
    segments for individuals, protein groups, and absent proteins.

    Sequence layout: [PST, ind_ctx/tgt, PAD..., grp_ctx/tgt, PAD..., abs_ctx/tgt, PAD...]

    Returns dict with tensors:
        role         (B, S):    0=padding, 1=context/PST, 2=target
        protein_ids  (B, S, G): protein IDs per position (G=max_group_size)
        bin_values   (B, S):    expression bins
        group_sizes  (B, S):    members per position (only when protein_groups_enabled)
    """

    def __init__(self, protein_context_ratio: float = 0.85, group_context_ratio: float = 1.0,
                 input_proteins: int = 1024, protein_groups_enabled: bool = False,
                 max_group_size: int = 1, input_groups: int = 0,
                 absent_species_enabled: bool = False, input_absent: int = 0,
                 absent_context_ratio: float = 1.0):
        self.protein_context_ratio = protein_context_ratio
        self.group_context_ratio = group_context_ratio
        self.input_proteins = input_proteins
        self.protein_groups_enabled = protein_groups_enabled
        self.max_group_size = max_group_size if protein_groups_enabled else 1
        self.input_groups = input_groups if protein_groups_enabled else 0
        self.absent_species_enabled = absent_species_enabled
        self.input_absent = input_absent if absent_species_enabled else 0
        self.absent_context_ratio = absent_context_ratio

    def __call__(self, batch):
        n_batches = len(batch)
        G = self.max_group_size
        n_input = 1 + self.input_proteins + self.input_groups + self.input_absent

        role        = torch.zeros(n_batches, n_input, dtype=torch.long)
        protein_ids = torch.zeros(n_batches, n_input, G, dtype=torch.long)
        bin_values  = torch.zeros(n_batches, n_input, dtype=torch.long)
        group_sizes = torch.zeros(n_batches, n_input, dtype=torch.long)

        for i, sample in enumerate(batch):
            # unpack: (exp_prot_ids, exp_bins) or (exp_prot_ids, exp_bins, absent_ids)
            if len(sample) == 3:
                exp_prot_ids, exp_bins, absent_ids = sample
            else:
                exp_prot_ids, exp_bins = sample
                absent_ids = None

            # derive group sizes: count non-zero members per position
            sample_group_sizes = (exp_prot_ids > 0).sum(dim=1)  # (n_positions,)

            # separate individuals (size==1) from groups (size>1)
            is_ind = sample_group_sizes == 1
            is_grp = sample_group_sizes > 1

            ind_ids, ind_bins = exp_prot_ids[is_ind], exp_bins[is_ind]
            grp_ids, grp_bins, grp_sizes_sel = exp_prot_ids[is_grp], exp_bins[is_grp], sample_group_sizes[is_grp]

            n_ind = len(ind_bins)
            n_grp = len(grp_bins)


            # subsample individuals
            if n_ind > self.input_proteins:
                sel = torch.randperm(n_ind)[:self.input_proteins]
                ind_ids, ind_bins = ind_ids[sel], ind_bins[sel]
                n_ind = self.input_proteins

            # subsample groups
            if self.input_groups > 0 and n_grp > self.input_groups:
                sel = torch.randperm(n_grp)[:self.input_groups]
                grp_ids, grp_bins, grp_sizes_sel = grp_ids[sel], grp_bins[sel], grp_sizes_sel[sel]
                n_grp = self.input_groups
            elif self.input_groups == 0:
                n_grp = 0

            if n_ind + n_grp < 2 and (absent_ids is None or self.input_absent == 0):
                continue

            # subsample absent proteins
            n_abs = 0
            if absent_ids is not None and self.input_absent > 0 and len(absent_ids) > 0:
                n_abs = min(self.input_absent, len(absent_ids))
                sel = torch.randperm(len(absent_ids))[:n_abs]
                abs_ids_sampled = absent_ids[sel]

            # PST at position 0
            role[i, 0] = 1


            # individual segment: [1, 1+n_ind) 
            if n_ind > 0:
                perm = torch.randperm(n_ind)
                n_ctx = max(1, int(n_ind * self.protein_context_ratio))
                pos = torch.arange(1, 1 + n_ind)

                role[i, pos[:n_ctx]] = 1
                role[i, pos[n_ctx:]] = 2
                bin_values[i, pos]   = ind_bins[perm]
                group_sizes[i, pos]  = 1
                protein_ids[i, pos]  = ind_ids[perm]

            # group segment: [1+input_proteins, 1+input_proteins+n_grp) 
            if n_grp > 0:
                perm = torch.randperm(n_grp)
                n_ctx = max(1, int(n_grp * self.group_context_ratio))
                offset = 1 + self.input_proteins
                pos = torch.arange(offset, offset + n_grp)

                role[i, pos[:n_ctx]] = 1
                role[i, pos[n_ctx:]] = 2
                bin_values[i, pos]   = grp_bins[perm]
                group_sizes[i, pos]  = grp_sizes_sel[perm]
                protein_ids[i, pos]  = grp_ids[perm]

            # absent segment: [1+input_proteins+input_groups, ...)
            if n_abs > 0:
                perm = torch.randperm(n_abs)
                n_ctx = int(n_abs * self.absent_context_ratio)
                offset = 1 + self.input_proteins + self.input_groups
                pos = torch.arange(offset, offset + n_abs)

                if n_ctx > 0:
                    role[i, pos[:n_ctx]] = 1
                role[i, pos[n_ctx:]] = 2
                protein_ids[i, pos, 0] = abs_ids_sampled[perm]  # already 1-based
                group_sizes[i, pos] = 1

        result = {"role": role, "protein_ids": protein_ids, "bin_values": bin_values}
        if self.protein_groups_enabled:
            result["group_sizes"] = group_sizes
            
        return result
