"""Convert dense PPI ground-truth matrices to the compact sparse format.

The released matrices are 20,431 x 20,431 float32 arrays holding three values:
NaN (pair not evaluable), 0 (evaluable, no known interaction) and 1 (known
interaction). Only the positives and the covered-protein universe are
irreducible; everything else is implied. Storing those instead is ~36x smaller
and reconstructs the original array bit-for-bit.

    python tools/convert_ground_truth.py            # verify only, write nothing
    python tools/convert_ground_truth.py --write

Never writes to the source directory. Refuses to overwrite an existing output
unless --force is given, because downstream results and figures are pinned to
these exact matrices.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data" / "ppi_ground_truth"
TARGET = REPO / "reference"

FORMAT_VERSION = 1
MAX_INDEX = np.iinfo(np.uint16).max


def encode(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce a dense ternary matrix to its irreducible parts."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got {matrix.shape}")
    if max(matrix.shape) > MAX_INDEX:
        raise ValueError(
            f"vocabulary of {max(matrix.shape)} exceeds uint16; widen the index dtype")

    values = np.unique(matrix[~np.isnan(matrix)])
    unexpected = set(values.tolist()) - {0.0, 1.0}
    if unexpected:
        raise ValueError(f"expected only 0/1/NaN, also found {sorted(unexpected)}")

    nan = np.isnan(matrix)
    universe = np.flatnonzero(~nan.all(axis=1))
    rows, cols = np.nonzero(matrix == 1)

    # The only NaN inside the universe block must be the diagonal (self-pairs).
    block = nan[np.ix_(universe, universe)]
    if int(block.sum()) != universe.size or not bool(np.diag(block).all()):
        raise ValueError("NaN pattern is not 'outside universe, plus diagonal'; "
                         "this encoding would be lossy")

    return {
        "universe": universe.astype(np.uint16),
        "rows": rows.astype(np.uint16),
        "cols": cols.astype(np.uint16),
        "shape": np.asarray(matrix.shape, dtype=np.int64),
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
    }


def decode(parts) -> np.ndarray:
    """Rebuild the dense matrix. Mirrors omicsfm.attention.load_ppi_ground_truth."""
    out = np.full(tuple(parts["shape"]), np.nan, dtype=np.float32)
    universe = parts["universe"].astype(np.int64)
    out[np.ix_(universe, universe)] = 0.0
    out[universe, universe] = np.nan          # self-pairs stay excluded
    out[parts["rows"].astype(np.int64), parts["cols"].astype(np.int64)] = 1.0
    return out


def identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality including NaN placement and dtype."""
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="write the converted files (default: verify only)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing outputs (refused by default)")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()

    if args.target.resolve() == args.source.resolve():
        sys.exit("refusing to write into the source directory")

    sources = sorted(args.source.glob("*_matrix.npz"))
    if not sources:
        sys.exit(f"no *_matrix.npz found in {args.source}")

    if args.write:
        args.target.mkdir(parents=True, exist_ok=True)

    old_total = new_total = 0
    failures = 0
    for path in sources:
        matrix = np.load(path, allow_pickle=True)["matrix"]
        parts = encode(matrix)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **parts)
        blob = buffer.getvalue()

        restored = decode(np.load(io.BytesIO(blob)))
        ok = identical(matrix, restored)
        failures += not ok

        old_total += path.stat().st_size
        new_total += len(blob)
        status = "exact" if ok else "MISMATCH"
        print(f"  {path.name:42} {path.stat().st_size/1024**2:6.1f} -> "
              f"{len(blob)/1024**2:5.2f} MB  {status}")

        if args.write and ok:
            out = args.target / path.name
            if out.exists() and not args.force:
                print(f"     skipped: {out.name} already exists (use --force)")
            else:
                out.write_bytes(blob)
        del matrix, restored

    print(f"\n  {'TOTAL':42} {old_total/1024**2:6.1f} -> {new_total/1024**2:5.2f} MB "
          f"({old_total/max(new_total,1):.0f}x)")
    if failures:
        sys.exit(f"\n{failures} matrices did not round-trip exactly; nothing trusted")
    print("  all matrices round-trip bit-exactly")
    if not args.write:
        print("\n  verification only; pass --write to produce "
              f"{args.target.relative_to(REPO).as_posix()}/")


if __name__ == "__main__":
    main()
