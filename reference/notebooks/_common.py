"""Shared setup for the ground-truth builder notebooks.

Import it from any of them:

    from _common import CACHE, FASTA, fetch, require_manual

Everything resolves relative to the installed omicsfm package, so the
notebooks work on any machine and from any working directory. Downloads are
cached under reference/notebooks/_downloads/ and reused on later runs.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from omicsfm.attention import GT_DIR_DEFAULT

# reference/ - where the matrices live and the builders write.
GT_DIR = Path(GT_DIR_DEFAULT)
# Repository root, one level up from reference/.
ROOT = GT_DIR.parent
# Canonical human proteome, tracked in the repository.
FASTA = ROOT / "fasta" / "human_proteome_canonical_31032022.fasta"
# Downloaded source files. Gitignored: they are large and change upstream.
CACHE = Path(__file__).resolve().parent / "_downloads"
CACHE.mkdir(parents=True, exist_ok=True)


def fetch(url: str, name: str | None = None, *, timeout: int = 300,
          attempts: int = 5) -> Path:
    """Download `url` into the cache once and return the local path.

    Re-running a notebook reuses the cached copy rather than hitting the
    source database again. Delete the file to force a refresh.

    Downloads stream to a .part file and are renamed on completion, so an
    interrupted transfer never leaves a truncated file that looks cached.
    Several of these sources drop long connections, hence the retries.
    """
    import time

    import requests

    target = CACHE / (name or url.rsplit("/", 1)[-1].split("?")[0])
    if target.exists():
        print(f"cached: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    tmp = target.with_suffix(target.suffix + ".part")
    print(f"downloading {url}")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, timeout)) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(target)
            last_error = None
            break
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as error:
            last_error = error
            print(f"  attempt {attempt}/{attempts} failed: {error!r}")
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    if last_error is not None:
        raise last_error

    print(f"cached: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def read_maybe_gzip(path: Path) -> bytes:
    data = Path(path).read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def require_manual(name: str, url: str, note: str = "") -> Path:
    """Return a cached file that has to be downloaded by hand.

    Some sources sit behind a licence click-through or a form, so they cannot
    be fetched programmatically. Raises with instructions when absent.
    """
    target = CACHE / name
    if target.exists():
        print(f"found: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    raise FileNotFoundError(
        f"{name} must be downloaded by hand.\n"
        f"  1. get it from: {url}\n"
        f"  2. place it at: {target}\n"
        + (f"  {note}\n" if note else "")
        + "This source is not downloadable programmatically."
    )
