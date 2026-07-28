"""Command line entry point: ``omicsfm ...``

Only prefetching lives here. OmicsFM downloads what it needs on first use, so
this is for staging artifacts ahead of time — offline machines, HPC compute
nodes without internet, or simply warming the cache before a long run.
"""

from __future__ import annotations

import argparse
import sys

from omicsfm import hub


def _human(mb: float) -> str:
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def cmd_download(args: argparse.Namespace) -> int:
    tiers = tuple(args.tiers) if args.tiers else ("models",)
    unknown = set(tiers) - set(hub.TIERS)
    if unknown:
        sys.exit(f"unknown tier(s) {sorted(unknown)}; choose from {list(hub.TIERS)}")

    pending = hub.plan(tiers)
    total = sum(a.megabytes for a in pending)

    print(f"OmicsFM artifacts -> {hub.omicsfm_home()}\n")
    print(hub.summarise())
    print()

    if not pending:
        print(f"  nothing to fetch for {', '.join(tiers)}: everything is already local")
        return 0

    print(f"  {len(pending)} file(s) to fetch for {', '.join(tiers)}, {_human(total)}:")
    for a in sorted(pending, key=lambda x: -x.megabytes)[:12]:
        print(f"    {_human(a.megabytes):>9}  {a.key}")
    if len(pending) > 12:
        print(f"    ... and {len(pending)-12} more")

    # Anything substantial needs an explicit yes, so a stray command cannot
    # start a multi-gigabyte transfer.
    if total > 1024 and not args.yes:
        print(f"\n  {_human(total)} is a large download. Re-run with --yes to proceed.")
        return 1
    if args.dry_run:
        print("\n  dry run; omit --dry-run to download")
        return 0

    print()
    for i, a in enumerate(pending, start=1):
        print(f"  [{i}/{len(pending)}] {a.key} ({_human(a.megabytes)})", flush=True)
        try:
            hub.fetch(a)
        except Exception as exc:
            print(f"      failed: {exc}")
            return 1
    print("\n  done")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omicsfm", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="prefetch checkpoints and datasets")
    d.add_argument("tiers", nargs="*", metavar="TIER",
                   help=f"any of {list(hub.TIERS)} (default: models)")
    d.add_argument("--yes", action="store_true", help="confirm a large download")
    d.add_argument("--dry-run", action="store_true", help="show the plan only")
    d.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
