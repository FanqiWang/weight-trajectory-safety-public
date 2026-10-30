"""Compute all metrics, figures and summary tables from an output directory.

    python scripts/analyze.py configs/pilot_qv.yaml
    python scripts/analyze.py configs/smoke.yaml --settings qv

Results go to <output_dir>/analysis/<setting>/: summary.md, *.csv, figures/fig1...fig7.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wts.analysis.pipeline import analyze_setting  # noqa: E402
from wts.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--settings", nargs="*", help="only these settings")
    ap.add_argument("--output-dir", help="override output_dir from the config")
    args = ap.parse_args(argv)
    overrides = {"output_dir": args.output_dir} if args.output_dir else None
    cfg = load_config(args.config, overrides)
    for s in cfg["settings"]:
        if args.settings and s["name"] not in args.settings:
            continue
        print(f"Analysing setting {s['name']}...")
        analyze_setting(cfg, s["name"])


if __name__ == "__main__":
    main()
