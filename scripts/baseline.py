"""Baseline refusal rate of the base model (should be >= 90% before running the experiment).

    python scripts/baseline.py configs/pilot_qv.yaml

Writes <output_dir>/baseline.json; generations go to baseline_generations.jsonl (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wts.config import load_config  # noqa: E402
from wts.experiment import run_baseline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--n-test", type=int, help="override data.n_test")
    ap.add_argument("--output-dir", help="override output_dir from the config")
    args = ap.parse_args(argv)
    overrides = {}
    if args.n_test:
        overrides["data"] = {"n_test": args.n_test}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    run_baseline(load_config(args.config, overrides or None))


if __name__ == "__main__":
    main()
