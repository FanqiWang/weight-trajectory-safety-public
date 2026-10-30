"""Run every setting x arm x seed in a config: build data -> train -> evaluate.

    python scripts/run_experiment.py configs/pilot_qv.yaml            # print plan and storage estimate, then confirm
    python scripts/run_experiment.py configs/pilot_qv.yaml --yes      # skip confirmation
    python scripts/run_experiment.py configs/pilot_qv.yaml --dry-run  # plan only
    python scripts/run_experiment.py configs/pilot_qv.yaml --arms C100 --seeds 100   # a subset

Re-running the same command resumes where it stopped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wts.config import load_config  # noqa: E402
from wts.experiment import run_experiment  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--yes", action="store_true", help="start without asking")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan and storage estimate")
    ap.add_argument("--settings", nargs="*", help="only these settings")
    ap.add_argument("--arms", nargs="*", help="only these arms")
    ap.add_argument("--seeds", nargs="*", type=int, help="only these seeds")
    ap.add_argument("--skip-eval", action="store_true", help="train only, no behavioural evaluation")
    ap.add_argument("--output-dir", help="override output_dir from the config")
    args = ap.parse_args(argv)
    overrides = {"output_dir": args.output_dir} if args.output_dir else None
    cfg = load_config(args.config, overrides)
    run_experiment(cfg, yes=args.yes, only_settings=args.settings, only_arms=args.arms, only_seeds=args.seeds,
                   skip_eval=args.skip_eval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
