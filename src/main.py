"""src/main.py – entry-point with CLI flags (smoke / full)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pprint import pformat

import yaml

# ----------------------------------------------------------------------------------
#  Repository paths & directories
# ----------------------------------------------------------------------------------

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config"
RESEARCH_DIR = REPO / ".research" / "iteration1"
IMAGE_DIR = RESEARCH_DIR / "images"

for d in (CONFIG_DIR, IMAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------
#  Default YAML configurations (auto-written on first run so that external
#  schedulers may overwrite them later).
# ----------------------------------------------------------------------------------

DEFAULT_FULL_CFG = """
experiment: full
seeds: [13, 17, 23]
experiments:
  - name: exp1_joint_budget
    type: joint
    budgets:
      - {mem_mb: 0.25, flops_g: 0.5}
    datasets:
      cifar100_split10:
        url: "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    methods: [tiger_lite, er]
"""

DEFAULT_SMOKE_CFG = """
experiment: smoke
seeds: [0]
experiments:
  - name: smoke_cifar
    type: joint
    budgets:
      - {mem_mb: 1.0, flops_g: 1.0}
    datasets:
      cifar100_split10:
        url: "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    methods: [tiger_lite]
"""

for fn, text in (
    (CONFIG_DIR / "full_experiment.yaml", DEFAULT_FULL_CFG),
    (CONFIG_DIR / "smoke_test.yaml", DEFAULT_SMOKE_CFG),
):
    if not fn.exists():
        fn.write_text(text)

# ----------------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------------

def parse_args():  # noqa: D401
    parser = argparse.ArgumentParser(description="TIGER-Lite experimental runner")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--smoke-test", action="store_true", help="Run quick smoke test only")
    g.add_argument("--full-experiment", action="store_true", help="Run full experiment only")
    return parser.parse_args()


# ----------------------------------------------------------------------------------
#  Main runner
# ----------------------------------------------------------------------------------

def _load_cfg(path: pathlib.Path):
    with open(path) as fp:
        return yaml.safe_load(fp)


def main():  # noqa: D401
    args = parse_args()

    if args.smoke_test:
        cfg_path = CONFIG_DIR / "smoke_test.yaml"
    elif args.full_experiment:
        cfg_path = CONFIG_DIR / "full_experiment.yaml"
    else:
        # two-phase: smoke first then full if successful
        cfg_path = CONFIG_DIR / "smoke_test.yaml"
        print("[INFO] Running smoke test …")
    print(f"[INFO] Using config: {cfg_path}")

    cfg = _load_cfg(cfg_path)

    # Heavy imports delayed until after CLI parsing for snappy `--help`.
    from src.preprocess import assure_datasets, cifar100_split10  # pylint: disable=import-error
    from src.train import run_joint_experiment, run_ablation, set_global_seed  # noqa: E402
    from src.evaluate import summarise_and_plot  # noqa: E402

    # ------------------------------------------------------------------
    all_json_outputs = []
    for exp in cfg["experiments"]:
        set_global_seed(cfg["seeds"][0])
        dataset_paths = assure_datasets(exp["datasets"], REPO / "data")

        if exp["type"] == "joint":
            json_res = run_joint_experiment(
                exp, dataset_paths, cfg["seeds"], RESEARCH_DIR, cifar_split_fn=cifar100_split10
            )
        elif exp["type"] == "ablation":
            json_res = run_ablation(exp, dataset_paths, cfg["seeds"], RESEARCH_DIR)
        elif exp["type"] == "hardware":
            from src.evaluate import run_mcu_demo  # noqa: E402

            json_res = run_mcu_demo(exp, dataset_paths, RESEARCH_DIR)
        else:
            raise ValueError(f"Unknown experiment type: {exp['type']}")

        # Store JSON result file & plot
        all_json_outputs.append(json_res)
        tmp_json_path = RESEARCH_DIR / "tmp_result.json"
        tmp_json_path.write_text(json.dumps(json_res, indent=2))
        summarise_and_plot(tmp_json_path, IMAGE_DIR)

    # ------------------------------------------------------------------
    print("\n======= EXPERIMENT DESCRIPTION =======\n")
    print(pformat(cfg))
    print("\n======= RAW NUMERICAL RESULTS =======\n")
    print(json.dumps(all_json_outputs, indent=2))

    print("\n======= FIGURES GENERATED  =======\n")
    for f in IMAGE_DIR.glob("*.pdf"):
        print(f.name)

    # If we just ran the smoke test and user asked for both phases, chain the full run
    if not args.smoke_test and not args.full_experiment:
        print("\n[INFO] Smoke test passed – launching full experiment …\n")
        # Recursively invoke ourselves with --full-experiment in a new process.
        cmd = [sys.executable, "-m", "src.main", "--full-experiment"]
        os.execv(sys.executable, cmd)  # replace current process


if __name__ == "__main__":  # pragma: no cover
    main()
