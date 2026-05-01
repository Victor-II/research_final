import argparse
import os
from pathlib import Path

from constants import EXPERIMENTS_DIR
from src.utils.utils import resolve_config, resolve_output_dir


def main():
    parser = argparse.ArgumentParser(description="ABSA experiment runner")
    parser.add_argument("--config", type=str, default=None, help="overlay config yaml")
    parser.add_argument("--set", action="append", default=[], help="dot-notation override, e.g. --set model.learning_rate=5e-4")
    parser.add_argument("--mode", choices=["train", "test", "aggregate", "plot"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path for test mode")
    parser.add_argument("--filter", type=str, default="*", help="glob pattern for aggregate mode")
    parser.add_argument("--experiments", type=str, nargs="+", default=None, help="list of experiment names for aggregate mode")
    parser.add_argument("--group-by", type=str, choices=["experiment", "dataset", "metric", "scope"], default=None, help="group results by column")
    parser.add_argument("--metric", type=str, default=None, help="filter by metric type (micro, soft, macro)")
    parser.add_argument("--dataset", type=str, default=None, help="filter by dataset name")
    parser.add_argument("--scope", type=str, default=None, help="filter by scope (e.g. aspect, polarity)")
    parser.add_argument("--latex", action="store_true", help="output aggregate tables as LaTeX")
    parser.add_argument("--save", action="store_true", help="save aggregate table to aggregated/tables/")
    parser.add_argument("--plot", type=str, nargs="+", choices=["val", "test", "loss"], default=None, help="plot types for plot mode (val, test, loss)")
    parser.add_argument("--plot-dir", type=str, default="aggregated/plots", help="output directory for plots")
    args = parser.parse_args()

    if args.mode == "aggregate":
        from src.eval.aggregate import comparison_table, comparison_latex
        fn = comparison_latex if args.latex else comparison_table
        output = fn(
            str(EXPERIMENTS_DIR),
            filter_pattern=args.filter,
            experiment_names=args.experiments,
            group_by=args.group_by,
            metric=args.metric,
            dataset=args.dataset,
            scope=args.scope,
        )
        print(output)
        if args.save:
            tables_dir = Path("aggregated/tables")
            tables_dir.mkdir(parents=True, exist_ok=True)
            ext = "tex" if args.latex else "txt"
            parts = [p for p in [args.metric, args.dataset, args.scope, args.group_by] if p]
            name = "_".join(parts) if parts else "all"
            path = tables_dir / f"{name}.{ext}"
            path.write_text(output + "\n")
            print(f"Saved: {path}")
        return

    if args.mode == "plot":
        from src.eval.aggregate import plot_val_curves, plot_test_bars, plot_loss_curves
        plot_types = args.plot or ["val", "test", "loss"]
        common = dict(
            experiments_dir=str(EXPERIMENTS_DIR),
            out_dir=args.plot_dir,
            filter_pattern=args.filter,
            experiment_names=args.experiments,
        )
        if "loss" in plot_types:
            plot_loss_curves(**common)
        if "val" in plot_types:
            plot_val_curves(**common, scope=args.scope or "aspect", metric=args.metric or "micro")
        if "test" in plot_types:
            plot_test_bars(**common, scope=args.scope or "aspect", metric=args.metric or "micro", dataset=args.dataset)
        return

    cfg = resolve_config(args.config, args.set)

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "test":
        from src.pipelines.pipeline import test
        ckpt = args.checkpoint or cfg.get("test", {}).get("from_checkpoint")
        if not ckpt:
            raise ValueError("Test mode requires --checkpoint or test.from_checkpoint in config")
        test(cfg, ckpt, output_dir)
    else:
        from src.pipelines.pipeline import run
        run(cfg, output_dir)

    os.environ.pop("_ABSA_OUTPUT_DIR", None)


if __name__ == "__main__":
    main()
