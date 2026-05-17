import argparse
import os
from pathlib import Path

from constants import EXPERIMENTS_DIR
from src.utils.utils import resolve_config, resolve_output_dir


def main():
    parser = argparse.ArgumentParser(description="ABSA experiment runner")
    parser.add_argument("--config", type=str, default=None, help="overlay config yaml")
    parser.add_argument("--set", action="append", default=[], help="dot-notation override, e.g. --set model.learning_rate=5e-4")
    parser.add_argument("--mode", choices=["train", "test", "aggregate", "plot", "generate"], default="train")
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
    parser.add_argument("--verbose", action="store_true", help="verbose output for generate mode")
    parser.add_argument("--device", type=str, default="cuda", help="device for generate mode verifier (cuda/cpu)")
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

    if args.mode == "generate":
        from src.augment.verified_paraphrase import generate_verified_paraphrases
        cfg = resolve_config(args.config, args.set) if args.config else {}
        train_files = cfg.get("data", {}).get("train_file", [
            "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/train.txt",
            "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/403.Restaurant15/train.txt",
            "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/404.Restaurant16/train.txt",
        ])
        if isinstance(train_files, str):
            train_files = [train_files]
        ckpt = args.checkpoint or "experiments/2026-05-01/nl-dep-compact/checkpoints/best.ckpt"
        output = cfg.get("generate", {}).get("output", "downloads/paraphrased_verified.json")
        model_name = cfg.get("generate", {}).get("model", "gemma2:27b")
        max_retries = cfg.get("generate", {}).get("max_retries", 3)
        threshold = cfg.get("generate", {}).get("aspect_sim_threshold", 0.5)
        explicit_thresh = cfg.get("generate", {}).get("explicit_threshold", 0.7)
        generate_verified_paraphrases(
            train_files=train_files,
            checkpoint_path=ckpt,
            output_path=output,
            model_name=model_name,
            max_retries=max_retries,
            aspect_sim_threshold=threshold,
            explicit_threshold=explicit_thresh,
            verbose=args.verbose,
            device=args.device,
        )
        return

    cfg = resolve_config(args.config, args.set)

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_type = cfg.get("model", {}).get("type", "seq2seq")

    if args.mode == "test":
        ckpt = args.checkpoint or cfg.get("test", {}).get("from_checkpoint")
        if model_type == "causal-lm":
            # causal LMs don't train — test mode runs inference directly
            from src.pipelines.gemma_pipeline import run_gemma_inference
            run_gemma_inference(cfg, output_dir)
        elif model_type == "span":
            if not ckpt:
                raise ValueError("Test mode requires --checkpoint or test.from_checkpoint in config")
            from src.pipelines.roberta_pipeline import test_roberta
            test_roberta(cfg, ckpt, output_dir)
        else:
            if not ckpt:
                raise ValueError("Test mode requires --checkpoint or test.from_checkpoint in config")
            from src.pipelines.t5_pipeline import test
            test(cfg, ckpt, output_dir)
    else:
        # train mode
        if model_type == "causal-lm":
            from src.pipelines.gemma_pipeline import run_gemma_inference
            run_gemma_inference(cfg, output_dir)
        elif model_type == "span":
            from src.pipelines.roberta_pipeline import run_roberta
            run_roberta(cfg, output_dir)
        else:
            from src.pipelines.t5_pipeline import run
            run(cfg, output_dir)

    os.environ.pop("_ABSA_OUTPUT_DIR", None)


if __name__ == "__main__":
    main()
