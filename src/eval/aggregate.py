import json
from pathlib import Path
from collections import defaultdict


def load_experiment_results(experiments_dir: str, filter_pattern: str = "*", experiment_names: list[str] = None) -> list[dict]:
    exp_path = Path(experiments_dir)
    if experiment_names:
        # search both top-level and date subdirectories
        dirs = []
        for name in experiment_names:
            direct = exp_path / name
            if direct.exists():
                dirs.append(direct)
                continue
            # search inside date subdirs
            for sub in sorted(exp_path.iterdir()):
                if sub.is_dir() and (sub / name).exists():
                    dirs.append(sub / name)
    else:
        # collect from top-level and date subdirectories
        dirs = sorted(exp_path.glob(filter_pattern))
        for sub in sorted(exp_path.iterdir()):
            if sub.is_dir() and not (sub / "results").exists():
                dirs.extend(sorted(sub.glob(filter_pattern)))
    results = []
    for exp_dir in dirs:
        results_file = exp_dir / "results" / "results.json"
        config_file = exp_dir / "config.yaml"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        name = exp_dir.name
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            name = cfg.get("name", exp_dir.name)
        results.append({"name": name, "dir": str(exp_dir), **data})
    return results


def _flatten_results(results: list[dict]) -> list[dict]:
    rows = []
    for exp in results:
        name = exp["name"]
        for entry in exp.get("test", []):
            data_path = entry.get("data", "")
            if data_path:
                p = Path(data_path)
                data_label = p.stem if p.stem not in ("test", "train", "dev") else p.parent.name
            else:
                data_label = "?"
            for scope_key, scope_val in entry.items():
                if scope_key in ("data", "epoch") or not isinstance(scope_val, dict):
                    continue
                for metric_type in ("micro", "soft"):
                    if metric_type in scope_val:
                        m = scope_val[metric_type]
                        rows.append({"experiment": name, "dataset": data_label,
                                     "scope": scope_key, "metric": metric_type, **m})
                if "macro" in scope_val:
                    m = scope_val["macro"]["macro"]
                    rows.append({"experiment": name, "dataset": data_label,
                                 "scope": scope_key, "metric": "macro", **m})
    return rows


def _filter_rows(rows: list[dict], metric: str = None, dataset: str = None, scope: str = None) -> list[dict]:
    if metric:
        rows = [r for r in rows if r["metric"] == metric]
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    if scope:
        rows = [r for r in rows if r["scope"] == scope]
    return rows


def _format_table(rows: list[dict], exclude_col: str = None) -> str:
    cols = ["experiment", "dataset", "scope", "metric"]
    if exclude_col:
        cols = [c for c in cols if c != exclude_col]
    widths = {c: max(len(c), max((len(str(r[c])) for r in rows), default=0)) for c in cols}

    header = "  ".join(f"{c:<{widths[c]}}" for c in cols) + f"  {'P':>8}  {'R':>8}  {'F1':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        line = "  ".join(f"{str(r[c]):<{widths[c]}}" for c in cols)
        lines.append(f"{line}  {r['precision']:>8.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}")
    return "\n".join(lines)


def comparison_table(
    experiments_dir: str,
    filter_pattern: str = "*",
    experiment_names: list[str] = None,
    group_by: str = None,
    metric: str = None,
    dataset: str = None,
    scope: str = None,
) -> str:
    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    if not results:
        return "No experiment results found."

    rows = _flatten_results(results)
    rows = _filter_rows(rows, metric=metric, dataset=dataset, scope=scope)
    if not rows:
        return "No results match the given filters."

    if not group_by:
        return _format_table(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[r[group_by]].append(r)

    sections = []
    for key in sorted(groups):
        sections.append(f"=== {key} ===")
        sections.append(_format_table(groups[key], exclude_col=group_by))
        sections.append("")
    return "\n".join(sections).rstrip()



def _format_latex(rows: list[dict], exclude_col: str = None, caption_parts: list[str] = None) -> str:
    cols = ["experiment", "dataset", "scope", "metric"]
    if exclude_col:
        cols = [c for c in cols if c != exclude_col]
    # drop columns where all values are identical (filtered columns)
    cols = [c for c in cols if len(set(r[c] for r in rows)) > 1]
    col_spec = "l" * len(cols) + "rrr"
    headers = " & ".join(cols + ["P", "R", "F1"])

    caption = ", ".join(caption_parts) if caption_parts else None
    lines = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    lines.extend([f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule", f"{headers} \\\\", "\\midrule"])
    for r in rows:
        vals = " & ".join(str(r[c]) for c in cols)
        lines.append(f"{vals} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def comparison_latex(
    experiments_dir: str,
    filter_pattern: str = "*",
    experiment_names: list[str] = None,
    group_by: str = None,
    metric: str = None,
    dataset: str = None,
    scope: str = None,
) -> str:
    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    if not results:
        return "% No experiment results found."

    rows = _flatten_results(results)
    rows = _filter_rows(rows, metric=metric, dataset=dataset, scope=scope)
    if not rows:
        return "% No results match the given filters."

    caption_parts = []
    if metric:
        caption_parts.append(f"metric={metric}")
    if dataset:
        caption_parts.append(f"dataset={dataset}")
    if scope:
        caption_parts.append(f"scope={scope}")

    if not group_by:
        return _format_latex(rows, caption_parts=caption_parts)

    groups = defaultdict(list)
    for r in rows:
        groups[r[group_by]].append(r)

    sections = []
    for key in sorted(groups):
        group_caption = [f"{group_by}={key}"] + caption_parts
        sections.append(_format_latex(groups[key], exclude_col=group_by, caption_parts=group_caption))
        sections.append("")
    return "\n".join(sections).rstrip()



_PLOT_STYLE = {
    "font.sans-serif": ["Noto Sans"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


def _plot_filename(out_dir: str, prefix: str, parts: list[str], experiment_names: list[str] = None) -> Path:
    """Build a unique plot filename from components."""
    name_parts = [prefix] + [p for p in parts if p]
    if experiment_names and len(experiment_names) <= 4:
        name_parts.append("_".join(experiment_names))
    elif experiment_names:
        name_parts.append(f"{len(experiment_names)}exps")
    return Path(out_dir) / ("+".join(name_parts) + ".png")


def plot_val_curves(
    experiments_dir: str,
    out_dir: str,
    filter_pattern: str = "*",
    experiment_names: list[str] = None,
    scope: str = "aspect",
    metric: str = "micro",
    value: str = "f1",
):
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PLOT_STYLE)

    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    results = [r for r in results if r.get("val")]
    if not results:
        print("No experiments with validation history found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for exp in results:
        epochs, vals = [], []
        for entry in exp["val"]:
            scope_val = entry.get(scope)
            if not scope_val or metric not in scope_val:
                continue
            epochs.append(entry["epoch"])
            vals.append(scope_val[metric][value])
        if vals:
            ax.plot(epochs, vals, marker="o", markersize=4, linewidth=1.5, label=exp["name"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{value.upper()}")
    ax.set_title(f"Validation {scope} ({metric} {value})")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = _plot_filename(out_dir, "val", [scope, metric, value], experiment_names)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_test_bars(
    experiments_dir: str,
    out_dir: str,
    filter_pattern: str = "*",
    experiment_names: list[str] = None,
    dataset: str = None,
    scope: str = "aspect",
    metric: str = "micro",
    show: list[str] = None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update(_PLOT_STYLE)

    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    rows = _flatten_results(results)
    rows = _filter_rows(rows, metric=metric, dataset=dataset, scope=scope)
    if not rows:
        print("No test results match the given filters.")
        return

    # which bars to show
    all_bars = ["precision", "recall", "f1"]
    if show:
        bars_to_show = [b for b in all_bars if b in show]
    else:
        bars_to_show = all_bars
    n_bars = len(bars_to_show)

    colors = {"precision": "#4c72b0", "recall": "#55a868", "f1": "#c44e52"}
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1"}

    names = [r["experiment"] for r in rows]
    x = np.arange(len(names))
    w = 0.7 / n_bars

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.8), 5))
    for i, bar_key in enumerate(bars_to_show):
        vals = [r[bar_key] for r in rows]
        offset = (i - (n_bars - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=labels[bar_key], color=colors[bar_key], alpha=0.85)
        if bar_key == bars_to_show[-1]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    max_val = max(r[bars_to_show[-1]] for r in rows)
    ax.set_ylim(0, min(1.0, max_val + 0.08))
    title_parts = [f"{scope} ({metric})"]
    if dataset:
        title_parts.append(dataset)
    ax.set_title(" — ".join(title_parts))
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds_label = dataset or "all"
    bars_label = "-".join(bars_to_show)
    path = _plot_filename(out_dir, "test", [scope, metric, ds_label, bars_label], experiment_names)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(
    experiments_dir: str,
    out_dir: str,
    filter_pattern: str = "*",
    experiment_names: list[str] = None,
    show: list[str] = None,
):
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PLOT_STYLE)

    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    results = [r for r in results if r.get("train_loss") or r.get("val_loss")]
    if not results:
        print("No experiments with loss history found.")
        return

    show = show or ["val"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for exp in results:
        name = exp["name"]
        if "train" in show and exp.get("train_loss"):
            ax.plot(range(len(exp["train_loss"])), exp["train_loss"],
                    linewidth=1.5, linestyle="-", label=f"{name} (train)")
        if "val" in show and exp.get("val_loss"):
            ax.plot(range(len(exp["val_loss"])), exp["val_loss"],
                    linewidth=1.5, linestyle="--", label=f"{name} (val)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss" if len(show) > 1 else f"{'Training' if 'train' in show else 'Validation'} Loss")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = _plot_filename(out_dir, "loss", show, experiment_names)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")



def cross_method_table(
    experiments_dir: str,
    experiment_names: list[str],
    scope: str = "aspect+sentiment+polarity",
    metric: str = "micro",
    latex: bool = False,
) -> str:
    """Generate a cross-method comparison table (experiments as rows, datasets as columns).

    Produces a pivot table with F1 scores: rows = experiments, columns = datasets.
    Highlights the best score per column.
    """
    results = load_experiment_results(experiments_dir, experiment_names=experiment_names)
    if not results:
        return "No experiment results found."

    rows = _flatten_results(results)
    rows = _filter_rows(rows, metric=metric, scope=scope)
    if not rows:
        return "No results match the given filters."

    # pivot: experiment -> dataset -> f1
    pivot = defaultdict(dict)
    datasets_seen = []
    for r in rows:
        pivot[r["experiment"]][r["dataset"]] = r["f1"]
        if r["dataset"] not in datasets_seen:
            datasets_seen.append(r["dataset"])

    # preserve experiment order from input
    exp_order = []
    for name in experiment_names:
        if name in pivot and name not in exp_order:
            exp_order.append(name)
    # add any found but not in input order
    for name in pivot:
        if name not in exp_order:
            exp_order.append(name)

    # find best per dataset
    best_per_ds = {}
    for ds in datasets_seen:
        vals = [(name, pivot[name].get(ds, 0)) for name in exp_order]
        if vals:
            best_per_ds[ds] = max(vals, key=lambda x: x[1])[1]

    if latex:
        return _cross_method_latex(exp_order, datasets_seen, pivot, best_per_ds, scope, metric)
    else:
        return _cross_method_text(exp_order, datasets_seen, pivot, best_per_ds)


def _cross_method_text(exp_order, datasets, pivot, best_per_ds) -> str:
    # column widths
    name_w = max(len(n) for n in exp_order)
    ds_w = max(max(len(d) for d in datasets), 8)

    header = f"{'Method':<{name_w}}" + "".join(f"  {d:>{ds_w+1}}" for d in datasets)
    sep = "-" * len(header)
    lines = [header, sep]

    for name in exp_order:
        parts = [f"{name:<{name_w}}"]
        for ds in datasets:
            val = pivot[name].get(ds)
            if val is None:
                parts.append(f"  {'—':>{ds_w+1}}")
            else:
                marker = "*" if val == best_per_ds.get(ds) else " "
                parts.append(f"  {val:>{ds_w}.4f}{marker}")
        lines.append("".join(parts))

    lines.append(sep)
    lines.append("* = best in column")
    return "\n".join(lines)


def _cross_method_latex(exp_order, datasets, pivot, best_per_ds, scope, metric) -> str:
    n_ds = len(datasets)
    col_spec = "l" + "r" * n_ds
    ds_headers = " & ".join(datasets)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Cross-method comparison ({scope}, {metric} F1)}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Method & {ds_headers} \\\\",
        "\\midrule",
    ]

    for name in exp_order:
        parts = [name.replace("_", "\\_")]
        for ds in datasets:
            val = pivot[name].get(ds)
            if val is None:
                parts.append("—")
            elif val == best_per_ds.get(ds):
                parts.append(f"\\textbf{{{val:.4f}}}")
            else:
                parts.append(f"{val:.4f}")
        lines.append(" & ".join(parts) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)
