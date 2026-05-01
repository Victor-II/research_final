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
            ax.plot(epochs, vals, marker="o", label=exp["name"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{scope} {metric} {value}")
    ax.set_title(f"Validation {scope} {metric} {value} over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"val_{scope}_{metric}_{value}.png"
    fig.savefig(path, dpi=150)
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
):
    import matplotlib.pyplot as plt
    import numpy as np

    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    rows = _flatten_results(results)
    rows = _filter_rows(rows, metric=metric, dataset=dataset, scope=scope)
    if not rows:
        print("No test results match the given filters.")
        return

    names = [r["experiment"] for r in rows]
    p = [r["precision"] for r in rows]
    r_ = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - w, p, w, label="Precision")
    ax.bar(x, r_, w, label="Recall")
    ax.bar(x + w, f1, w, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    title_parts = [f"{scope} {metric}"]
    if dataset:
        title_parts.append(f"dataset={dataset}")
    ax.set_title("Test: " + ", ".join(title_parts))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds_label = dataset or "all"
    path = Path(out_dir) / f"test_{scope}_{metric}_{ds_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(
    experiments_dir: str,
    out_dir: str,
    filter_pattern: str = "*",
    experiment_names: list[str] = None,
):
    import matplotlib.pyplot as plt

    results = load_experiment_results(experiments_dir, filter_pattern, experiment_names)
    results = [r for r in results if r.get("train_loss") or r.get("val_loss")]
    if not results:
        print("No experiments with loss history found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for exp in results:
        if exp.get("train_loss"):
            ax1.plot(range(len(exp["train_loss"])), exp["train_loss"], marker="o", label=exp["name"])
        if exp.get("val_loss"):
            ax2.plot(range(len(exp["val_loss"])), exp["val_loss"], marker="o", label=exp["name"])

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Val Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
