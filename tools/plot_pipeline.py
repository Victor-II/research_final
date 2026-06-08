"""Generate pipeline flowchart diagrams as PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


# Shared drawing helpers
BLUE = "#3498db"
ORANGE = "#f39c12"
GREEN = "#2ecc71"
PURPLE = "#9b59b6"
RED = "#e74c3c"
DARK = "#2c3e50"
LIGHT_BLUE = "#85c1e9"


def _box(ax, x, y, w, h, text, color=BLUE, fontsize=8, text_color="white"):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor=DARK, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, wrap=True)
    return (x, y, w, h)


def _arrow(ax, x1, y1, x2, y2, color=DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                                color=color, lw=1.3))


def _section_label(ax, x, y, text, fontsize=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=DARK, style="italic")


# ============================================================================
# Diagram 1: Complete Architecture (dissertation)
# ============================================================================
def draw_complete(out_dir):
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # --- DATA SOURCES ---
    _section_label(ax, 7, 15.5, "DATA SOURCES")
    _box(ax, 3, 14.8, 3.2, 0.9, "SemEval ASTE\nRest14/15/16, Laptop14", BLUE)
    _box(ax, 7, 14.8, 3.2, 0.9, "DMASTE\n8 Amazon domains", BLUE)
    _box(ax, 11, 14.8, 3.2, 0.9, "eMAG Romanian\nGPT-4 annotated", PURPLE)

    # --- CANONICAL FORMAT ---
    _box(ax, 7, 13.3, 10, 0.7, "CANONICAL FORMAT: {sentence, tokens, annotations[{aspect, opinion, polarity, category}]}", DARK, fontsize=7.5)
    _arrow(ax, 3, 14.3, 5, 13.7)
    _arrow(ax, 7, 14.3, 7, 13.7)
    _arrow(ax, 11, 14.3, 9, 13.7)

    # --- THREE PATHS ---
    _section_label(ax, 3.5, 12.4, "GENERATIVE PATH")
    _section_label(ax, 7, 12.4, "CLASSIFIER PATH")
    _section_label(ax, 10.5, 12.4, "LLM INFERENCE PATH")

    _arrow(ax, 4, 12.9, 3.5, 12.1)
    _arrow(ax, 7, 12.9, 7, 12.1)
    _arrow(ax, 10, 12.9, 10.5, 12.1)

    # --- GENERATIVE PATH (left) ---
    _box(ax, 3.5, 11.5, 3.8, 0.8, "AUGMENTATION (optional)\nMasking | Duplication | NLPaug\nCross-domain mixing", ORANGE, fontsize=7)
    _arrow(ax, 3.5, 11.1, 3.5, 10.5)

    _box(ax, 3.5, 10.0, 3.8, 0.8, "SYNTAX ENRICHMENT (optional)\nCompact dep. | Compact POS\nInline dep. | Inline POS", ORANGE, fontsize=7)
    _arrow(ax, 3.5, 9.6, 3.5, 9.0)

    _box(ax, 3.5, 8.4, 3.8, 1.0, "FORMAT CONVERSION\nTask subset (curriculum waypoints)\nOutput: Structured | NL | MvP\nMulti-ordering (6 perms)\nSTAR sub-task decomp. (7×)", BLUE, fontsize=6.5)
    _arrow(ax, 3.5, 7.9, 3.5, 7.3)

    _box(ax, 3.5, 7.0, 2.0, 0.5, "PER-EPOCH\nRE-RANDOMISE", LIGHT_BLUE, fontsize=6.5, text_color=DARK)
    _arrow(ax, 3.5, 6.75, 3.5, 6.2)

    _box(ax, 3.5, 5.7, 3.8, 0.9, "FLAN-T5-base (250M)\nTrain: CE + ls=0.05, cosine, 30ep\nInfer: greedy decoding", BLUE, fontsize=7)
    _arrow(ax, 3.5, 5.25, 3.5, 4.6)

    # --- CLASSIFIER PATH (middle) ---
    _box(ax, 7, 11.5, 3.0, 0.7, "TOKENISE + ENCODE", BLUE, fontsize=7.5)
    _arrow(ax, 7, 11.1, 7, 10.5)

    _box(ax, 7, 10.0, 3.0, 1.0, "Models:\nRoBERT-base (114M, RO)\nbert-ro-cased (110M, RO)\nXLM-R-base (278M, multi)\nRoBERTa-base (125M, EN)", PURPLE, fontsize=6.5)
    _arrow(ax, 7, 9.5, 7, 8.9)

    _box(ax, 7, 8.4, 3.0, 0.8, "CLASSIFICATION HEADS\nPolarity head (linear)\nCategory head (linear)", PURPLE, fontsize=7)
    _arrow(ax, 7, 8.0, 7, 4.6)

    # --- LLM PATH (right) ---
    _box(ax, 10.5, 11.5, 3.2, 0.7, "DEMO RETRIEVAL\n3× BM25 + 3× SimCSE", RED, fontsize=7)
    _arrow(ax, 10.5, 11.1, 10.5, 10.5)

    _box(ax, 10.5, 10.0, 3.2, 0.8, "PROMPT CONSTRUCTION\n0-shot (task desc. only)\n6-shot (+ demo examples)", RED, fontsize=7)
    _arrow(ax, 10.5, 9.6, 10.5, 9.0)

    _box(ax, 10.5, 8.4, 3.2, 0.9, "LLM GENERATION (4-bit)\nGemma2 27B\nQwen2.5 32B\nCommand-R 35B", RED, fontsize=7)
    _arrow(ax, 10.5, 7.95, 10.5, 4.6)

    # --- OUTPUT PARSING ---
    _box(ax, 7, 4.2, 11, 0.7, "OUTPUT PARSING: NL regex | Structured brackets | MvP markers | Classifier argmax | LLM regex (malformed → 0)", DARK, fontsize=7)
    _arrow(ax, 7, 3.8, 7, 3.3)

    # --- EVALUATION ---
    _box(ax, 7, 2.5, 11, 1.3,
         "EVALUATION\n"
         "Exact-match micro P/R/F1 (primary) | Macro P/R/F1\n"
         "Lenient F1 (token overlap, τ=0.8) | Lenient F1 (LCS, τ=0.8)\n"
         "Soft F1 (embedding cosine, aspect key) | Avg. overlap + match rate\n"
         "Romanian: Polarity F1, Category F1, Pair F1",
         GREEN, fontsize=6.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=BLUE, edgecolor=DARK, label="Core pipeline"),
        mpatches.Patch(facecolor=ORANGE, edgecolor=DARK, label="Optional / configurable"),
        mpatches.Patch(facecolor=PURPLE, edgecolor=DARK, label="Classifier path"),
        mpatches.Patch(facecolor=RED, edgecolor=DARK, label="LLM inference path"),
        mpatches.Patch(facecolor=GREEN, edgecolor=DARK, label="Evaluation"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=8,
              bbox_to_anchor=(0.5, -0.01))

    ax.set_title("Complete System Architecture", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    path = Path(out_dir) / "pipeline_complete.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Diagram 2: Report 3 (OOD English, no LLMs, no Romanian)
# ============================================================================
def draw_report3(out_dir):
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # --- DATA ---
    _section_label(ax, 5.5, 13.5, "DATA SOURCES")
    _box(ax, 3.5, 12.8, 3.5, 0.8, "SemEval ASTE\nRest14 (train)\nRest14/15/16/Lap14 (test)", BLUE, fontsize=7.5)
    _box(ax, 7.5, 12.8, 3.5, 0.8, "DMASTE\n4 source → 4 target domains", BLUE, fontsize=7.5)

    # --- CANONICAL ---
    _box(ax, 5.5, 11.5, 9, 0.6, "CANONICAL FORMAT: {sentence, tokens, annotations[{aspect, opinion, polarity}]}", DARK, fontsize=7.5)
    _arrow(ax, 3.5, 12.4, 4.5, 11.8)
    _arrow(ax, 7.5, 12.4, 6.5, 11.8)

    # --- AUGMENTATION ---
    _arrow(ax, 5.5, 11.2, 5.5, 10.7)
    _box(ax, 5.5, 10.2, 7, 0.8, "AUGMENTATION (optional)\nAspect masking (configurable %) | Duplication\nCross-domain data mixing (DMASTE fractions)", ORANGE, fontsize=7)

    # --- SYNTAX ---
    _arrow(ax, 5.5, 9.8, 5.5, 9.3)
    _box(ax, 5.5, 8.8, 7, 0.7, "SYNTAX ENRICHMENT (optional)\nCompact dep. | Compact POS | Inline dep. | Inline POS", ORANGE, fontsize=7)

    # --- FORMAT CONVERSION ---
    _arrow(ax, 5.5, 8.45, 5.5, 7.9)
    _box(ax, 5.5, 7.2, 7, 1.2,
         "GENERATIVE FORMAT CONVERSION\n"
         "Task subset: full triplet | singles | pairs\n"
         "  └ Curriculum: waypoint interpolation across epochs\n"
         "Output format: Structured | NL templates | MvP markers\n"
         "Multi-ordering: 6 permutations of (A, O, P)\n"
         "STAR: pairwise sub-task decomposition (7× data)",
         BLUE, fontsize=6.5)

    # --- PER-EPOCH ---
    _arrow(ax, 5.5, 6.6, 5.5, 6.2)
    _box(ax, 5.5, 5.9, 2.2, 0.45, "PER-EPOCH RE-RANDOMISE", LIGHT_BLUE, fontsize=6.5, text_color=DARK)

    # --- MODELS ---
    _arrow(ax, 5.5, 5.65, 5.5, 5.2)
    _box(ax, 3.5, 4.7, 4.5, 0.8, "FLAN-T5-base (250M)\nAdamW, lr=3e-4, cosine, 30ep\nls=0.05, greedy decoding", BLUE, fontsize=7)
    _box(ax, 8, 4.7, 3.5, 0.8, "RoBERTa BASELINE (125M)\nBIO tagger + biaffine\n+ polarity classifier", PURPLE, fontsize=7)

    # --- PARSING ---
    _arrow(ax, 3.5, 4.3, 4.5, 3.7)
    _arrow(ax, 8, 4.3, 6.5, 3.7)
    _box(ax, 5.5, 3.3, 9, 0.6, "OUTPUT PARSING: NL regex | Structured brackets | MvP markers (malformed → 0)", DARK, fontsize=7)

    # --- EVALUATION ---
    _arrow(ax, 5.5, 3.0, 5.5, 2.5)
    _box(ax, 5.5, 1.8, 9, 1.2,
         "EVALUATION\n"
         "Exact-match micro P/R/F1 (primary) | Macro P/R/F1\n"
         "Lenient F1 (token overlap, τ=0.8) | Lenient F1 (LCS, τ=0.8)\n"
         "Soft F1 (embedding cosine, aspect key)\n"
         "Avg. token overlap + LCS overlap + match rate\n"
         "Multi-seed: 5 seeds, mean ± std",
         GREEN, fontsize=6.5)

    legend_elements = [
        mpatches.Patch(facecolor=BLUE, edgecolor=DARK, label="Core pipeline"),
        mpatches.Patch(facecolor=ORANGE, edgecolor=DARK, label="Optional / configurable"),
        mpatches.Patch(facecolor=PURPLE, edgecolor=DARK, label="Discriminative baseline"),
        mpatches.Patch(facecolor=GREEN, edgecolor=DARK, label="Evaluation"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.01))

    ax.set_title("Report 3: Training Strategies Pipeline (English OOD)", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    path = Path(out_dir) / "pipeline_report3.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Diagram 3: Report 4 (Multilingual + LLM)
# ============================================================================
def draw_report4(out_dir):
    fig, ax = plt.subplots(figsize=(13, 14))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # --- DATA ---
    _section_label(ax, 6.5, 13.5, "DATA SOURCES")
    _box(ax, 3.5, 12.8, 3.8, 0.8, "SemEval ASTE (English)\nRest14 train\nRest14/15/16/Lap14 test", BLUE, fontsize=7)
    _box(ax, 9, 12.8, 4.2, 0.8, "eMAG Romanian (CSV)\n27,630 train / 3,093 test\nGPT-4 annotated, 11 categories", PURPLE, fontsize=7)

    # --- CANONICAL ---
    _box(ax, 6.5, 11.5, 11, 0.6, "CANONICAL FORMAT: {sentence, tokens, annotations[{aspect, opinion, polarity, category}]}", DARK, fontsize=7.5)
    _arrow(ax, 3.5, 12.4, 5, 11.8)
    _arrow(ax, 9, 12.4, 8, 11.8)

    # --- THREE PATHS ---
    _arrow(ax, 3, 11.2, 2.5, 10.7)
    _arrow(ax, 6.5, 11.2, 6.5, 10.7)
    _arrow(ax, 10, 11.2, 10.5, 10.7)

    _section_label(ax, 2.5, 10.8, "GENERATIVE")
    _section_label(ax, 6.5, 10.8, "CLASSIFIER")
    _section_label(ax, 10.5, 10.8, "LLM INFERENCE")

    # --- GENERATIVE (left, condensed) ---
    _box(ax, 2.5, 9.8, 3.5, 0.9, "NL FORMAT CONVERSION\nRomanian templates\nPolarity + Category output", BLUE, fontsize=7)
    _arrow(ax, 2.5, 9.35, 2.5, 8.8)
    _box(ax, 2.5, 8.3, 3.5, 0.8, "FLAN-T5-base (250M)\nlr=3e-4, cosine, 12 epochs\nGreedy decoding", BLUE, fontsize=7)
    _arrow(ax, 2.5, 7.9, 2.5, 4.2)

    # --- CLASSIFIER (middle) ---
    _box(ax, 6.5, 9.8, 3.8, 0.7, "TOKENISE + ENCODE", PURPLE, fontsize=7.5)
    _arrow(ax, 6.5, 9.45, 6.5, 8.9)
    _box(ax, 6.5, 8.3, 3.8, 1.0,
         "Models (all same architecture):\n"
         "RoBERT-base (114M, Romanian)\n"
         "bert-ro-cased (110M, Romanian)\n"
         "XLM-R-base (278M, multilingual)",
         PURPLE, fontsize=6.5)
    _arrow(ax, 6.5, 7.8, 6.5, 7.3)
    _box(ax, 6.5, 6.8, 3.8, 0.8, "CLASSIFICATION HEADS\nPolarity (linear) + Category (linear)\nlr=2e-5, 8 epochs, early stop", PURPLE, fontsize=7)
    _arrow(ax, 6.5, 6.4, 6.5, 4.2)

    # --- LLM (right) ---
    _box(ax, 10.5, 9.8, 3.2, 0.7, "DEMO RETRIEVAL\n3× BM25 + 3× SimCSE", RED, fontsize=7)
    _arrow(ax, 10.5, 9.45, 10.5, 8.9)
    _box(ax, 10.5, 8.3, 3.2, 0.8, "PROMPT CONSTRUCTION\n0-shot: task desc. + taxonomy\n6-shot: + 6 demo examples", RED, fontsize=7)
    _arrow(ax, 10.5, 7.9, 10.5, 7.3)
    _box(ax, 10.5, 6.8, 3.2, 0.9, "LLM GENERATION (4-bit)\nGemma2 27B\nQwen2.5 32B\nCommand-R 35B", RED, fontsize=7)
    _arrow(ax, 10.5, 6.35, 10.5, 4.2)

    # --- PARSING ---
    _box(ax, 6.5, 3.8, 11, 0.6, "OUTPUT PARSING: NL regex | Classifier argmax | LLM regex (malformed → 0)", DARK, fontsize=7.5)
    _arrow(ax, 6.5, 3.5, 6.5, 3.0)

    # --- EVALUATION ---
    _box(ax, 6.5, 2.2, 11, 1.3,
         "EVALUATION\n"
         "English ASTE: Exact-match triplet micro-F1 | Lenient F1 | Soft F1 | Avg. overlap\n"
         "Romanian: Polarity + Category pair F1 | Polarity F1 | Category F1\n"
         "Cross-method: fine-tuned (110-278M) vs LLM (27-35B) comparison",
         GREEN, fontsize=7)

    legend_elements = [
        mpatches.Patch(facecolor=BLUE, edgecolor=DARK, label="Generative (FLAN-T5)"),
        mpatches.Patch(facecolor=PURPLE, edgecolor=DARK, label="Classifier (BERT variants)"),
        mpatches.Patch(facecolor=RED, edgecolor=DARK, label="LLM inference"),
        mpatches.Patch(facecolor=GREEN, edgecolor=DARK, label="Evaluation"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.01))

    ax.set_title("Report 4: Multilingual and LLM Pipeline", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    path = Path(out_dir) / "pipeline_report4.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dissertation/figures/")
    parser.add_argument("--which", choices=["all", "complete", "r3", "r4"], default="all")
    args = parser.parse_args()

    if args.which in ("all", "complete"):
        draw_complete(args.output)
    if args.which in ("all", "r3"):
        draw_report3(args.output)
    if args.which in ("all", "r4"):
        draw_report4(args.output)
    print("\nDone.")
