"""Terminal-based annotator for resolving implicit aspects in ASTE data.

Loads LLM suggestions if available, shows them as defaults.
Press Enter to accept suggestion, type to override, 'x' to delete.

Usage:
    python tools/annotate_aste.py <data_file> [--suggestions <suggestions.json>]
"""

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("downloads/annotated")


def parse_aste_file(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("####")
            text = parts[0]
            tokens = text.split()
            triplets = ast.literal_eval(parts[1])
            domain = parts[2].strip() if len(parts) > 2 else None
            annotations = []
            for a_idx, o_idx, pol in triplets:
                implicit_a = (a_idx == [-1] or a_idx == -1)
                implicit_o = (o_idx == [-1] or o_idx == -1)
                annotations.append({
                    "aspect": "IMPLICIT" if implicit_a else " ".join(tokens[j] for j in a_idx),
                    "aspect_idx": None if implicit_a else list(a_idx),
                    "opinion": "IMPLICIT" if implicit_o else " ".join(tokens[j] for j in o_idx),
                    "opinion_idx": None if implicit_o else list(o_idx),
                    "polarity": {"POS": "positive", "NEG": "negative", "NEU": "neutral"}.get(pol, pol),
                })
            examples.append({
                "sentence": text,
                "tokens": tokens,
                "annotations": annotations,
                "domain": domain,
            })
    return examples


def load_suggestions(path: str) -> dict:
    """Load LLM suggestions. Returns {sentence: {ann_index: suggested_aspect}}."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    lookup = {}
    for entry in data:
        sent = entry["sentence"]
        suggestions = {}
        for i, ann in enumerate(entry["annotations"]):
            if "suggested_aspect" in ann:
                suggestions[i] = ann["suggested_aspect"]
        if suggestions:
            lookup[sent] = suggestions
    return lookup


def load_progress(output_path: Path) -> dict:
    if output_path.exists():
        with open(output_path) as f:
            return {ex["sentence"]: ex for ex in json.load(f)}
    return {}


def save_progress(output_path: Path, annotated: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(annotated.values()), f, indent=2)


def print_example(ex: dict, idx: int, total: int, suggestions: dict):
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{total}] {ex['sentence']}")
    print(f"{'='*80}")
    for i, ann in enumerate(ex["annotations"]):
        aspect = ann["aspect"]
        opinion = ann["opinion"]
        polarity = ann["polarity"]
        markers = []
        if aspect == "IMPLICIT":
            sug = suggestions.get(i)
            if sug:
                markers.append(f"💡 suggested: {sug}")
            else:
                markers.append("⚠ NULL aspect")
        if opinion == "IMPLICIT":
            markers.append("⚠ NULL opinion")
        marker_str = f"  ← {', '.join(markers)}" if markers else ""
        print(f"  [{i}] aspect={aspect} | opinion={opinion} | polarity={polarity}{marker_str}")


def edit_example(ex: dict, suggestions: dict) -> bool:
    """Edit implicit annotations. Returns True if changes were made."""
    changed = False
    implicits = [(i, ann) for i, ann in enumerate(ex["annotations"])
                 if ann["aspect"] == "IMPLICIT" or ann["opinion"] == "IMPLICIT"]

    if not implicits:
        return False

    for ann_idx, ann in implicits:
        sug = suggestions.get(ann_idx)
        print(f"\n  Annotation [{ann_idx}]: opinion=\"{ann['opinion']}\" polarity={ann['polarity']}")

        if ann["aspect"] == "IMPLICIT":
            if sug and sug != "UNKNOWN":
                prompt = f"  Aspect [{sug}] (enter=accept, type to override, x=delete, s=skip): "
            else:
                prompt = "  Aspect (type aspect, x=delete, s=skip): "
            resp = input(prompt).strip()

            if resp == "s":
                continue
            elif resp == "x":
                ex["annotations"][ann_idx] = None  # mark for deletion
                changed = True
                print("    Deleted.")
                continue
            elif resp == "":
                if sug and sug != "UNKNOWN":
                    ann["aspect_original"] = "IMPLICIT"
                    ann["aspect"] = sug
                    changed = True
                    print(f"    → {sug}")
                else:
                    print("    Kept as IMPLICIT.")
            else:
                ann["aspect_original"] = "IMPLICIT"
                ann["aspect"] = resp
                changed = True
                print(f"    → {resp}")

        if ann.get("opinion") == "IMPLICIT" and ann.get("aspect") != "IMPLICIT":
            resp = input("  Opinion (type opinion, s=skip): ").strip()
            if resp and resp != "s":
                ann["opinion"] = resp
                changed = True
                print(f"    → {resp}")

    # remove deleted annotations
    ex["annotations"] = [a for a in ex["annotations"] if a is not None]
    return changed


def main():
    parser = argparse.ArgumentParser(description="Annotate implicit aspects in ASTE data")
    parser.add_argument("data_file", help="ASTE-format data file")
    parser.add_argument("--suggestions", default=None, help="LLM suggestions JSON file")
    parser.add_argument("--output", default=None, help="output JSON file")
    args = parser.parse_args()

    data_file = args.data_file
    stem = Path(data_file).stem
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{stem}_annotated.json"

    print("=" * 60)
    print("  ASTE Implicit Aspect Annotator")
    print("=" * 60)

    # Load data
    examples = parse_aste_file(data_file)
    candidates = [ex for ex in examples if any(a["aspect"] == "IMPLICIT" for a in ex["annotations"])]
    print(f"\nLoaded {len(examples)} sentences, {len(candidates)} with implicit aspects")

    # Load suggestions
    suggestions_lookup = {}
    if args.suggestions:
        suggestions_lookup = load_suggestions(args.suggestions)
        n_with_sug = sum(1 for ex in candidates if ex["sentence"] in suggestions_lookup)
        print(f"Loaded suggestions for {n_with_sug}/{len(candidates)} sentences")

    # Load progress
    annotated = load_progress(output_path)
    done = set(annotated.keys())
    remaining = [ex for ex in candidates if ex["sentence"] not in done]
    print(f"Already annotated: {len(done)}, remaining: {len(remaining)}")
    print(f"Output: {output_path}\n")

    if not remaining:
        print("All done!")
        return

    try:
        for idx, ex in enumerate(remaining):
            sug = suggestions_lookup.get(ex["sentence"], {})
            print_example(ex, len(done) + idx, len(candidates), sug)
            changed = edit_example(ex, sug)
            annotated[ex["sentence"]] = ex
            save_progress(output_path, annotated)
            status = "saved" if changed else "skipped"
            print(f"  [{status}] ({len(annotated)}/{len(candidates)} done)")

    except (KeyboardInterrupt, EOFError):
        save_progress(output_path, annotated)
        print(f"\n\nInterrupted. Progress saved ({len(annotated)} sentences).")
        return

    print(f"\nDone! {len(annotated)} sentences annotated → {output_path}")


if __name__ == "__main__":
    main()
