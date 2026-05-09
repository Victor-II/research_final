"""Terminal-based annotator for resolving implicit (NULL) aspects and opinions
in SemEval ABSA XML data.

Shows full review context, highlights implicit annotations, lets you fill in
the actual aspect/opinion terms. Saves to a separate JSON file.

Usage:
    python tools/annotate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.data import load_semeval_xml_reviews

DATASETS = {
    "1": ("SemEval 2015 Restaurant Train", "downloads/semeval_xml/dataset/SemEval/15/rest/ABSA-15_Restaurants_Train_Final.xml"),
    "2": ("SemEval 2015 Restaurant Test", "downloads/semeval_xml/dataset/SemEval/15/rest/ABSA15_Restaurants_Test.xml"),
    "3": ("SemEval 2015 Laptop Train", "downloads/semeval_xml/dataset/SemEval/15/laptop/ABSA-15_Laptops_Train_Data.xml"),
    "4": ("SemEval 2015 Laptop Test", "downloads/semeval_xml/dataset/SemEval/15/laptop/ABSA15_Laptops_Test.xml"),
    "5": ("SemEval 2016 Restaurant Train", "downloads/semeval_xml/dataset/SemEval/16/rest/ABSA16_Restaurants_Train_SB1_v2.xml"),
    "6": ("SemEval 2016 Laptop Train", "downloads/semeval_xml/dataset/SemEval/16/laptop/ABSA16_Laptops_Train_SB1_v2.xml"),
}

OUTPUT_DIR = Path("downloads/annotated")


def has_implicit(review: dict) -> bool:
    for sent in review["sentences"]:
        for ann in sent["annotations"]:
            if ann["aspect"] == "IMPLICIT" or ann.get("sentiment") == "IMPLICIT":
                return True
    return False


def load_progress(output_path: Path) -> dict:
    """Load previously annotated reviews. Returns {review_id: review_dict}."""
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        return {r["review_id"]: r for r in data}
    return {}


def save_progress(output_path: Path, annotated: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(list(annotated.values()), f, indent=2)


def fmt_ann(ann: dict, idx: int) -> str:
    aspect = ann["aspect"]
    sentiment = ann.get("sentiment") or "—"
    polarity = ann.get("polarity", "?")
    category = ann.get("category", "?")
    markers = []
    if aspect == "IMPLICIT":
        markers.append("⚠ NULL aspect")
    if sentiment == "IMPLICIT":
        markers.append("⚠ NULL opinion")
    marker_str = f"  ← {', '.join(markers)}" if markers else ""
    return f"    [{idx}] aspect={aspect} | opinion={sentiment} | polarity={polarity} | category={category}{marker_str}"


def print_review(review: dict, review_num: int, total: int):
    print(f"\n{'='*80}")
    print(f"Review {review_num}/{total}  (id: {review['review_id']})")
    print(f"{'='*80}")
    for sent in review["sentences"]:
        has_null = any(
            a["aspect"] == "IMPLICIT" or a.get("sentiment") == "IMPLICIT"
            for a in sent["annotations"]
        )
        prefix = ">>> " if has_null else "    "
        print(f"\n{prefix}{sent['sentence']}")
        for i, ann in enumerate(sent["annotations"]):
            print(fmt_ann(ann, i))


def edit_loop(review: dict) -> bool:
    """Interactive editing for one review. Returns True if any changes were made."""
    changed = False
    while True:
        # collect all editable annotations across sentences
        editable = []
        for sent in review["sentences"]:
            for i, ann in enumerate(sent["annotations"]):
                if ann["aspect"] == "IMPLICIT" or ann.get("sentiment") == "IMPLICIT":
                    editable.append((sent, i, ann))

        if not editable:
            print("\n  No implicit annotations left in this review.")
            break

        print(f"\n  Editable annotations ({len(editable)}):")
        for j, (sent, i, ann) in enumerate(editable):
            sent_preview = sent["sentence"][:60] + ("..." if len(sent["sentence"]) > 60 else "")
            print(f"    {j+1}. [{sent_preview}]")
            print(f"       aspect={ann['aspect']} | opinion={ann.get('sentiment', '—')} | "
                  f"polarity={ann.get('polarity', '?')} | category={ann.get('category', '?')}")

        print("\n  Commands: <number> to edit, [s]kip review, [q]uit")
        cmd = input("  > ").strip().lower()

        if cmd == "s":
            break
        if cmd == "q":
            return changed  # caller checks for quit separately via exception

        try:
            choice = int(cmd) - 1
            if choice < 0 or choice >= len(editable):
                print("  Invalid number.")
                continue
        except ValueError:
            print("  Invalid input.")
            continue

        sent, ann_idx, ann = editable[choice]
        print(f"\n  Editing: {sent['sentence']}")
        print(f"  Current: aspect={ann['aspect']} | opinion={ann.get('sentiment', '—')} | "
              f"polarity={ann.get('polarity', '?')} | category={ann.get('category', '?')}")

        if ann["aspect"] == "IMPLICIT":
            new_aspect = input("  New aspect (enter to keep IMPLICIT, 'x' to delete annotation): ").strip()
            if new_aspect == "x":
                sent["annotations"].pop(ann_idx)
                print("  Annotation deleted.")
                changed = True
                continue
            if new_aspect:
                ann["aspect"] = new_aspect
                ann["aspect_original"] = "IMPLICIT"
                ann["aspect_idx"] = None  # no span indices for inferred aspects
                changed = True
                print(f"  → aspect set to '{new_aspect}'")

        if ann.get("sentiment") == "IMPLICIT" or ann.get("sentiment") is None:
            new_opinion = input("  New opinion (enter to keep as-is): ").strip()
            if new_opinion:
                ann["sentiment"] = new_opinion
                ann["sentiment_idx"] = None
                changed = True
                print(f"  → opinion set to '{new_opinion}'")

    return changed


def main():
    print("=" * 60)
    print("  Implicit Aspect Annotator")
    print("=" * 60)
    print("\nSelect dataset:")
    for key, (name, path) in DATASETS.items():
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  [{key}] {exists} {name}")

    choice = input("\n> ").strip()
    if choice not in DATASETS:
        print("Invalid choice.")
        return

    name, path = DATASETS[choice]
    if not Path(path).exists():
        print(f"File not found: {path}")
        return

    # derive output filename
    safe_name = name.lower().replace(" ", "_")
    output_path = OUTPUT_DIR / f"{safe_name}.json"

    print(f"\nLoading {name}...")
    reviews = load_semeval_xml_reviews(path)
    implicit_reviews = [r for r in reviews if has_implicit(r)]
    print(f"  {len(reviews)} reviews total, {len(implicit_reviews)} with implicit annotations")

    # load previous progress
    annotated = load_progress(output_path)
    done_ids = set(annotated.keys())
    remaining = [r for r in implicit_reviews if r["review_id"] not in done_ids]
    print(f"  {len(done_ids)} already annotated, {len(remaining)} remaining")

    if not remaining:
        print("  All done!")
        return

    print(f"\nOutput: {output_path}")
    print("Starting annotation...\n")

    try:
        for idx, review in enumerate(remaining):
            review_num = len(done_ids) + idx + 1
            total = len(implicit_reviews)

            print_review(review, review_num, total)
            changed = edit_loop(review)

            # always save the review (even if unchanged, marks it as seen)
            annotated[review["review_id"]] = review
            save_progress(output_path, annotated)

            if changed:
                print(f"\n  Saved. ({len(annotated)}/{total} reviews done)")
            else:
                print(f"\n  Skipped. ({len(annotated)}/{total} reviews done)")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\nInterrupted. Progress saved ({len(annotated)} reviews).")
        save_progress(output_path, annotated)
        return

    print(f"\nDone! {len(annotated)} reviews annotated → {output_path}")


if __name__ == "__main__":
    main()
