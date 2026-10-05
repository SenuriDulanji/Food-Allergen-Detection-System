"""
evaluate_dish_identification. 
────────────────────────────────────────────────────────────────────────────
Standalone evaluation script for the Multimodal RAG dish-identification
pipeline (Gemini Vision + Text RAG + CLIP).

Metrics computed
─────────────────
  • Top-1 Accuracy   – correct dish is the #1 prediction
  • Top-3 Accuracy   – correct dish appears in the top-3 candidates
                       (uses text-RAG retrieved candidates)
  • Average Inference Time (seconds per image)
  • Per-Class Accuracy  – breakdown per dish
  • Confusion matrix    – saved as a PNG
  • Full results CSV    – every image with predicted vs ground-truth label

Dataset layout expected
────────────────────────
  dataset/test-images/
      fish_curry/       ← folder name IS the ground-truth label
          img001.jpg
          img002.jpg
          …
      chicken_curry/
          img001.jpg
          …
      …

Usage
──────
  # From the project root with the venv activated:
  python scripts/evaluate_dish_identification.py

  # Limit to N images per class for a quick smoke-test:
  python scripts/evaluate_dish_identification.py --max-per-class 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# ── path bootstrap so we can import from backend/app ──────────────────────── #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR  = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Suppress noisy library logs during evaluation
logging.basicConfig(level=logging.WARNING)

# ── project imports ──────────────────────────────────────────────────────── #
# pyrefly: ignore [missing-import]
from app.services.rag_service import RAGService          # noqa: E402

# ── constants ────────────────────────────────────────────────────────────── #
TEST_IMAGES_DIR = PROJECT_ROOT / "dataset" / "test-images"/"Unseen Data"
RESULTS_DIR     = PROJECT_ROOT / "scripts" / "eval_results"
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}

TOP_K = 3   # used when computing Top-3 accuracy from RAG candidates


# ─────────────────────────────────────────────────────────────────────────── #
#  Helper: normalise a predicted dish name to match ground-truth folder names  #
# ─────────────────────────────────────────────────────────────────────────── #

def _normalise(name: str) -> str:
    """Lower-case, strip, collapse spaces/dashes to underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

KNOWN_CLASSES = {
    "bandakka_curry",
    "batu_moju",
    "beef_curry",
    "beetroot_curry",
    "cashew_curry",
    "chicken_curry",
    "crab_curry",
    "dell_curry",
    "dhal_curry",
    "dry_fish_curry",
    "egg_curry",
    "egg_hoppers",      
    "fish_ambulthiyal",
    "fish_curry",
    "fish_cutlets",
    "gotukola_sambal",
    "green_beans_curry",
    "hot_butter_cuttlefish",
    "isso_vade",
    "karawila_curry",
    "kos_curry",
    "milk_rice",
    "mutton_curry",
    "pittu",
    "plain_hoppers",
    "pol_roti",
    "pol_sambal",
    "polos_curry",       
    "pork_curry",
    "potato_curry",
    "prawn_curry",
    "soya_meat_curry",
    "squid_curry",
    "string_hoppers"
}

def _normalise(name: str) -> str:
    """Lowercase and replace spaces/hyphens with underscores."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


# ─────────────────────────────────────────────────────────────────────────── #
#  Helper: build list of (image_path, ground_truth_label) tuples               #
# ─────────────────────────────────────────────────────────────────────────── #

def collect_test_samples(max_per_class: int | None = None, classes_filter: list[str] | None = None) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []

    if not TEST_IMAGES_DIR.exists():
        print(f"[ERROR] Test images directory not found: {TEST_IMAGES_DIR}")
        sys.exit(1)

    class_dirs = sorted(p for p in TEST_IMAGES_DIR.iterdir() if p.is_dir())

    if not class_dirs:
        print(f"[ERROR] No class subdirectories found inside: {TEST_IMAGES_DIR}")
        sys.exit(1)

    for class_dir in class_dirs:
        if classes_filter and class_dir.name not in classes_filter:
            continue

        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        if max_per_class:
            images = images[:max_per_class]
        for img_path in images:
            samples.append((img_path, class_dir.name))

    return samples


# ─────────────────────────────────────────────────────────────────────────── #
#  Helper: extract top-3 candidates from the RAGService analysis              #
# ─────────────────────────────────────────────────────────────────────────── #

def get_top3_from_rag(rag_service: RAGService, image_bytes: bytes) -> tuple[str, list[str], float]:
    """
    Returns:
        top1     – best dish name predicted
        top3     – list of up to 3 candidate dish names
        elapsed  – wall-clock seconds for the full pipeline call
    """
    start = time.perf_counter()
    result = rag_service.analyze_image(image_bytes)
    elapsed = time.perf_counter() - start

    top1 = _normalise(result.get("dish_name", "unknown"))

    # ── Build Top-3 candidate list ─────────────────────────────────────── #
    # The RAGService returns a single best dish.
    # To get Top-3, we peek into the text RAG retrieval internally.
    # If RAGService is extended to return candidates, use that.
    # For now we construct a best-effort Top-3:
    #   Slot 1  ->  top1  (the final reasoning answer)
    #   Slots 2-3  ->  populated from 'candidates' key if present (future-proof)
    candidates: list[str] = result.get("candidates", [])
    top3 = [top1]
    for c in candidates:
        c_norm = _normalise(c)
        if c_norm not in top3:
            top3.append(c_norm)
    top3 = top3[:TOP_K]   # cap at 3

    return top1, top3, elapsed


# ─────────────────────────────────────────────────────────────────────────── #
#  Main evaluation loop                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def run_evaluation(max_per_class: int | None = None, classes_filter: list[str] | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = RESULTS_DIR / f"results_{timestamp}.csv"
    txt_path  = RESULTS_DIR / f"summary_{timestamp}.txt"

    print("=" * 65)
    print("  Sri Lankan Dish Identification — Evaluation")
    print("  Pipeline: Gemini Vision + Text RAG + CLIP Image RAG")
    print("=" * 65)

    # ── Collect samples ──────────────────────────────────────────────── #
    samples = collect_test_samples(max_per_class, classes_filter)
    classes = sorted({label for _, label in samples})

    print(f"\n  Classes   : {len(classes)}")
    print(f"  Total imgs: {len(samples)}")
    if max_per_class:
        print(f"  (capped at {max_per_class} images per class)")
    print(f"\n  Classes : {classes}\n")
    print("-" * 65)

    # ── Initialise RAGService (once, not per image) ───────────────────── #
    print("  Initialising RAGService …")
    rag_service = RAGService()
    print("  RAGService ready.\n")
    print("-" * 65)

    # ── Tracking counters ─────────────────────────────────────────────── #
    top1_correct  = 0
    top3_correct  = 0
    total         = 0
    total_time    = 0.0
    errors        = 0

    # Per-class: {label: {"total": int, "top1": int, "top3": int}}
    per_class: dict[str, dict] = {c: {"total": 0, "top1": 0, "top3": 0} for c in classes}

    # For confusion matrix
    conf_matrix: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}

    csv_rows: list[dict] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    # ── Iterate over every test image ────────────────────────────────── #
    for idx, (img_path, ground_truth) in enumerate(samples, start=1):
        label_norm = _normalise(ground_truth)

        # ── Rate limit retry wrapper ─────────────────────────────────────── #
        max_retries = 5
        success = False
        top1, top3, elapsed = "unknown", [], 0.0

        for attempt in range(1, max_retries + 1):
            try:
                image_bytes = img_path.read_bytes()
                top1, top3, elapsed = get_top3_from_rag(rag_service, image_bytes)
                success = True
                break
            except Exception as exc:
                exc_str = str(exc)
                if any(x in exc_str for x in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    # Sleep for a full 65 seconds to allow the window to reset or wait out the server overload.
                    sleep_seconds = 65.0
                    print(
                        f"  [API LIMIT/OVERLOAD] Attempt {attempt}/{max_retries} failed for {img_path.name}. "
                        f"Sleeping {sleep_seconds:.1f}s to retry..."
                    )
                    time.sleep(sleep_seconds)
                else:
                    # Non-rate-limit exception
                    print(f"  [{idx:4d}/{len(samples)}]  ERROR  {img_path.name}  ({ground_truth}) — {exc}")
                    break

        if not success:
            errors += 1
            csv_rows.append({
                "index": idx, "image": str(img_path.relative_to(PROJECT_ROOT)),
                "ground_truth": ground_truth, "top1_pred": "ERROR",
                "top3_preds": "", "top1_correct": False,
                "top3_correct": False, "inference_time_s": ""
            })
            continue

        is_top1 = top1 == label_norm
        is_top3 = label_norm in top3

        top1_correct += int(is_top1)
        top3_correct += int(is_top3)
        total        += 1
        total_time   += elapsed

        y_true.append(ground_truth)
        y_pred.append(top1 if top1 in classes else "unknown")

        per_class[ground_truth]["total"] += 1
        if is_top1:
            per_class[ground_truth]["top1"] += 1
        if is_top3:
            per_class[ground_truth]["top3"] += 1

        # Update confusion matrix (only if top1 is a known class)
        if top1 in conf_matrix:
            conf_matrix[ground_truth][top1] += 1

        status = "OK" if is_top1 else ("T3" if is_top3 else "FAIL")
        print(
            f"  [{idx:4d}/{len(samples)}]  {status}  "
            f"{img_path.name:<15}  GT: {ground_truth:<20}  "
            f"Pred: {top1:<20}  ({elapsed:.2f}s)"
        )

        csv_rows.append({
            "index": idx,
            "image": str(img_path.relative_to(PROJECT_ROOT)),
            "ground_truth": ground_truth,
            "top1_pred": top1,
            "top3_preds": "|".join(top3),
            "top1_correct": is_top1,
            "top3_correct": is_top3,
            "inference_time_s": f"{elapsed:.4f}",
        })

        # Sleep briefly between images to avoid triggering the free tier 15/20 RPM rate limit.
        # Since each image makes 2 API calls, a 5.0 second sleep ensures safe pacing.
        time.sleep(5.0)

    # ── Compute overall metrics ───────────────────────────────────────── #
    evaluated = total
    top1_acc  = (top1_correct / evaluated * 100) if evaluated else 0.0
    top3_acc  = (top3_correct / evaluated * 100) if evaluated else 0.0
    avg_time  = (total_time  / evaluated)         if evaluated else 0.0

    # ── Print summary ─────────────────────────────────────────────────── #
    summary_lines = [
        "",
        "=" * 65,
        "  EVALUATION SUMMARY" ,
        "=" * 65,
        f"  Total Images Tested  : {len(samples)}",
        f"  Successfully Eval'd  : {evaluated}",
        f"  Errors / Skipped     : {errors}",
        f"",
        f"  Top-1 Accuracy       : {top1_acc:.2f}%  ({top1_correct}/{evaluated})",
        f"  Top-3 Accuracy       : {top3_acc:.2f}%  ({top3_correct}/{evaluated})",
        f"  Average Inference Time: {avg_time:.2f} seconds",
        f"  Total Inference Time : {total_time:.2f} seconds",
        "",
        "-" * 65,
        "  Per-Class Accuracy",
        "-" * 65,
    ]

    for cls in classes:
        n   = per_class[cls]["total"]
        t1  = per_class[cls]["top1"]
        t3  = per_class[cls]["top3"]
        t1p = (t1 / n * 100) if n else 0.0
        t3p = (t3 / n * 100) if n else 0.0
        summary_lines.append(
            f"  {cls:<25}  Top-1: {t1p:5.1f}%  Top-3: {t3p:5.1f}%  (n={n})"
        )

    summary_lines += [
        "",
        "-" * 65,
        "  Classification Report (Precision, Recall, F1-Score)",
        "-" * 65,
    ]
    
    try:
        from sklearn.metrics import classification_report
        # Calculate precision, recall, f1 for all classes
        report_str = classification_report(y_true, y_pred, labels=classes, zero_division=0)
        summary_lines.append(report_str)
    except ImportError:
        summary_lines.append("  [INFO] scikit-learn not installed — skipping precision/recall/f1 calculation.")

    summary_lines += [
        "",
        "-" * 65,
        "  Confusion Matrix  (rows=Ground Truth, cols=Predicted)",
        "-" * 65,
    ]

    # Header row
    col_w = max(len(c) for c in classes) + 2
    header = "GT \\ Pred".ljust(col_w) + "  ".join(c.ljust(col_w) for c in classes)
    summary_lines.append("  " + header)
    for gt in classes:
        row_vals = "  ".join(str(conf_matrix[gt].get(c, 0)).ljust(col_w) for c in classes)
        summary_lines.append(f"  {gt.ljust(col_w)}{row_vals}")

    summary_lines += [
        "",
        f"  Results CSV : {csv_path}",
        f"  Summary TXT : {txt_path}",
        "=" * 65,
    ]

    output_text = "\n".join(summary_lines)
    print(output_text)

    # ── Save outputs ──────────────────────────────────────────────────── #

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n  Saved CSV     -> {csv_path}")
    print(f"  Saved Summary -> {txt_path}\n")

    # ── Optional: save confusion matrix as PNG ────────────────────────── #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        mat = np.array([[conf_matrix[gt].get(c, 0) for c in classes] for gt in classes])

        fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes))))
        im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([c.replace("_", "\n") for c in classes], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Ground Truth Label", fontsize=11)
        ax.set_title(
            f"Dish ID Confusion Matrix\nTop-1: {top1_acc:.1f}%  |  Top-3: {top3_acc:.1f}%  |  "
            f"Avg Time: {avg_time:.2f}s",
            fontsize=12,
        )

        thresh = mat.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(mat[i, j]),
                        ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black",
                        fontsize=9)

        plt.tight_layout()
        png_path = RESULTS_DIR / f"confusion_matrix_{timestamp}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"  Saved Confusion Matrix -> {png_path}\n")
    except ImportError:
        print("  [INFO] matplotlib not installed — skipping confusion matrix PNG.")


# ─────────────────────────────────────────────────────────────────────────── #
#  Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dish Identification via RAGService.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limit the number of images to evaluate per class (useful for quick tests).")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Evaluate only on these specific class folder names (e.g. --classes koththu egg_hopper)")
    args = parser.parse_args()

    run_evaluation(max_per_class=args.max_per_class, classes_filter=args.classes)
