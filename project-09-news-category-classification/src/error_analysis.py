"""src/error_analysis.py — Deep misclassification analysis and JSON report."""
import json, logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

def run_error_analysis(texts, y_true, y_pred, y_proba, class_names, output_path, n_examples=5):
    df = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred,
                        "correct": [t==p for t,p in zip(y_true,y_pred)]})
    if y_proba is not None:
        df["confidence"] = y_proba.max(axis=1)
        df["true_prob"]  = [y_proba[i, class_names.index(l)] if l in class_names else 0.
                             for i, l in enumerate(y_true)]
    else:
        df["confidence"] = df["true_prob"] = None

    errors = df[~df["correct"]].copy()
    n_total, n_err = len(df), len(errors)
    logger.info(f"\nError Analysis → {n_err:,} errors / {n_total:,} ({n_err/n_total:.1%})")

    confused = (errors.groupby(["true","pred"]).size()
                .reset_index(name="count").sort_values("count", ascending=False))

    per_class = {}
    for cls in class_names:
        m = df["true"]==cls
        tot, errs = m.sum(), (~df.loc[m,"correct"]).sum()
        per_class[cls] = {"total": int(tot), "errors": int(errs),
                           "error_rate": round(float(errs/tot) if tot else 0, 4)}

    pair_examples = {}
    for _, row in confused.head(6).iterrows():
        key = f"{row['true']} → {row['pred']}"
        sub = errors[(errors["true"]==row["true"])&(errors["pred"]==row["pred"])]
        if y_proba is not None: sub = sub.sort_values("confidence", ascending=False)
        pair_examples[key] = {
            "count": int(row["count"]),
            "examples": [{"text": s["text"][:300],
                          "model_confidence": round(float(s["confidence"]),4) if s["confidence"] is not None else None,
                          "true_class_prob":  round(float(s["true_prob"]),4)  if s["true_prob"]  is not None else None}
                          for s in sub.head(n_examples).to_dict("records")]
        }

    hce = []
    if y_proba is not None:
        hce_df = errors[errors["confidence"]>=0.80].sort_values("confidence", ascending=False)
        hce = [{"text": r["text"][:300], "true": r["true"], "pred": r["pred"],
                "confidence": round(float(r["confidence"]),4)}
               for r in hce_df.head(10).to_dict("records")]
        logger.info(f"High-confidence errors (≥80%): {len(hce_df)}")

    insights = []
    if not confused.empty:
        top = confused.iloc[0]
        insights.append(f"Most confused pair: '{top['true']}' → '{top['pred']}' ({int(top['count'])} cases). "
                        f"Likely share overlapping vocabulary (e.g., financial/geopolitical terminology).")
    if per_class:
        hard = max(per_class.items(), key=lambda x: x[1]["error_rate"])
        insights.append(f"Hardest category: '{hard[0]}' with {hard[1]['error_rate']:.1%} error rate.")
    if hce:
        insights.append(f"{len(hce)} high-confidence errors (≥80%) — model is wrong but certain. "
                        f"A transformer model would resolve these boundary cases.")
    insights.append("Recommendation: TF-IDF n-grams miss semantic context. "
                    "DistilBERT fine-tuning would improve boundary-case accuracy.")

    report = {
        "summary": {"total_samples": n_total, "total_errors": n_err,
                    "error_rate": round(n_err/n_total, 4), "accuracy": round(1-n_err/n_total, 4)},
        "per_class_errors":       per_class,
        "top_confused_pairs":     pair_examples,
        "high_confidence_errors": hce,
        "insights":               insights,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Error analysis → {output_path}")
    return report
