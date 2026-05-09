"""train.py — End-to-end training pipeline for NewsLens."""
import argparse, json, logging, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import joblib, numpy as np, yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader  import load_from_csv, load_from_huggingface, LABEL_MAP
from src.preprocessor import clean_dataframe
from src.splitter     import split_data
from src.model        import build_logreg_pipeline, build_svm_pipeline, get_top_features
from src.evaluate     import (compute_metrics, save_metrics, plot_confusion_matrix,
                               plot_feature_importance, plot_class_distribution)
from src.error_analysis import run_error_analysis

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")


def load_config(path="config/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   choices=["logreg","svm"], default="logreg")
    p.add_argument("--source",  choices=["csv","huggingface"], default="csv")
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--config",  default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    class_names = list(LABEL_MAP.values())

    logger.info("="*60)
    logger.info("  NewsLens — Training Pipeline")
    logger.info(f"  Model={args.model.upper()} | Source={args.source} | Tune={not args.no_tune}")
    logger.info("="*60)
    t0 = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if args.source == "csv":
        train_df = load_from_csv(cfg["data"]["raw_train_path"])
        test_df  = load_from_csv(cfg["data"]["raw_test_path"])
    else:
        train_df = load_from_huggingface("train")
        test_df  = load_from_huggingface("test")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    train_df = clean_dataframe(train_df, "text")
    test_df  = clean_dataframe(test_df,  "text")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv("data/processed/train_clean.csv", index=False)
    test_df.to_csv( "data/processed/test_clean.csv",  index=False)

    # ── 3. EDA ────────────────────────────────────────────────────────────────
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_class_distribution(train_df["label"].values, class_names,
        "reports/figures/train_class_distribution.png", "AG News — Training Class Distribution")

    # ── 4. Split ──────────────────────────────────────────────────────────────
    train_s, val_s = split_data(train_df, val_size=cfg["data"]["val_size"], random_state=seed)
    X_tr = np.array(train_s["text_clean"].tolist())
    y_tr = np.array(train_s["label"].tolist())
    X_va = np.array(val_s["text_clean"].tolist())
    y_va = np.array(val_s["label"].tolist())
    X_te = np.array(test_df["text_clean"].tolist())
    y_te = np.array(test_df["label"].tolist())
    logger.info(f"Sizes → Train:{len(X_tr):,} | Val:{len(X_va):,} | Test:{len(X_te):,}")

    # ── 5. Build pipelines ────────────────────────────────────────────────────
    lr_pipe  = build_logreg_pipeline(
        max_features=cfg["tfidf"]["max_features"],
        ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        C=cfg["logistic_regression"]["C"],
        solver=cfg["logistic_regression"]["solver"],
        max_iter=cfg["logistic_regression"]["max_iter"],
        random_state=seed)
    svm_pipe = build_svm_pipeline(
        max_features=cfg["tfidf"]["max_features"],
        ngram_range=tuple(cfg["tfidf"]["ngram_range"]),
        C=cfg["linear_svm"]["C"],
        max_iter=cfg["linear_svm"]["max_iter"],
        random_state=seed)

    if not args.no_tune:
        primary = lr_pipe if args.model == "logreg" else svm_pipe
        param_dist = ({"tfidf__max_features":[100000,150000,200000],
                       "tfidf__ngram_range":[(1,2),(1,3)], "tfidf__min_df":[1,2],
                       "clf__C": loguniform(1,20)} if args.model=="logreg"
                      else {"tfidf__max_features":[100000,150000],
                            "tfidf__ngram_range":[(1,2),(1,3)],
                            "clf__estimator__C": loguniform(0.1,10)})
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(primary, param_dist, n_iter=20, scoring="f1_macro",
                                    cv=cv, n_jobs=-1, random_state=seed, verbose=1, refit=True)
        search.fit(X_tr, y_tr)
        logger.info(f"Best CV F1: {search.best_score_:.4f} | params: {search.best_params_}")
        best_pipe = search.best_estimator_
        # Fit the other model
        (svm_pipe if args.model=="logreg" else lr_pipe).fit(X_tr, y_tr)
    else:
        logger.info("Fitting LogReg…")
        lr_pipe.fit(X_tr, y_tr)
        logger.info("Fitting SVM…")
        svm_pipe.fit(X_tr, y_tr)
        best_pipe = lr_pipe if args.model == "logreg" else svm_pipe

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info("\n── Validation ──────────────────────")
    for name, pipe in [("LogReg", lr_pipe), ("LinearSVM", svm_pipe)]:
        try:
            m = compute_metrics(y_va, pipe.predict(X_va), class_names)
            logger.info(f"  {name:10s} → Macro F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f}")
        except: pass

    logger.info("\n── Test ─────────────────────────────")
    for name, pipe in [("LogReg", lr_pipe), ("LinearSVM", svm_pipe)]:
        try:
            m = compute_metrics(y_te, pipe.predict(X_te), class_names)
            logger.info(f"  {name:10s} → Macro F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f}")
        except: pass

    y_pred  = best_pipe.predict(X_te)
    y_proba = best_pipe.predict_proba(X_te)
    metrics = compute_metrics(y_te, y_pred, class_names)

    Path("models").mkdir(exist_ok=True)
    save_metrics(metrics, cfg["evaluation"]["metrics_output"])
    plot_confusion_matrix(y_te, y_pred, class_names, cfg["evaluation"]["confusion_matrix_fig"])
    top_feats = get_top_features(best_pipe, class_names, 20)
    if top_feats:
        plot_feature_importance(top_feats, cfg["evaluation"]["feature_importance_fig"])
    run_error_analysis(list(X_te), list(y_te), list(y_pred), y_proba,
                       class_names, cfg["evaluation"]["error_analysis_output"])

    # ── 7. Save ───────────────────────────────────────────────────────────────
    # CRITICAL: save COMPLETE pipeline — vectoriser + classifier always paired.
    # Prevents the classic dimension-mismatch error when loading in Streamlit.
    joblib.dump(best_pipe, cfg["model"]["output_path"], compress=3)
    joblib.dump(lr_pipe,   "models/best_model.joblib",  compress=3)
    joblib.dump(svm_pipe,  "models/svm_model.joblib",   compress=3)

    meta = {"model_type": args.model, "class_names": class_names,
            "label_map": {str(k): v for k,v in LABEL_MAP.items()},
            "top_features": top_feats,
            "training_samples": int(len(X_tr)), "val_samples": int(len(X_va)),
            "test_samples": int(len(X_te)),
            "metrics": {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}}
    with open("models/model_meta.json","w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time()-t0
    logger.info(f"\n{'='*60}")
    logger.info(f"  ✅  Done in {elapsed/60:.1f} min")
    logger.info(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    logger.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"  Model    → {cfg['model']['output_path']}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
