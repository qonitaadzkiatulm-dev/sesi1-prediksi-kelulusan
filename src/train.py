"""Modeling + evaluation (CRISP-DM tahap 4-5).

Target positif = mahasiswa BERISIKO TERLAMBAT (lulus_tepat_waktu == 0),
karena itulah yang ingin dijaring dosen wali. Recall kelas ini = metrik utama.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score,
                             classification_report, fbeta_score, precision_score, recall_score,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_prep import LEAKAGE_COLS, build_features, clean, quality_report

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TRAIN_UNTIL = 2018  # split berbasis waktu: angkatan <= 2018 latih, >= 2019 uji


def make_preprocessor(X):
    cat = ["jalur_masuk"]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def metrics(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"threshold": round(float(thr), 3),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f2": fbeta_score(y_true, pred, beta=2),
            "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba)}


def best_threshold(y, proba):
    grid = np.linspace(0.1, 0.9, 81)
    return grid[int(np.argmax([fbeta_score(y, proba >= t, beta=2) for t in grid]))]


def main():
    (ROOT / "reports").mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "mahasiswa_raw.csv")
    report = {"kualitas_data_sebelum": quality_report(raw)}
    df = clean(raw)
    report["kualitas_data_sesudah"] = quality_report(df)
    report["kolom_bocor_dibuang"] = [c for c in LEAKAGE_COLS if c in df.columns]

    X = build_features(df, upto_semester=4)
    y = 1 - df["lulus_tepat_waktu"]  # 1 = terlambat (berisiko)
    tr, te = (df["angkatan"] <= TRAIN_UNTIL), (df["angkatan"] > TRAIN_UNTIL)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    report["ukuran"] = {"train": int(tr.sum()), "test": int(te.sum()),
                        "rasio_terlambat_train": round(float(ytr.mean()), 3),
                        "rasio_terlambat_test": round(float(yte.mean()), 3)}

    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("clf", clf)])
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = 0.5 if name.startswith("Baseline") else best_threshold(ytr, oof)  # dipilih TANPA melihat test
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        results[name] = {"cv_roc_auc_train": roc_auc_score(ytr, oof), **metrics(yte, proba, thr)}
        fitted[name] = (pipe, proba, thr)
        print(f"{name:22s} recall={results[name]['recall']:.2f} prec={results[name]['precision']:.2f} "
              f"AUC={results[name]['roc_auc']:.3f} PR-AUC={results[name]['pr_auc']:.3f}")

    cand = {k: v for k, v in results.items() if not k.startswith("Baseline")}
    best = max(cand, key=lambda k: cand[k]["pr_auc"])
    report["model_terbaik"] = best
    report["hasil"] = results
    pipe, proba, thr = fitted[best]
    pred = (proba >= thr).astype(int)
    report["classification_report_terbaik"] = classification_report(
        yte, pred, target_names=["Tepat waktu", "Terlambat"], output_dict=True)
    json.dump(report, open(ROOT / "reports" / "hasil.json", "w"), indent=2, default=float)
    pd.DataFrame(results).T.round(3).to_csv(ROOT / "reports" / "perbandingan_model.csv")

    # Visualisasi
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["Tepat waktu", "Terlambat"],
                                            cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {best}")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (p, pr, _) in fitted.items():
        if not name.startswith("Baseline"):
            RocCurveDisplay.from_predictions(yte, pr, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("Kurva ROC (data uji: angkatan 2019-2020)")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "roc_curve.png", dpi=150); plt.close(fig)

    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__")[1] for n in names]
    imp = (np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_)
    top = pd.Series(imp, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    top.plot.barh(ax=ax, color="#3b6ea5"); ax.set_title(f"Fitur terpenting - {best}")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "feature_importance.png", dpi=150); plt.close(fig)

    joblib.dump({"pipeline": pipe, "threshold": float(thr), "model": best,
                 "fitur": list(X.columns)}, ROOT / "reports" / "model.joblib")

    # Daftar pendampingan untuk dosen wali (data uji)
    out = df.loc[te, ["id_mhs", "angkatan"]].copy()
    out["skor_risiko"] = proba.round(3)
    out["perlu_pendampingan"] = pred
    out.sort_values("skor_risiko", ascending=False).to_csv(
        ROOT / "reports" / "daftar_pendampingan.csv", index=False)
    print(f"\nModel terbaik: {best} (threshold {thr:.2f}). Laporan di folder reports/")


if __name__ == "__main__":
    main()
