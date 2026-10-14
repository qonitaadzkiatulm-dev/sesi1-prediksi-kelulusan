"""Contoh pemakaian model tersimpan untuk mahasiswa baru (data akhir semester 4).

Jalankan:  python src/predict.py data/mahasiswa_raw.csv
Keluaran: skor risiko keterlambatan. Ini BANTUAN keputusan; keputusan
pendampingan tetap diambil dosen wali (human-in-the-loop).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import build_features, clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(csv_path):
    bundle = joblib.load(ROOT / "reports" / "model.joblib")
    df = clean(pd.read_csv(csv_path))
    proba = bundle["pipeline"].predict_proba(build_features(df, 4))[:, 1]
    out = df[["id_mhs"]].assign(skor_risiko=proba.round(3),
                                perlu_pendampingan=(proba >= bundle["threshold"]).astype(int))
    print(out.sort_values("skor_risiko", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "mahasiswa_raw.csv")
