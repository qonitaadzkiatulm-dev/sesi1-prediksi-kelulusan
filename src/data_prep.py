"""Data understanding + data preparation (CRISP-DM tahap 2-3)."""
import numpy as np
import pandas as pd

GRADE_POINT = {"A": 4.0, "AB": 3.5, "B": 3.0, "BC": 2.5, "C": 2.0, "D": 1.0, "E": 0.0}
KEY_COURSES = ["kalkulus", "algoritma", "struktur_data", "basis_data"]

# Kolom bocor: baru diketahui SETELAH prediksi seharusnya dibuat (akhir semester 4).
LEAKAGE_COLS = (["lama_studi_semester", "ipk_akhir", "status_yudisium"]
                + [f"{p}_s{s}" for s in range(5, 9) for p in ("ips", "sks_ambil")])


def quality_report(df: pd.DataFrame) -> dict:
    """Ringkasan masalah kualitas data sebelum dibersihkan."""
    ips_cols = [c for c in df.columns if c.startswith("ips_s")]
    return {
        "jumlah_baris": len(df),
        "duplikat": int(df.duplicated().sum()),
        "missing_per_kolom": df.isna().sum()[lambda s: s > 0].to_dict(),
        "ips_di_luar_0_4": int(((df[ips_cols] < 0) | (df[ips_cols] > 4)).sum().sum()),
        "huruf_mutu_tidak_baku": int(
            sum((df[f"nilai_{c}"].dropna() != df[f"nilai_{c}"].dropna().str.strip().str.upper()).sum()
                for c in KEY_COURSES)),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for c in KEY_COURSES:  # konsistensi kategori (kamus data)
        df[f"nilai_{c}"] = df[f"nilai_{c}"].str.strip().str.upper()
    for c in [c for c in df.columns if c.startswith("ips_s")]:  # aturan bisnis: IPS 0-4
        df.loc[(df[c] < 0) | (df[c] > 4), c] = np.nan
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame, upto_semester: int = 4) -> pd.DataFrame:
    """Fitur point-in-time: hanya data sampai akhir semester `upto_semester`."""
    X = pd.DataFrame(index=df.index)
    sems = range(1, upto_semester + 1)
    for s in sems:
        X[f"ips_s{s}"] = df[f"ips_s{s}"]
    ips = df[[f"ips_s{s}" for s in sems]]
    X["ips_rata2"] = ips.mean(axis=1)
    X["ips_min"] = ips.min(axis=1)
    X["tren_ips"] = df[f"ips_s{upto_semester}"] - df["ips_s1"]
    X["sks_ambil_total"] = df[[f"sks_ambil_s{s}" for s in sems]].sum(axis=1)
    X["sks_lulus_total"] = df[[f"sks_lulus_s{s}" for s in sems]].sum(axis=1)
    X["rasio_sks_lulus"] = X["sks_lulus_total"] / X["sks_ambil_total"]
    for c in KEY_COURSES:
        X[f"nilai_{c}"] = df[f"nilai_{c}"].map(GRADE_POINT)
    for c in ["jml_mk_mengulang", "jml_nilai_d_e", "cuti_semester", "kerja_paruh_waktu"]:
        X[c] = df[c]
    X["jalur_masuk"] = df["jalur_masuk"]
    return X
