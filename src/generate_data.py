"""Membangkitkan dataset SINTETIS mahasiswa (tidak memakai data pribadi asli).

Tabel mentah sengaja dibuat "berantakan" seperti data dunia nyata:
duplikat, nilai kosong, IPS > 4, huruf mutu tidak konsisten, serta kolom
yang BOCOR (baru diketahui setelah lulus) agar bisa dilatih menanganinya.
Granularitas: 1 baris = 1 mahasiswa.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 1800
GRADE_POINT = {"A": 4.0, "AB": 3.5, "B": 3.0, "BC": 2.5, "C": 2.0, "D": 1.0, "E": 0.0}
LETTERS = list(GRADE_POINT)


def to_letter(score):
    cuts = [3.75, 3.25, 2.75, 2.25, 1.5, 0.5]
    for letter, c in zip(LETTERS, cuts):
        if score >= c:
            return letter
    return "E"


def main():
    ability = RNG.normal(0, 1, N)  # kemampuan laten (tidak ikut dataset)
    angkatan = RNG.choice(range(2015, 2021), N)
    jalur = RNG.choice(["SNBP", "SNBT", "Mandiri"], N, p=[0.3, 0.45, 0.25])
    cuti = (RNG.random(N) < (0.06 + 0.04 * (ability < -0.5))).astype(int)
    kerja = (RNG.random(N) < 0.18).astype(int)

    df = pd.DataFrame({"id_mhs": [f"MHS{i:05d}" for i in range(N)],
                       "angkatan": angkatan, "jalur_masuk": jalur,
                       "cuti_semester": cuti, "kerja_paruh_waktu": kerja})

    drift = RNG.normal(0, 0.15, N)
    for s in range(1, 9):
        ips = 2.9 + 0.45 * ability - 0.25 * cuti - 0.2 * kerja \
              + drift * (s - 1) / 3 + RNG.normal(0, 0.28, N)
        df[f"ips_s{s}"] = np.clip(ips, 0, 4).round(2)
        sks = RNG.choice([18, 20, 21, 22, 24], N, p=[.1, .15, .2, .3, .25])
        if s >= 3:
            sks = np.where(df[f"ips_s{s-1}"] < 2.0, np.minimum(sks, 18), sks)
        df[f"sks_ambil_s{s}"] = sks

    for s in range(1, 5):
        pass_rate = np.clip(0.72 + 0.12 * ability - 0.04 * cuti + RNG.normal(0, .06, N), .3, 1)
        df[f"sks_lulus_s{s}"] = np.round(df[f"sks_ambil_s{s}"] * pass_rate).astype(int)

    for mk, w in {"kalkulus": 0.65, "algoritma": 0.55, "struktur_data": 0.6, "basis_data": 0.5}.items():
        sc = 2.6 + w * ability + RNG.normal(0, .65, N)
        df[f"nilai_{mk}"] = [to_letter(x) for x in np.clip(sc, 0, 4)]

    df["jml_mk_mengulang"] = RNG.poisson(np.exp(0.2 - 0.55 * ability))
    df["jml_nilai_d_e"] = RNG.poisson(np.exp(0.0 - 0.6 * ability))

    z = -1.15 - 1.35 * ability + 0.9 * cuti + 0.25 * df["jml_mk_mengulang"] \
        + 0.35 * kerja + RNG.normal(0, 0.9, N)
    terlambat = (1 / (1 + np.exp(-z)) > RNG.random(N)).astype(int)
    df["lulus_tepat_waktu"] = 1 - terlambat

    # Kolom BOCOR: baru ada setelah mahasiswa lulus / menjelang lulus
    df["lama_studi_semester"] = np.where(terlambat == 1, RNG.integers(9, 14, N), RNG.integers(7, 9, N))
    df["ipk_akhir"] = np.clip(2.9 + 0.4 * ability + RNG.normal(0, .2, N), 1.8, 4).round(2)
    df["status_yudisium"] = np.where(terlambat == 1, "Terlambat", "Tepat Waktu")

    # Kotoran data realistis
    df.loc[RNG.choice(N, 40, replace=False), "ips_s2"] = np.nan
    df.loc[RNG.choice(N, 25, replace=False), "nilai_basis_data"] = np.nan
    bad = RNG.choice(N, 6, replace=False)
    df.loc[bad, "ips_s3"] = df.loc[bad, "ips_s3"] + 4
    messy = RNG.choice(N, 60, replace=False)
    df.loc[messy, "nilai_algoritma"] = df.loc[messy, "nilai_algoritma"].str.lower() + " "
    df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)

    out = Path(__file__).resolve().parent.parent / "data" / "mahasiswa_raw.csv"
    df.to_csv(out, index=False)
    print(f"Tersimpan: {out} | baris={len(df)} | terlambat={(1-df['lulus_tepat_waktu']).mean():.1%}")


if __name__ == "__main__":
    main()
