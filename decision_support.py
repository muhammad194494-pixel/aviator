#!/usr/bin/env python3
"""
decision_support.py — Aturan Main dari Log Prediksi
Author: Malcu v3
Date: 2026-06-15

Cara kerja:
- Membaca prediction_log.txt
- Menghitung akurasi binary (>2x) pada dua kondisi:
  1. Rata‑rata 4 input < 2x
  2. Ada input > 10x
- Mode interaktif: masukkan 4 multiplier, dapat saran Bet/Skip.
"""

import os
import numpy as np

LOG_FILE = "/home/muhammad194494/aviator/prediction_log.txt"
BINARY_THRESHOLD = 2.0

def load_logs():
    if not os.path.exists(LOG_FILE):
        print("❌ File log belum ada.")
        return []

    logs = []
    with open(LOG_FILE) as f:
        for line in f:
            if "Binary:" not in line:
                continue
            try:
                # Parse input
                input_part = line.split("]")[1].split("→")[0].strip()
                inputs = [float(x) for x in input_part.split()]

                # Parse pred & actual
                parts = line.split("→ Pred:")[1]
                pred_str = parts.split("|")[0].replace("x", "").strip()
                actual_str = parts.split("Actual:")[1].split("|")[0].replace("x", "").strip()
                pred = float(pred_str)
                actual = float(actual_str)

                # Binary status
                binary_correct = "BENAR" in line.split("|")[-1].strip()
                logs.append((inputs, pred, actual, binary_correct))
            except:
                continue
    return logs

def analyze(logs):
    total = len(logs)
    if total == 0:
        return

    binary_benar = sum(1 for _, _, _, bc in logs if bc)
    print(f"📊 Total log: {total}")
    print(f"✅ Binary benar: {binary_benar} ({100*binary_benar/total:.1f}%)\n")

    # Kondisi 1: rata‑rata < 2x
    low_avg = [l for l in logs if np.mean(l[0]) < 2.0]
    if low_avg:
        benar = sum(1 for _, _, _, bc in low_avg if bc)
        print(f"🔸 Rata‑rata input < 2x: {len(low_avg)} kejadian")
        print(f"   Binary benar: {benar} ({100*benar/len(low_avg):.0f}%)")
        if benar / len(low_avg) < 0.5:
            print("   ➡️ SARAN: Model sering salah → lebih baik SKIP.\n")
        else:
            print("   ➡️ SARAN: Model cukup akurat → bisa dipertimbangkan.\n")

    # Kondisi 2: ada input > 10x
    high_in = [l for l in logs if any(v > 10 for v in l[0])]
    if high_in:
        benar = sum(1 for _, _, _, bc in high_in if bc)
        print(f"🔸 Ada input > 10x: {len(high_in)} kejadian")
        print(f"   Binary benar: {benar} ({100*benar/len(high_in):.0f}%)")
        if benar / len(high_in) < 0.5:
            print("   ➡️ SARAN: Model sering salah → jangan bet.\n")
        else:
            print("   ➡️ SARAN: Model bisa diandalkan dalam kondisi ini.\n")

def rekomendasi(inputs, logs):
    """Beri saran Bet/Skip berdasarkan aturan yang ada."""
    mean_in = np.mean(inputs)
    has_high = any(v > 10 for v in inputs)

    # Hitung akurasi dari log untuk kondisi yang sesuai
    low_avg_logs = [l for l in logs if np.mean(l[0]) < 2.0]
    high_in_logs = [l for l in logs if any(v > 10 for v in l[0])]

    saran = []
    if mean_in < 2.0:
        benar = sum(1 for _, _, _, bc in low_avg_logs if bc) if low_avg_logs else 0
        total = len(low_avg_logs) if low_avg_logs else 1
        if benar / total < 0.5:
            saran.append("SKIP (rata‑rata input rendah, model sering over‑estimate)")
        else:
            saran.append("BOLEH BET (model cukup akurat di kondisi ini)")
    elif has_high:
        benar = sum(1 for _, _, _, bc in high_in_logs if bc) if high_in_logs else 0
        total = len(high_in_logs) if high_in_logs else 1
        if benar / total < 0.5:
            saran.append("SKIP (ada input >10x, model sering salah)")
        else:
            saran.append("BOLEH BET (model bisa diandalkan)")
    else:
        saran.append("BELUM ADA ATURAN – pantau dulu atau bet kecil")

    return " | ".join(saran)

def main():
    print("="*50)
    print("🧠 DECISION SUPPORT – Aturan dari Log")
    print("="*50)

    logs = load_logs()
    analyze(logs)

    print("─"*50)
    print("Mode interaktif: masukkan 4 multiplier terakhir")
    print("Ketik 'q' untuk keluar\n")

    while True:
        try:
            inp = input("📥 4 multiplier: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if inp.lower() == 'q':
            break
        parts = inp.split()
        if len(parts) != 4:
            print("❌ Harus tepat 4 angka!\n")
            continue
        try:
            inputs = [float(x) for x in parts]
        except ValueError:
            print("❌ Input angka!\n")
            continue

        saran = rekomendasi(inputs, logs)
        print(f"   ➡️ {saran}\n")

if __name__ == "__main__":
    main()
