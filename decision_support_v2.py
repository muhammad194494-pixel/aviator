#!/usr/bin/env python3
"""
decision_support_v2.py — Dynamic Decision Support System
Author: Malcu v3
Date: 2026-06-18

Upgrade:
- Analisis multi-kondisi otomatis (rata-rata, nilai max, input terakhir, dll)
- Confidence score pada setiap rekomendasi
- Deteksi aturan yang paling bisa diandalkan
"""

import os
import numpy as np

LOG_FILE = "/home/muhammad194494/aviator/prediction_log.txt"  # Sesuaikan path-nya
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
    """Menganalisis log dan mencari semua aturan yang valid."""
    total = len(logs)
    if total == 0:
        return {}

    # Hitung semua metrik yang mungkin relevan
    rules = {}

    # 1. Rata-rata input
    for threshold in [2.0]:
        subset = [l for l in logs if np.mean(l[0]) < threshold]
        if len(subset) >= 5:  # Minimal 5 kejadian biar valid
            benar = sum(1 for _, _, _, bc in subset if bc)
            rules[f"Rata-rata input < {threshold}x"] = {
                "total": len(subset),
                "benar": benar,
                "akurasi": benar / len(subset)
            }

    # 2. Ada input > X
    for threshold in [10.0]:
        subset = [l for l in logs if any(v > threshold for v in l[0])]
        if len(subset) >= 5:
            benar = sum(1 for _, _, _, bc in subset if bc)
            rules[f"Ada input > {threshold}x"] = {
                "total": len(subset),
                "benar": benar,
                "akurasi": benar / len(subset)
            }

    # 3. Input terakhir sangat kecil (< 1.2x)
    subset = [l for l in logs if l[0][-1] < 1.2]
    if len(subset) >= 5:
        benar = sum(1 for _, _, _, bc in subset if bc)
        rules["Input terakhir < 1.2x (sangat kecil)"] = {
            "total": len(subset),
            "benar": benar,
            "akurasi": benar / len(subset)
        }

    # 4. Semua input < 3x (aman, tidak ada yang tinggi)
    subset = [l for l in logs if max(l[0]) < 3.0]
    if len(subset) >= 5:
        benar = sum(1 for _, _, _, bc in subset if bc)
        rules["Semua input < 3x (zona aman)"] = {
            "total": len(subset),
            "benar": benar,
            "akurasi": benar / len(subset)
        }

    # 5. Ada input di atas 50x (ekstrem)
    subset = [l for l in logs if any(v > 50 for v in l[0])]
    if len(subset) >= 5:
        benar = sum(1 for _, _, _, bc in subset if bc)
        rules["Ada input > 50x (sangat ekstrem)"] = {
            "total": len(subset),
            "benar": benar,
            "akurasi": benar / len(subset)
        }

    # 6. Prediksi model < 3.5x (model sedang pesimis)
    subset = [l for l in logs if l[1] < 3.5]
    if len(subset) >= 5:
        benar = sum(1 for _, _, _, bc in subset if bc)
        rules["Prediksi model < 3.5x (pesimis)"] = {
            "total": len(subset),
            "benar": benar,
            "akurasi": benar / len(subset)
        }

    # 7. Prediksi model > 5.0x (model sedang optimis)
    subset = [l for l in logs if l[1] > 5.0]
    if len(subset) >= 5:
        benar = sum(1 for _, _, _, bc in subset if bc)
        rules["Prediksi model > 5.0x (optimis)"] = {
            "total": len(subset),
            "benar": benar,
            "akurasi": benar / len(subset)
        }

    return rules

def rekomendasi(inputs, rules):
    """Memberikan rekomendasi berdasarkan aturan yang paling cocok."""
    hasil = []
    
    # Cek setiap aturan yang berlaku untuk input ini
    for rule_name, stats in rules.items():
        cocok = False
        confidence = stats["akurasi"]
        
        if rule_name.startswith("Rata-rata input <"):
            threshold = float(rule_name.split("<")[1].replace("x", ""))
            if np.mean(inputs) < threshold:
                cocok = True
        elif rule_name.startswith("Ada input >"):
            threshold = float(rule_name.split(">")[1].replace("x", ""))
            if any(v > threshold for v in inputs):
                cocok = True
        elif rule_name == "Input terakhir < 1.2x (sangat kecil)":
            if inputs[-1] < 1.2:
                cocok = True
        elif rule_name == "Semua input < 3x (zona aman)":
            if max(inputs) < 3.0:
                cocok = True
        elif rule_name == "Ada input > 50x (sangat ekstrem)":
            if any(v > 50 for v in inputs):
                cocok = True
        elif rule_name == "Prediksi model < 3.5x (pesimis)":
            # Untuk rules yang butuh prediksi, kita skip dulu
            # Karena dari input saja kita belum tahu prediksinya
            continue
        elif rule_name == "Prediksi model > 5.0x (optimis)":
            continue
            
        if cocok:
            saran = "BET" if confidence >= 0.5 else "SKIP"
            hasil.append({
                "aturan": rule_name,
                "confidence": confidence,
                "saran": saran,
                "total": stats["total"],
                "benar": stats["benar"]
            })
    
    # Urutkan berdasarkan confidence tertinggi
    hasil.sort(key=lambda x: x["confidence"], reverse=True)
    
    if not hasil:
        return "BELUM ADA ATURAN YANG COCOK – tetap waspada"
    
    # Format output yang informatif
    output_lines = []
    for i, h in enumerate(hasil[:3]):  # Maks 3 aturan teratas
        star = "⭐" if h["confidence"] > 0.55 else "📊"
        output_lines.append(
            f"{star} {h['aturan']}:\n"
            f"   Akurasi: {h['benar']}/{h['total']} ({h['confidence']*100:.0f}%)\n"
            f"   Rekomendasi: {h['saran']} (confidence: {h['confidence']*100:.0f}%)"
        )
    
    return "\n\n".join(output_lines)

def main():
    print("="*60)
    print("🧠 DECISION SUPPORT V2 — Dynamic Analyzer")
    print("="*60)

    logs = load_logs()
    rules = analyze(logs)
    
    if not rules:
        print("❌ Belum cukup data untuk membentuk aturan.")
        return
    
    print(f"\n📊 {len(rules)} aturan terdeteksi dari {len(logs)} log:\n")
    for rule_name, stats in sorted(rules.items(), key=lambda x: x[1]["akurasi"], reverse=True):
        bar = "█" * int(stats["akurasi"] * 20)
        print(f"  {stats['akurasi']*100:5.1f}% {bar} {rule_name}")
        print(f"         ({stats['benar']}/{stats['total']} kejadian)\n")

    print("─"*60)
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

        saran = rekomendasi(inputs, rules)
        print(f"\n{saran}\n")

if __name__ == "__main__":
    main()
