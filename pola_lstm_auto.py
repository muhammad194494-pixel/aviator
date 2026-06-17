#!/usr/bin/env python3
"""
pola_lstm_auto.py — Aviator LSTM + Binary Decision + Auto-Loop
Author: Malcu v3
Date: 2026-06-18

Fungsi:
- Training session‑aware dengan data asli (tanpa augmentasi).
- Mode auto‑loop: membaca aviator_data_live.json, memprediksi 4 multiplier,
  mencatat log binary (>2x), otomatis.
- Menu interaktif (prediksi manual, statistik, retrain, analisis error).
"""

import json, os, glob, re, time
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ========== KONFIGURASI ==========
DATA_DIR = "/home/muhammad194494/aviator"
MODEL_PATH = os.path.join(DATA_DIR, "pola_lstm_auto.keras")
META_PATH = os.path.join(DATA_DIR, "pola_lstm_auto_meta.json")
LOG_FILE = os.path.join(DATA_DIR, "prediction_log.txt")
LIVE_FILE = os.path.join(DATA_DIR, "aviator_data_live.json")

SEQ_LEN = 4
EPOCHS = 200
BATCH = 32
LR = 0.001
SPLIT = 0.8
BINARY_THRESHOLD = 2.0
# =================================


def log_transform(x): return np.log1p(x)
def inv_log(x): return np.expm1(x)


# ──────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────
def load_all_sessions(data_dir):
    files = glob.glob(os.path.join(data_dir, "aviator_data*.json"))
    # sort numerik, file tanpa angka dianggap terakhir (live)
    def sort_key(fname):
        base = os.path.basename(fname)
        m = re.match(r'aviator_data(\d+)\.json', base)
        if m:
            return (0, int(m.group(1)))
        # aviator_data.json -> 999, aviator_data_live.json -> 1000
        if base == "aviator_data.json":
            return (1, 999)
        return (2, 1000)
    files = sorted(files, key=sort_key)
    sessions = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            if len(data) == 0:
                continue
            sessions.append(np.array(data, dtype=np.float32))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   ⚠️ Gagal load {os.path.basename(f)}: {e} — DILEWATI")
            continue
    return sessions


# ──────────────────────────────────────────
#  MODEL (ARSITEKTUR LEBIH KECIL)
# ──────────────────────────────────────────
def build_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(LR), loss='mse', metrics=['mae'])
    return model


def train_model(sessions):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    X_all, y_all = [], []
    for sesi in sessions:
        if len(sesi) <= SEQ_LEN:
            continue
        sesi_log = log_transform(sesi)
        dmin, dmax = sesi_log.min(), sesi_log.max()
        sesi_norm = (sesi_log - dmin) / (dmax - dmin + 1e-8)
        X, y = [], []
        for i in range(len(sesi_norm) - SEQ_LEN):
            X.append(sesi_norm[i:i+SEQ_LEN])
            y.append(sesi_norm[i+SEQ_LEN])
        X = np.array(X).reshape(-1, SEQ_LEN, 1)
        X_all.append(X)
        y_all.append(np.array(y))

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    split = int(len(X_all) * SPLIT)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    model = build_model((SEQ_LEN, 1))
    es = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    print(f"🏋️ Training ({len(X_all)} sequences, {EPOCHS} epochs)...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(X_val, y_val), callbacks=[es, rlp], verbose=1)

    model.save(MODEL_PATH)
    all_data = np.concatenate(sessions)
    all_log = log_transform(all_data)
    with open(META_PATH, 'w') as f:
        json.dump({"min": float(all_log.min()), "max": float(all_log.max())}, f)
    print(f"💾 Model disimpan: {MODEL_PATH}")
    return model


def load_model():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta["min"], meta["max"]


def predict(model, inputs, dmin, dmax):
    arr = np.array(inputs, dtype=np.float32)
    arr_log = log_transform(arr)
    arr_norm = (arr_log - dmin) / (dmax - dmin + 1e-8)
    X = arr_norm.reshape(1, SEQ_LEN, 1)
    pred_norm = model.predict(X, verbose=0)[0][0]
    pred_log = pred_norm * (dmax - dmin) + dmin
    return max(inv_log(pred_log), 1.0)


def classify_zone(val):
    if val < 1.5: return "🟢 RENDAH"
    elif val < 5.0: return "🟡 SEDANG"
    elif val < 15.0: return "🟠 TINGGI"
    else: return "🔴 EKSTREM"


# ──────────────────────────────────────────
#  MENU 1 – PREDIKSI MANUAL (dipakai jika tidak auto)
# ──────────────────────────────────────────
def menu_prediksi(model, dmin, dmax):
    print("\n" + "─"*50)
    print("🧪 MODE PREDIKSI (Binary > 2x)")
    print("   Masukkan 4 multiplier terakhir (spasi)")
    print("   Ketik 'q' untuk kembali ke menu\n")
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

        pred = predict(model, inputs, dmin, dmax)
        zone = classify_zone(pred)
        binary_pred = "YA" if pred > BINARY_THRESHOLD else "TIDAK"
        print(f"""
   ┌──────────────────────────────────────┐
   │ Input : {inputs[0]:.2f}  {inputs[1]:.2f}  {inputs[2]:.2f}  {inputs[3]:.2f}          │
   │ Pred  : {pred:.2f}x  ({zone}) │
   │ Bet 2x? {binary_pred}                        │
   └──────────────────────────────────────┘
""")
        try:
            actual_inp = input("📊 Hasil aktual di game (enter untuk skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if actual_inp == '':
            continue
        try:
            actual = float(actual_inp)
        except ValueError:
            print("❌ Input tidak valid, lanjut...\n")
            continue

        actual_binary = "YA" if actual > BINARY_THRESHOLD else "TIDAK"
        binary_correct = (binary_pred == actual_binary)
        log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " \
                    f"{inputs[0]} {inputs[1]} {inputs[2]} {inputs[3]} → Pred:{pred:.2f}x | Actual:{actual:.2f}x | " \
                    f"Binary: {binary_pred} vs {actual_binary} | {'BENAR' if binary_correct else 'SALAH'}"
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(log_entry + "\n")
            print(f"   💾 {log_entry}")
        except Exception as e:
            print(f"   ⚠️ Gagal menulis log: {e}")
        print()


# ──────────────────────────────────────────
#  MENU 2 – STATISTIK DATA
# ──────────────────────────────────────────
def menu_statistik():
    print("\n" + "─"*50)
    print("📊 STATISTIK SEMUA DATA")
    sessions = load_all_sessions(DATA_DIR)
    all_data = np.concatenate(sessions)

    print(f"\n   Total sesi     : {len(sessions)}")
    print(f"   Total rounds   : {len(all_data)}")
    print(f"   Min: {all_data.min():.2f}x | Max: {all_data.max():.2f}x")
    print(f"   Mean: {all_data.mean():.2f}x | Median: {np.median(all_data):.2f}x")
    print(f"   Std Dev: {all_data.std():.2f}x")

    rendah = np.sum(all_data < 1.5)
    sedang = np.sum((all_data >= 1.5) & (all_data < 5.0))
    tinggi = np.sum((all_data >= 5.0) & (all_data < 15.0))
    ekstrem = np.sum(all_data >= 15.0)
    n = len(all_data)
    print(f"\n   🟢 Rendah (<1.5x)  : {rendah:>5} ({100*rendah/n:5.1f}%)")
    print(f"   🟡 Sedang (1.5-5x) : {sedang:>5} ({100*sedang/n:5.1f}%)")
    print(f"   🟠 Tinggi (5-15x)  : {tinggi:>5} ({100*tinggi/n:5.1f}%)")
    print(f"   🔴 Ekstrem (>15x)  : {ekstrem:>5} ({100*ekstrem/n:5.1f}%)")

    print(f"\n   📁 DATA PER SESI:")
    for i, sesi in enumerate(sessions):
        z = [classify_zone(v) for v in sesi]
        print(f"   Sesi {i+1:2d}: {len(sesi):3d} rounds | "
              f"🟢{z.count('🟢 RENDAH')} "
              f"🟡{z.count('🟡 SEDANG')} "
              f"🟠{z.count('🟠 TINGGI')} "
              f"🔴{z.count('🔴 EKSTREM')}")
    print()


# ──────────────────────────────────────────
#  MENU 3 – RETRAIN
# ──────────────────────────────────────────
def menu_retrain():
    print("\n" + "─"*50)
    print("🔄 RETRAIN FULL MODEL")
    confirm = input("   Yakin? (y/n): ").strip().lower()
    if confirm != 'y':
        print("   ❌ Dibatalkan.\n")
        return None, None, None
    sessions = load_all_sessions(DATA_DIR)
    print(f"   📂 {len(sessions)} sesi, {sum(len(s) for s in sessions)} rounds")
    train_model(sessions)
    model, dmin, dmax = load_model()
    print("   ✅ Retrain selesai!\n")
    return model, dmin, dmax


# ──────────────────────────────────────────
#  MENU 4 – ANALISIS ERROR
# ──────────────────────────────────────────
def menu_analisis_error():
    print("\n" + "─"*50)
    print("📊 ANALISIS POLA ERROR + BINARY (>2x)")
    if not os.path.exists(LOG_FILE):
        print("   ❌ Belum ada file log. Jalankan prediksi dulu.\n")
        return

    with open(LOG_FILE) as f:
        lines = f.readlines()

    total = 0
    over = 0
    under = 0
    benar = 0
    binary_benar = 0
    detail = []

    for line in lines:
        try:
            if "Binary:" not in line:
                continue
            parts = line.split("→ Pred:")[1]
            pred_str = parts.split("|")[0].replace("x", "").strip()
            actual_str = parts.split("Actual:")[1].split("|")[0].replace("x", "").strip()
            pred = float(pred_str)
            actual = float(actual_str)
            
            binary_part = line.split("Binary:")[1].split("|")[0].strip()
            binary_correct = "BENAR" in line.split("|")[-1].strip()
            
            total += 1
            if pred > actual:
                over += 1
            elif pred < actual:
                under += 1
            else:
                benar += 1
            if binary_correct:
                binary_benar += 1

            input_part = line.split("]")[1].split("→")[0].strip()
            inputs = [float(x) for x in input_part.split()]
            if len(inputs) == 4:
                detail.append((inputs, pred, actual, binary_correct))
        except:
            continue

    if total == 0:
        print("   ❌ Tidak ada log binary yang bisa dianalisis.\n")
        return

    print(f"\n   Total log     : {total}")
    print(f"   Binary benar  : {binary_benar} ({100*binary_benar/total:.1f}%)")
    print(f"   Over‑estimate : {over} ({100*over/total:.1f}%)")
    print(f"   Under‑estimate: {under} ({100*under/total:.1f}%)")

    low_avg_total = 0
    low_avg_binary_benar = 0
    for inp, pred, actual, binary_correct in detail:
        if np.mean(inp) < 2.0:
            low_avg_total += 1
            if binary_correct:
                low_avg_binary_benar += 1
    if low_avg_total > 0:
        print(f"\n   🔸 Jika rata‑rata input < 2x ({low_avg_total} kejadian):")
        print(f"      Binary benar: {low_avg_binary_benar} ({100*low_avg_binary_benar/low_avg_total:.0f}%)")
        if low_avg_binary_benar / low_avg_total > 0.6:
            print("      ➡️ Saran: Model cukup akurat → bisa ikuti rekomendasi.")
        else:
            print("      ➡️ Saran: Akurasi rendah → lebih baik skip.")

    high_input_total = 0
    high_input_binary_benar = 0
    for inp, pred, actual, binary_correct in detail:
        if any(v > 10 for v in inp):
            high_input_total += 1
            if binary_correct:
                high_input_binary_benar += 1
    if high_input_total > 0:
        print(f"\n   🔸 Jika ada input > 10x ({high_input_total} kejadian):")
        print(f"      Binary benar: {high_input_binary_benar} ({100*high_input_binary_benar/high_input_total:.0f}%)")
        if high_input_binary_benar / high_input_total > 0.6:
            print("      ➡️ Saran: Model bisa diandalkan dalam kondisi ini.")
        else:
            print("      ➡️ Saran: Model sering salah → jangan bet.")

    print()


# ──────────────────────────────────────────
#  MODE AUTO‑LOOP (PREDIKSI OTOMATIS)
# ──────────────────────────────────────────
def auto_prediction_loop(model, dmin, dmax):
    live_file = os.path.join(DATA_DIR, "aviator_data_live.json")
    last_idx = 0
    print("[AUTO] Menunggu data live...")
    while True:
        if not os.path.exists(live_file):
            time.sleep(5)
            continue
        try:
            with open(live_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            time.sleep(5)
            continue

        if len(data) < SEQ_LEN + 1:
            time.sleep(5)
            continue

        new_data = data[last_idx:]
        while len(new_data) >= SEQ_LEN + 1:
            inputs = new_data[:SEQ_LEN]
            actual = new_data[SEQ_LEN]
            pred = predict(model, inputs, dmin, dmax)
            binary_pred = "YA" if pred > BINARY_THRESHOLD else "TIDAK"
            binary_actual = "YA" if actual > BINARY_THRESHOLD else "TIDAK"
            binary_correct = (binary_pred == binary_actual)
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " \
                        f"{' '.join(f'{x:.2f}' for x in inputs)} → Pred:{pred:.2f}x | Actual:{actual:.2f}x | " \
                        f"Binary: {binary_pred} vs {binary_actual} | {'BENAR' if binary_correct else 'SALAH'}"
            try:
                with open(LOG_FILE, 'a') as f:
                    f.write(log_entry + "\n")
                print(f"[AUTO] {log_entry}")
            except Exception as e:
                print(f"[AUTO] ⚠️ Gagal menulis log: {e}")

            # maju 1 langkah
            last_idx += 1
            new_data = data[last_idx:] if last_idx < len(data) else []
        
        time.sleep(5)


# ──────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────
def main():
    print("="*55)
    print("🎲 POLA LSTM AUTO — Binary Analyzer + Live Loop")
    print("="*55)

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        print("📦 Loading existing model...")
        model, dmin, dmax = load_model()
    else:
        print("🆕 Training model baru...")
        sessions = load_all_sessions(DATA_DIR)
        train_model(sessions)
        model, dmin, dmax = load_model()

    # Langsung tawarkan mode auto atau manual
    print("\nPilih mode:")
    print("  a = Auto‑loop (baca live data & prediksi otomatis)")
    print("  m = Menu interaktif manual")
    choice = input("Mode (a/m): ").strip().lower()
    if choice == 'a':
        auto_prediction_loop(model, dmin, dmax)
    else:
        # Menu manual (sama seperti sebelumnya)
        while True:
            print("\n" + "─"*50)
            print("--- MENU ---")
            print(" 1. Prediksi (log ke file) + Binary")
            print(" 2. Lihat statistik semua data")
            print(" 3. Retrain full model")
            print(" 4. Analisis pola error (binary)")
            print(" q. Keluar")
            print("─"*50)

            try:
                choice = input("Pilih menu: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == '1':
                menu_prediksi(model, dmin, dmax)
            elif choice == '2':
                menu_statistik()
            elif choice == '3':
                new_model, new_dmin, new_dmax = menu_retrain()
                if new_model is not None:
                    model, dmin, dmax = new_model, new_dmin, new_dmax
            elif choice == '4':
                menu_analisis_error()
            elif choice == 'q':
                print("👋 Bye! Sampai jumpa lagi.\n")
                break
            else:
                print("❌ Pilihan tidak valid!")


if __name__ == "__main__":
    main()
