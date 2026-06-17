#!/usr/bin/env python3
"""
AI AVIATOR v2 FINAL
- SEQ_LEN = 4 (fixed)
- JSON format: [1.2, 2.5, 3.1]
- Binary prediction (>2x)
- Stable LSTM classifier
"""

import os
import json
import glob
import numpy as np
from datetime import datetime

# ================= CONFIG =================
DATA_DIR = "/home/bangmael229/prediksi"
MODEL_PATH = os.path.join(DATA_DIR, "aviator_v2.keras")
LOG_FILE = os.path.join(DATA_DIR, "log_v2.txt")

SEQ_LEN = 4
THRESHOLD = 2.0
EPOCHS = 120
BATCH = 32
# =========================================


# ─────────────────────────────
# LOAD DATA
# ─────────────────────────────
def load_all_sessions(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "aviator_data*.json")))
    sessions = []

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            data = np.array(data, dtype=np.float32)

            # clean noise
            data = data[data > 1.0]

            # FIX: reverse biar urutan time correct
            data = data[::-1]

            sessions.append(data)

        except Exception as e:
            print(f"Skip {f}: {e}")

    return sessions


# ─────────────────────────────
# BUILD DATASET
# ─────────────────────────────
def build_dataset(sessions):
    X_all, y_all = [], []

    for sesi in sessions:
        if len(sesi) <= SEQ_LEN:
            continue

        for i in range(len(sesi) - SEQ_LEN):
            x = sesi[i:i+SEQ_LEN]
            y = sesi[i+SEQ_LEN]

            label = 1 if y > THRESHOLD else 0

            X_all.append(x)
            y_all.append(label)

    X_all = np.array(X_all).reshape(-1, SEQ_LEN, 1)
    y_all = np.array(y_all)

    return X_all, y_all


# ─────────────────────────────
# MODEL
# ─────────────────────────────
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.25),

        LSTM(32),
        Dropout(0.25),

        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ─────────────────────────────
# TRAIN
# ─────────────────────────────
def train():
    from tensorflow.keras.callbacks import EarlyStopping

    sessions = load_all_sessions(DATA_DIR)
    X, y = build_dataset(sessions)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model()

    es = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[es],
        verbose=1
    )

    model.save(MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)

    return model


# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
def load_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


# ─────────────────────────────
# PREDICT
# ─────────────────────────────
def predict(model, inputs):
    x = np.array(inputs, dtype=np.float32).reshape(1, SEQ_LEN, 1)

    prob = model.predict(x, verbose=0)[0][0]

    return {
        "probability": float(prob),
        "decision": "BET (>2x)" if prob > 0.65 else "SKIP",
        "zone": "LOW" if prob < 0.4 else "MID" if prob < 0.7 else "HIGH"
    }


# ─────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────
def run_cli(model):
    print("\n🎲 AI AVIATOR v2 READY (SEQ_LEN=4)")
    print("Masukkan 4 multiplier terakhir\n")

    while True:
        inp = input("Input (4 angka / q): ").strip()
        if inp.lower() == 'q':
            break

        try:
            nums = [float(x) for x in inp.split()]
            if len(nums) != 4:
                print("❌ Harus 4 angka")
                continue
        except:
            print("❌ Input salah")
            continue

        result = predict(model, nums)

        print("\n──────── RESULT ────────")
        print("Input :", nums)
        print("Prob  :", round(result["probability"], 3))
        print("Zone  :", result["zone"])
        print("Signal:", result["decision"])
        print("------------------------\n")


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("📦 Loading model...")
        model = load_model()
    else:
        print("🆕 Training model...")
        model = train()

    run_cli(model)