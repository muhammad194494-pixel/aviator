import json
import os
import glob

# Folder tempat data lo berada
DATA_DIR = "/home/muhammad194494/aviator"

def check_data_anomali(threshold=100.0):
    print(f"🔍 Mencari data dengan multiplier > {threshold}x...\n")
    
    files = glob.glob(os.path.join(DATA_DIR, "aviator_data*.json"))
    found_any = False
    
    for f in files:
        filename = os.path.basename(f)
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            
            # Cek apakah ada nilai > threshold
            anomalies = [val for val in data if val > threshold]
            
            if anomalies:
                found_any = True
                print(f"🚨 Ditemukan di {filename}:")
                print(f"   Jumlah anomali: {len(anomalies)} kejadian")
                print(f"   Nilai tertinggi: {max(anomalies)}x")
                print("-" * 30)
                
        except Exception as e:
            print(f"⚠️ Gagal membaca {filename}: {e}")
            
    if not found_any:
        print(f"✅ Aman! Tidak ada data di atas {threshold}x.")

if __name__ == "__main__":
    check_data_anomali(100.0)
