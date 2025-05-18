import joblib
import numpy as np
from collections import deque
from scipy.io import loadmat
import serial
import time

# === Lade Modell, Scaler, PCA
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')

# === Arduino-Serielle Verbindung (falls angeschlossen)
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Passe Port ggf. an
    time.sleep(2)  # Zeit geben zum Verbinden
    print("✅ Arduino verbunden.")
except Exception as e:
    arduino = None
    print("⚠️  Arduino konnte nicht verbunden werden:", e)

# === Lade EMG-Daten aus .mat-Datei
mat = loadmat('./sample/Condition-O/O9.mat')  # Dateipfad anpassen
signal_data = mat['data']
print("Daten:", signal_data)

# === Fenstergröße
WINDOW_SIZE = 50
window = deque(maxlen=WINDOW_SIZE)

# === Feature-Berechnung
def calculate_mav(sig):
    return np.mean(np.abs(sig))

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig)))

print("🔍 Starte Verarbeitung...")

# === Simulation: Schrittweises Senden der Signale
for sample in signal_data:
    window.append(sample)

    if len(window) == WINDOW_SIZE:
        mav = calculate_mav(window)
        wl = calculate_wl(window)
        features = np.array([[mav, wl]])

        # Skalierung & PCA
        scaled = scaler.transform(features)
        reduced = pca.transform(scaled)

        # Vorhersage
        prediction = model.predict(reduced)[0]
        print(f"📣 Vorhersage: {prediction}")

        # An Arduino senden, falls verbunden
        if arduino:
            try:
                arduino.write(f"{prediction}\n".encode())
            except Exception as err:
                print("❌ Fehler beim Senden an Arduino:", err)

        time.sleep(0.05)  # simuliert Live-Datenstrom

if arduino:
    arduino.close()
