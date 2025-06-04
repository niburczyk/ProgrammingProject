import joblib
import numpy as np
from scipy.signal import firwin, lfilter
import serial
import time
import threading
import os
from datetime import datetime

# === Lade Modell, Scaler, PCA, Label Encoder
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')

# === Bandpassfilter Parameter
lowcut = 1.25
highcut = 22.5
fs = 2000
order = 4

# === Arduino-Serielle Verbindung
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino verbunden.")
except Exception as e:
    arduino = None
    print("⚠️ Arduino konnte nicht verbunden werden:", e)

def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut/nyq, highcut/nyq], pass_zero=False)
    filtered = lfilter(taps, 1.0, data, axis=0)
    return filtered[numtaps:]  # Verzögerung kompensieren

def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

WINDOW_SIZE = 250
num_channels = None
recording = False  # globaler Status
buffer = []         # globaler Datenpuffer

def save_buffer_to_file(buffer):
    if not buffer:
        print("⚠️ Kein Datenpuffer zum Speichern.")
        return
    os.makedirs('./data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_data_{timestamp}.txt"
    filepath = os.path.join('./data', filename)
    with open(filepath, 'w') as f:
        for sample in buffer:
            line = ','.join(str(x) for x in sample)
            f.write(line + '\n')
    print(f"✅ Daten gespeichert in {filepath}")

def input_thread():
    global recording
    print("Eingabe-Thread gestartet, bitte START oder STOP eingeben.")
    while True:
        cmd = input("Eingabe (START/STOP): ").strip().upper()
        if cmd == "START":
            recording = True
            print("▶️ Aufnahme gestartet.")
        elif cmd == "STOP":
            recording = False
            print("⏹️ Aufnahme gestoppt.")

def main():
    global recording, buffer, num_channels
    if not arduino:
        print("ℹ️ Kein Arduino verbunden, Programm beendet.")
        return

    # Starte Eingabe-Thread
    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()

    while True:
        try:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parts = line.split(',')
            if num_channels is None:
                num_channels = len(parts)
                print(f"Anzahl Kanäle erkannt: {num_channels}")

            try:
                sample = [float(x) for x in parts]
            except ValueError:
                print(f"⚠️ Ungültige Daten: {line}")
                continue

            if recording:
                buffer.append(sample)

                if len(buffer) >= WINDOW_SIZE:
                    data_np = np.array(buffer[-WINDOW_SIZE:])

                    filtered = bandpass_filter(data_np, lowcut, highcut, fs, order)

                    mav = calculate_mav(filtered)
                    wl = calculate_wl(filtered)
                    features = np.concatenate((mav, wl)).reshape(1, -1)

                    scaled = scaler.transform(features)
                    reduced = pca.transform(scaled)
                    prediction = model.predict(reduced)[0]
                    class_label = label_encoder.inverse_transform([prediction])[0]

                    print(f"📣 Vorhersage: {class_label} ({prediction}) | MAV: {mav}, WL: {wl}")

                    # Buffer trimmen, um Speicher zu sparen
                    if len(buffer) > WINDOW_SIZE * 10:
                        buffer = buffer[-WINDOW_SIZE * 5:]

        except KeyboardInterrupt:
            print("⏹️ Beendet durch Benutzer")
            break
        except Exception as e:
            print("⚠️ Fehler beim Lesen/Verarbeiten:", e)
            continue

    arduino.close()

if __name__ == "__main__":
    main()
