import joblib
import numpy as np
from scipy.signal import firwin, lfilter
import serial
import time
import threading
import os
from datetime import datetime

# === Lade Modell, Scaler, PCA, Label Encoder ===
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')

# === Bandpassfilter Parameter ===
lowcut = 1.25
highcut = 22.5
fs = 2000
order = 4

# === Arduino-Verbindung ===
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino verbunden.")
except Exception as e:
    arduino = None
    print("⚠️ Arduino konnte nicht verbunden werden:", e)

# === Funktionen ===
def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    filtered = lfilter(taps, 1.0, data, axis=0)
    return filtered[numtaps:]

def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

def save_buffer_to_file(recording_save):
    if not recording_save:
        print("⚠️ Kein Datenpuffer zum Speichern.")
        return
    os.makedirs('/data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_data_{timestamp}.txt"
    filepath = os.path.join('/data', filename)
    with open(filepath, 'w') as f:
        for sample in recording_save:
            line = ','.join(str(x) for x in sample)
            f.write(line + '\n')
    print(f"✅ Daten gespeichert in {filepath}")

def save_prediction_to_file(prediction):
    os.makedirs('/data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{timestamp}.txt"
    filepath = os.path.join('/data', filename)
    with open(filepath, 'w') as f:
        f.write(f"Vorhersage: {prediction}\n")
    print(f"✅ Vorhersage gespeichert in {filepath}")

# === Globale Variablen ===
WINDOW_SIZE = 250
num_channels = None
recording = False
buffer = []
recording_save = []
predictions_save = []

# === Eingabe-Thread für START/STOP ===
def input_thread():
    global recording, recording_save, predictions_save, buffer
    print("Eingabe-Thread gestartet (START/STOP).")
    while True:
        cmd = input("Eingabe (START/STOP): ").strip().upper()
        if cmd == "START":
            recording = True
            recording_save = []
            predictions_save = []
            buffer = []
            print("Aufnahme gestartet.")
        elif cmd == "STOP":
            recording = False
            print("Aufnahme gestoppt.")
            save_buffer_to_file(recording_save)
            save_prediction_to_file(predictions_save)

# === Hauptprogramm ===
def main():
    global recording, buffer, num_channels, recording_save, predictions_save

    if not arduino:
        print("Kein Arduino verbunden. Programm beendet.")
        return

    threading.Thread(target=input_thread, daemon=True).start()

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
                print(f"Ungültige Daten empfangen: {line}")
                continue

            if recording:
                buffer.append(sample)
                recording_save.append(sample)

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

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    predictions_save.append((timestamp, class_label, int(prediction), *mav, *wl))

                    print(f"Vorhersage: {class_label} ({prediction}) | MAV: {mav}, WL: {wl}")

                    if len(buffer) > WINDOW_SIZE * 10:
                        buffer = buffer[-WINDOW_SIZE * 5:]

        except KeyboardInterrupt:
            print("Programm durch Benutzer beendet.")
            break
        except Exception as e:
            print("Fehler:", e)
            continue

    arduino.close()

if __name__ == "__main__":
    main()
