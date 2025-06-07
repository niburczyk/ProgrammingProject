import joblib
import numpy as np
from scipy.signal import firwin, lfilter
import serial
import time
import threading
import os
import sys
from queue import Queue

# === Konfiguration ===
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 230400

DATA_SAVE_DIR = os.path.expanduser('/data')  # Beispiel für Windows & Linux kompatibel

# === Lade Modell, Scaler, PCA, Label Encoder ===
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')

# === Bandpassfilter Parameter ===
lowcut = 1.25
highcut = 22.5
fs = 2000
numtaps = 101

# === Arduino-Verbindung ===
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Arduino verbunden an {SERIAL_PORT}.")
except Exception as e:
    arduino = None
    print(f"Arduino konnte nicht verbunden werden: {e}")
    sys.exit(1)

# === Funktionen ===
def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    filtered = lfilter(taps, 1.0, data, axis=0)
    return filtered[numtaps:]  # Verzögerung durch Filter berücksichtigen

def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

def save_buffer_to_file(recording_save):
    if not recording_save:
        print("Kein Datenpuffer zum Speichern.")
        return
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    filename = f"recorded_data_{int(time.time()*1000)}.txt"
    filepath = os.path.join(DATA_SAVE_DIR, filename)

    with open(filepath, 'w', encoding='utf-8', buffering=1) as f:  # line buffering
        for sample in recording_save:
            line = ','.join(map(str, sample))
            f.write(line + '\n')

    print(f"Daten gespeichert in {filepath}")

def save_prediction_to_file(predictions):
    if not predictions:
        print("Keine Vorhersagedaten zum Speichern.")
        return
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    filename = f"prediction_{int(time.time()*1000)}.txt"
    filepath = os.path.join(DATA_SAVE_DIR, filename)

    with open(filepath, 'w', encoding='utf-8', buffering=1) as f:  # line buffering
        for pred in predictions:
            line = f"{pred[0]},{pred[2]}"
            f.write(line + '\n')

    print(f"Vorhersagen gespeichert in {filepath}")

# === Globale Variablen ===
WINDOW_SIZE = 250
num_channels = None
recording = False
recording_save = []
predictions_save = []
data_queue = Queue()
write_queue = Queue()

# === Eingabe-Thread für START/STOP ===
def input_thread():
    global recording, recording_save, predictions_save
    print("Eingabe-Thread gestartet (START/STOP).")
    while True:
        cmd = input("Eingabe (START/STOP): ").strip().upper()
        if cmd == "START":
            recording = True
            recording_save = []
            predictions_save = []
            print("Aufnahme gestartet.")
        elif cmd == "STOP":
            recording = False
            print("Aufnahme gestoppt.")
            save_buffer_to_file(recording_save)
            save_prediction_to_file(predictions_save)

# === Thread zum Schreiben in Datei ===
def file_writer():
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    filepath = os.path.join(DATA_SAVE_DIR, "emg_raw_live.txt")
    with open(filepath, "a", encoding="utf-8") as f:
        while True:
            line = write_queue.get()
            f.write(line + '\n')

# === Thread zum Einlesen der seriellen Daten ===
def serial_reader():
    global num_channels
    while True:
        try:
            line_bytes = arduino.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parts = line.split(',')
            if num_channels is None:
                num_channels = len(parts)
                print(f"Anzahl Kanäle erkannt: {num_channels}")

            sample = [float(x) for x in parts]
            timestamp_ms = int(time.time() * 1000)

            if recording:
                recording_save.append([timestamp_ms] + sample)
                data_queue.put((timestamp_ms, sample))
                write_queue.put(','.join(map(str, [timestamp_ms] + sample)))
        except Exception as e:
            print("Lesefehler:", e)

# === Thread zum Verarbeiten der Daten ===
def data_processor():
    buffer = []
    while True:
        timestamp_ms, sample = data_queue.get()
        buffer.append(sample)
        if len(buffer) >= WINDOW_SIZE:
            try:
                data_np = np.array(buffer[-WINDOW_SIZE:])
                filtered = bandpass_filter(data_np, lowcut, highcut, fs, numtaps)
                if filtered.shape[0] < WINDOW_SIZE - numtaps:
                    continue

                mav = calculate_mav(filtered)
                wl = calculate_wl(filtered)
                features = np.concatenate((mav, wl)).reshape(1, -1)

                scaled = scaler.transform(features)
                reduced = pca.transform(scaled)
                prediction = model.predict(reduced)[0]
                class_label = label_encoder.inverse_transform([prediction])[0]

                predictions_save.append((timestamp_ms, class_label, int(prediction)))
                print(f"Vorhersage: {class_label} ({prediction})")

                if len(buffer) > WINDOW_SIZE * 10:
                    buffer = buffer[-WINDOW_SIZE * 5:]
            except Exception as e:
                print("Verarbeitungsfehler:", e)

# === Hauptprogramm ===
def main():
    threading.Thread(target=input_thread, daemon=True).start()
    threading.Thread(target=serial_reader, daemon=True).start()
    threading.Thread(target=data_processor, daemon=True).start()
    threading.Thread(target=file_writer, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet.")
    finally:
        if arduino and arduino.is_open:
            arduino.close()
        print("Port geschlossen.")

if __name__ == "__main__":
    main()
