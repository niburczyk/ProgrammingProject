import joblib
import numpy as np
from scipy.signal import firwin, lfilter
import serial
import time
import threading
import os
from queue import Queue

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
    arduino = serial.Serial('COM5', 230400, timeout=1)  # Passe COM-Port ggf. an
    time.sleep(2)
    print("Arduino verbunden.")
except Exception as e:
    print("Arduino konnte nicht verbunden werden:", e)
    arduino = None

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
        print("Kein Datenpuffer zum Speichern.")
        return
    os.makedirs('./data', exist_ok=True)
    filename = f"recorded_data_{int(time.time()*1000)}.txt"
    filepath = os.path.join('./data', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in recording_save:
            line = ','.join(map(str, sample))
            f.write(line + '\n')
    print(f"Rohdaten gespeichert in {filepath}")

def save_prediction_to_file(predictions):
    if not predictions:
        print("Keine Vorhersagedaten zum Speichern.")
        return
    os.makedirs('./data', exist_ok=True)
    filename = f"prediction_{int(time.time()*1000)}.txt"
    filepath = os.path.join('./data', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pred in predictions:
            line = f"{pred[0]},{pred[2]}"
            f.write(line + '\n')
    print(f"Vorhersagen gespeichert in {filepath}")

# === Globale Variablen ===
WINDOW_SIZE = 250
recording = False
recording_save = []
predictions_save = []
data_queue = Queue()
num_channels = None

# === Threads ===
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

def serial_reader():
    global num_channels
    while True:
        try:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
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
                with open("./data/emg_raw_live.txt", "a", encoding="utf-8") as f:
                    f.write(','.join(map(str, [timestamp_ms] + sample)) + '\n')
        except Exception as e:
            print("Lesefehler:", e)

def data_processor():
    buffer = []
    while True:
        timestamp_ms, sample = data_queue.get()
        buffer.append(sample)
        if len(buffer) >= WINDOW_SIZE:
            try:
                data_np = np.array(buffer[-WINDOW_SIZE:])
                filtered = bandpass_filter(data_np, lowcut, highcut, fs, order)
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

# === Main ===
if arduino:
    threading.Thread(target=input_thread, daemon=True).start()
    threading.Thread(target=serial_reader, daemon=True).start()
    threading.Thread(target=data_processor, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Programm beendet durch Benutzer.")
        arduino.close()
else:
    print("Programm beendet – keine serielle Verbindung.")
