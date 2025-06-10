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
DATA_SAVE_DIR = os.path.expanduser('./data')

# === Modell laden ===
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')

# === Filterparameter ===
lowcut = 1.25
highcut = 22.5
fs = 2000
numtaps = 101

# === Globale Variablen ===
WINDOW_SIZE = 500
num_channels = None
recording = False
buffer = []
recording_save = []
predictions_save = []
data_queue = Queue()
start_time = None

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

def save_buffer_to_file(data):
    if not data:
        print("Kein Datenpuffer zum Speichern.")
        return
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    filename = f"recorded_data_{int(time.time()*1000)}.txt"
    filepath = os.path.join(DATA_SAVE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(','.join(map(str, sample)) + '\n')
    print(f"Daten gespeichert in {filepath}")

def save_prediction_to_file(predictions):
    if not predictions:
        print("Keine Vorhersagedaten zum Speichern.")
        return
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    filename = f"prediction_{int(time.time()*1000)}.txt"
    filepath = os.path.join(DATA_SAVE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred[0]},{pred[2]}\n")
    print(f"Vorhersagen gespeichert in {filepath}")

# === Lesethread ===
def serial_reader_thread(arduino):
    while True:
        try:
            line_bytes = arduino.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if line:
                data_queue.put(line)
        except Exception as e:
            print("Fehler im Lesethread:", e)

# === Eingabe-Thread ===
def input_thread():
    global recording, recording_save, predictions_save, buffer, start_time
    print("Eingabe-Thread gestartet (START/STOP).")
    while True:
        cmd = input("Eingabe (START/STOP): ").strip().upper()
        if cmd == "START":
            recording = True
            recording_save = []
            predictions_save = []
            buffer = []
            start_time = time.time()
            print("Aufnahme gestartet.")
        elif cmd == "STOP":
            recording = False
            duration = time.time() - start_time if start_time else 0
            print(f"Aufnahme gestoppt. Dauer: {duration:.2f} Sek. Samples: {len(recording_save)}")
            save_buffer_to_file(recording_save)
            save_prediction_to_file(predictions_save)

# === Hauptprogramm ===
def main():
    global num_channels, recording, buffer, recording_save, predictions_save

    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Arduino verbunden an {SERIAL_PORT}.")
    except Exception as e:
        print(f"Verbindung fehlgeschlagen: {e}")
        sys.exit(1)

    threading.Thread(target=input_thread, daemon=True).start()
    threading.Thread(target=serial_reader_thread, args=(arduino,), daemon=True).start()

    print("Warte auf serielle Daten...")

    try:
        while True:
            if not data_queue.empty():
                line = data_queue.get()
                parts = line.split(',')
                if num_channels is None:
                    num_channels = len(parts)
                    print(f"Anzahl Kanäle erkannt: {num_channels}")

                try:
                    # === Rohdaten umrechnen: ADC → Volt → mV ===
                    adc_resolution = 1023
                    v_ref = 3.0  # Referenzspannung (Volt)
                    gain_total = 2848  # Gesamter Verstärkungsfaktor

                    # Umrechnung auf jeden Kanal anwenden
                    sample = [(x / adc_resolution) * v_ref for x in sample]
                    sample = [(x / gain_total) * 1e3 for x in sample]  # jetzt in mV
                except ValueError:
                    print(f"Ungültige Zeile: {line}")
                    continue

                if recording:
                    timestamp_ms = int(time.time() * 1000)
                    buffer.append(sample)
                    recording_save.append([timestamp_ms] + sample)

                    if len(buffer) >= WINDOW_SIZE:
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
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet.")
    finally:
        if arduino and arduino.is_open:
            arduino.close()
        print("Serielle Verbindung geschlossen.")

if __name__ == "__main__":
    main()
