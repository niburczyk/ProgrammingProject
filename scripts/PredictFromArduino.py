import joblib
import numpy as np
from scipy.signal import firwin, lfilter
from serial import Serial
import time
import os
import sys

# === Konfiguration ===
SERIAL_PORT = 'COM5'
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

WINDOW_SIZE = 500
num_channels = None

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

def main():
    global num_channels

    try:
        arduino = Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Arduino verbunden an {SERIAL_PORT}.")
    except Exception as e:
        print(f"Verbindung fehlgeschlagen: {e}")
        sys.exit(1)

    while True:
        cmd = input("Eingabe (START/STOP/EXIT): ").strip().upper()

        if cmd == "EXIT":
            break
        elif cmd != "START":
            continue

        print("Aufnahme gestartet... (STOP zum Beenden)")

        buffer = []
        recording_save = []
        predictions_save = []
        start_time = time.time()

        try:
            while True:
                if arduino.in_waiting:
                    line_bytes = arduino.readline()
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    parts = line.split(',')
                    if num_channels is None:
                        num_channels = len(parts)
                        print(f"Anzahl Kanäle erkannt: {num_channels}")

                    try:
                        sample = list(map(float, parts))
                        adc_resolution = 1023
                        v_ref = 3.0
                        gain_total = 2848

                        sample = [(x / adc_resolution) * v_ref for x in sample]
                        sample = [(x / gain_total) * 1e3 for x in sample]
                    except ValueError:
                        print(f"Ungültige Zeile: {line}")
                        continue

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

                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    stop_cmd = input().strip().upper()
                    if stop_cmd == "STOP":
                        break

        except KeyboardInterrupt:
            print("Aufnahme unterbrochen.")
        finally:
            duration = time.time() - start_time
            print(f"Aufnahme gestoppt. Dauer: {duration:.2f} Sek. Samples: {len(recording_save)}")
            save_buffer_to_file(recording_save)
            save_prediction_to_file(predictions_save)

    arduino.close()
    print("Serielle Verbindung geschlossen.")

if __name__ == "__main__":
    import select  # Für stdin-check im Hauptloop
    main()
