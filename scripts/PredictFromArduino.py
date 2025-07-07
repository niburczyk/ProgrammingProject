import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from serial import Serial
import time
import os
import sys
import threading
from multiprocessing import Process, Queue, Event, freeze_support
from concurrent.futures import ThreadPoolExecutor

# === Konfiguration ===
SERIAL_PORT = 'COM3'  # Passe ggf. an
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
nyq = 0.5 * fs
sos = butter(4, [lowcut / nyq, highcut / nyq], btype='bandpass', output='sos')  # ersetzt firwin

WINDOW_SIZE = 500
num_channels = None
stop_event = Event()
start_time_global = None

data_queue = Queue()

def bandpass_filter(data):
    return sosfiltfilt(sos, data, axis=0)

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

def input_thread():
    try:
        while not stop_event.is_set():
            try:
                cmd = input().strip().upper()
            except EOFError:
                print("Input EOF erkannt, Thread wird beendet.")
                stop_event.set()
                break
            if cmd == "STOP":
                stop_event.set()
    except Exception as e:
        print(f"Unerwarteter Fehler im Input-Thread: {e}")

def read_serial_thread(arduino):
    global num_channels
    while not stop_event.is_set():
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
                continue
            timestamp_ms = int(time.time() * 1000)
            data_queue.put((timestamp_ms, sample))
            if start_time_global and (time.time() - start_time_global) >= 30:
                print("30 Sekunden erreicht – Aufnahme wird gestoppt.")
                stop_event.set()
        else:
            time.sleep(0.001)

def processing_process(data_queue, stop_event):
    buffer = []
    recording_save = []
    predictions_save = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        while not stop_event.is_set() or not data_queue.empty():
            try:
                timestamp_ms, sample = data_queue.get(timeout=0.1)
            except:
                continue
            buffer.append(sample)
            recording_save.append([timestamp_ms] + sample)
            if len(buffer) >= WINDOW_SIZE:
                data_np = np.array(buffer[-WINDOW_SIZE:])
                future = executor.submit(bandpass_filter, data_np)
                filtered = future.result()
                mav = calculate_mav(filtered)
                wl = calculate_wl(filtered)
                features = np.concatenate((mav, wl)).reshape(1, -1)
                scaled = scaler.transform(features)
                reduced = pca.transform(scaled)
                prediction = model.predict(reduced)[0]
                class_label = label_encoder.inverse_transform([prediction])[0]
                predictions_save.append((timestamp_ms, class_label, int(prediction)))
                if len(buffer) > WINDOW_SIZE * 10:
                    buffer = buffer[-WINDOW_SIZE * 5:]
    duration = time.time() - start_time
    print(f"Aufnahme gestoppt. Dauer: {duration:.2f} Sek. Samples: {len(recording_save)}")
    save_buffer_to_file(recording_save)
    save_prediction_to_file(predictions_save)

def timer_thread():
    while not stop_event.is_set():
        if start_time_global:
            elapsed = time.time() - start_time_global
            print(f"{elapsed:.1f} Sekunden seit START...")
        time.sleep(1)

def main():
    global num_channels, start_time_global
    try:
        arduino = Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Arduino verbunden an {SERIAL_PORT}.")
    except Exception as e:
        print(f"Verbindung fehlgeschlagen: {e}")
        sys.exit(1)
    while True:
        cmd = input("Eingabe (START/EXIT): ").strip().upper()
        if cmd == "EXIT":
            break
        elif cmd != "START":
            continue
        print("Aufnahme gestartet... (wird nach 30 Sekunden automatisch gestoppt)")
        stop_event.clear()
        start_time_global = time.time()

        input_thr = threading.Thread(target=input_thread)
        read_thr = threading.Thread(target=read_serial_thread, args=(arduino,))
        timer_thr = threading.Thread(target=timer_thread)
        proc_proc = Process(target=processing_process, args=(data_queue, stop_event))

        input_thr.start()
        read_thr.start()
        timer_thr.start()
        proc_proc.start()

        while not stop_event.is_set():
            time.sleep(0.1)

        input_thr.join()
        read_thr.join()
        timer_thr.join()
        proc_proc.join()

        print("Threads und Prozesse beendet, Aufnahme gestoppt.")

    arduino.close()
    print("Serielle Verbindung geschlossen.")

if __name__ == "__main__":
    freeze_support()  # Wichtig für Windows + multiprocessing
    main()