import joblib
import numpy as np
from collections import deque
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
import serial
import time
import matplotlib.pyplot as plt

# === Lade Modell, Scaler, PCA
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')

# === Bandpassfilter Parameter
lowcut = 1.25
highcut = 22.5
fs = 2000
order = 4

# === Arduino-Serielle Verbindung (optional)
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino verbunden.")
except Exception as e:
    arduino = None
    print("⚠️ Arduino konnte nicht verbunden werden:", e)

# === Filterfunktion
def bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=0)

# === Featurefunktionen
def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

# === Parameter Fenster
WINDOW_SIZE = 250
STEP_SIZE = 125
num_channels = None

# === Datenstruktur für Streaming
window = deque(maxlen=WINDOW_SIZE)

if arduino:
    # === Wenn Arduino verbunden: Laufend Daten einlesen und verarbeiten
    print("🔍 Starte Live-Datenverarbeitung vom Arduino...")

    # Warte auf genügend Samples für das Fenster
    buffer = []

    plt.ion()
    fig, ax = plt.subplots()
    line_signal, = ax.plot([], [], label="EMG-Kanal 1")
    window_box = ax.axvspan(0, WINDOW_SIZE / fs, color='red', alpha=0.3, label="Aktuelles Fenster")
    ax.set_title("Live EMG-Signal mit gleitendem Fenster")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()

    while True:
        try:
            line = arduino.readline().decode('utf-8').strip()
            if not line:
                continue
            # Angenommen, Daten sind CSV: "123,456,789"
            parts = line.split(',')
            if num_channels is None:
                num_channels = len(parts)
                print(f"Anzahl Kanäle erkannt: {num_channels}")

            # Konvertiere in float
            sample = list(map(float, parts))
            buffer.append(sample)

            if len(buffer) >= WINDOW_SIZE:
                data_np = np.array(buffer[-WINDOW_SIZE:])  # letze WINDOW_SIZE Samples

                # Filterung
                filtered = bandpass_filter(data_np, lowcut, highcut, fs, order)

                # Featureberechnung
                mav = calculate_mav(filtered)
                wl = calculate_wl(filtered)
                features = np.concatenate((mav, wl)).reshape(1, -1)

                # Transformation und Vorhersage
                scaled = scaler.transform(features)
                reduced = pca.transform(scaled)
                prediction = model.predict(reduced)[0]
                class_label = label_encoder.inverse_transform([prediction])[0]

                print(f"📣 Vorhersage: {class_label} ({prediction}) | MAV: {mav}, WL: {wl}")

                # Sende Vorhersage an Arduino
                try:
                    arduino.write(f"{prediction}\n".encode())
                except Exception as err:
                    print("❌ Fehler beim Senden an Arduino:", err)

                # Plot aktualisieren
                time_axis = np.arange(len(filtered)) / fs
                line_signal.set_data(time_axis, filtered[:, 0])
                ax.set_xlim(time_axis[0], time_axis[-1])
                ax.set_ylim(np.min(filtered[:,0])*1.1, np.max(filtered[:,0])*1.1)
                window_box.remove()
                window_box = ax.axvspan(time_axis[0], time_axis[-1], color='red', alpha=0.3)
                plt.pause(0.001)

                # Optional: buffer kürzen, um Speicher zu sparen
                if len(buffer) > WINDOW_SIZE * 10:
                    buffer = buffer[-WINDOW_SIZE*5:]

        except KeyboardInterrupt:
            print("⏹️ Beendet durch Benutzer")
            break
        except Exception as e:
            print("⚠️ Fehler beim Lesen/Verarbeiten:", e)
            continue

    plt.ioff()
    plt.show()

    arduino.close()

else:
    # === Kein Arduino, lade Daten aus Datei
    print("ℹ️ Kein Arduino, lade Daten aus Datei.")

    mat = loadmat('./sample/Condition-P/P20.mat')
    fs = 2000  # Samplingrate sicherheitshalber nochmal setzen
    remove_seconds = 3
    remove_samples = remove_seconds * fs

    raw_data = mat['data']
    print("Vor Abschneiden:", raw_data.shape)
    raw_data = raw_data[remove_samples:-remove_samples, :]
    print("Nach Abschneiden:", raw_data.shape)

    filtered_data = bandpass_filter(raw_data, lowcut, highcut, fs, order)

    time_axis = np.arange(filtered_data.shape[0]) / fs

    plt.ion()
    fig, ax = plt.subplots()
    line_signal, = ax.plot(time_axis, filtered_data[:, 0], label="EMG-Kanal 1")
    window_box = ax.axvspan(0, WINDOW_SIZE / fs, color='red', alpha=0.3, label="Aktuelles Fenster")
    ax.set_title("EMG-Signal mit gleitendem Fenster")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()

    print("🔍 Starte Verarbeitung der Datei-Daten...")

    for i in range(WINDOW_SIZE, len(filtered_data), STEP_SIZE):
        window_np = filtered_data[i - WINDOW_SIZE:i]
        mav = calculate_mav(window_np)
        wl = calculate_wl(window_np)
        features = np.concatenate((mav, wl)).reshape(1, -1)

        scaled = scaler.transform(features)
        reduced = pca.transform(scaled)
        prediction = model.predict(reduced)[0]
        class_label = label_encoder.inverse_transform([prediction])[0]

        print(f"📣 Vorhersage: {class_label} ({prediction}) | MAV: {mav}, WL: {wl}")

        # Sende an Arduino falls verbunden (hier nicht nötig)

        window_start = i - WINDOW_SIZE
        window_end = i
        window_box.remove()
        window_box = ax.axvspan(time_axis[window_start], time_axis[window_end], color='red', alpha=0.3)
        plt.pause(0.001)

    plt.ioff()
    plt.show()
