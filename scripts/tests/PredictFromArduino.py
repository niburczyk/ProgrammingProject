import joblib
import numpy as np
from collections import deque
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
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
    print("⚠️  Arduino konnte nicht verbunden werden:", e)

# === Lade EMG-Daten
mat = loadmat('./sample/Condition-P/P20.mat')
fs = 2000  # Abtastrate
remove_seconds = 3
remove_samples = remove_seconds * fs

# raw_data: (Samples x Channels)
print("Vor Abschneiden:", mat['data'].shape)
raw_data = mat['data']
raw_data = raw_data[remove_samples : -remove_samples, :]
print("Nach Abschneiden:", raw_data.shape)


# === Filter
def bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

filtered_data = bandpass_filter(raw_data, lowcut, highcut, fs, order)

# === Fenstergröße
WINDOW_SIZE = 250
STEP_SIZE = 125 
window = deque(maxlen=WINDOW_SIZE)

# === Features
def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

# === Zeitachse in Sekunden
time_axis = np.arange(filtered_data.shape[0]) / fs

# === Visualisierung vorbereiten (1 Kanal, z. B. Kanal 0)
plt.ion()
fig, ax = plt.subplots()
line_signal, = ax.plot(time_axis, filtered_data[:, 0], label="EMG-Kanal 1")
window_box = ax.axvspan(0, WINDOW_SIZE / fs, color='red', alpha=0.3, label="Aktuelles Fenster")
ax.set_title("EMG-Signal mit gleitendem Fenster")
ax.set_xlabel("Zeit [s]")
ax.set_ylabel("Amplitude")
ax.legend()

print("🔍 Starte Verarbeitung...")

# === Verarbeitung
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

    if arduino:
        try:
            arduino.write(f"{prediction}\n".encode())
        except Exception as err:
            print("❌ Fehler beim Senden an Arduino:", err)

    # === Update Fenster-Position im Plot
    window_start = i - WINDOW_SIZE
    window_end = i
    window_box.remove()
    window_box = ax.axvspan(time_axis[window_start], time_axis[window_end], color='red', alpha=0.3)
    plt.pause(0.001)


plt.ioff()
plt.show()

if arduino:
    arduino.close()
