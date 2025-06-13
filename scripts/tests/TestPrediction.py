import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
from scipy.signal import firwin, lfilter
import time

# === Parameter ===
WINDOW_SIZE = 500
STEP_SIZE = 500
DATA_FILE = './sample/emg_data.txt'

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

# === Filterfunktion ===
def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    return lfilter(taps, 1.0, data, axis=0)[numtaps:]

# === Feature-Berechnung ===
def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

# === EMG-Daten laden & umrechnen ===
raw_data = np.loadtxt(DATA_FILE).reshape(-1, 1)

# ADC → Volt → mV (Olimex Shield)
adc_resolution = 1023
v_ref = 3.0
gain_total = 2848
raw_data = (raw_data / adc_resolution) * v_ref
raw_data = (raw_data / gain_total) * 1e3  # in mV

n_samples, n_channels = raw_data.shape

# === Plot-Vorbereitung ===
plt.ion()
fig, ax = plt.subplots(figsize=(12, 4))
t = np.arange(len(raw_data))
lines = [ax.plot(t, raw_data[:, ch], label=f"Kanal {ch+1}")[0] for ch in range(n_channels)]

ax.set_title("EMG-Signal mit Sliding Window Vorhersage")
ax.set_xlabel("Samples")
ax.set_ylabel("mV")
ax.legend()

# Rotes Rechteck für Sliding-Window
y_min, y_max = ax.get_ylim()
window_rect = patches.Rectangle((0, y_min), WINDOW_SIZE, y_max - y_min,
                                linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
ax.add_patch(window_rect)

# Text für Vorhersage
text_prediction = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                          fontsize=14, verticalalignment='top',
                          bbox=dict(boxstyle="round", facecolor="white"))

# === Fenster iterieren ===
for i in range(0, len(raw_data) - WINDOW_SIZE, STEP_SIZE):
    window = raw_data[i:i + WINDOW_SIZE]
    filtered = bandpass_filter(window, lowcut, highcut, fs, numtaps)

    if filtered.shape[0] < WINDOW_SIZE - numtaps:
        continue

    mav = calculate_mav(filtered)
    wl = calculate_wl(filtered)
    features = np.concatenate((mav, wl)).reshape(1, -1)

    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)[0]
    class_label = label_encoder.inverse_transform([prediction])[0]

    # Rechteck verschieben
    window_rect.set_x(i)
    window_rect.set_height(y_max - y_min)
    window_rect.set_y(y_min)

    # Vorhersage aktualisieren
    text_prediction.set_text(f"Vorhersage: {class_label} ({prediction})")

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.05)

plt.ioff()
plt.show()
