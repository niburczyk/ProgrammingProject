import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# === Modell, Scaler und PCA laden ===
model = joblib.load('./model/svm_model_optimized.pkl')
scaler = joblib.load('./model/scaler.pkl')
pca = joblib.load('./model/pca_components.pkl')

# === Trainingsdaten laden ===
train_data = pd.read_csv('./sample/data/training_dataset_windowed.csv')
features_train = train_data.iloc[:, :-1].values
labels_train = train_data.iloc[:, -1].values

# === Trainingsdaten vorbereiten (Scaling + PCA) ===
features_train_scaled = scaler.transform(features_train)
features_train_pca = pca.transform(features_train_scaled)

# === Funktionen für Feature-Berechnung ===
def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

# === Rohdaten laden und umrechnen (TXT) ===
raw_data = np.loadtxt('./sample/emg_data.txt').reshape(-1, 1)

# ADC → Volt → mV (Olimex Shield)
adc_resolution = 1023
v_ref = 3.0
gain_total = 2848
raw_data = (raw_data / adc_resolution) * v_ref
raw_data = (raw_data / gain_total) * 1e3  # in mV

# === Bandpassfilter anwenden ===
fs = 2000
lowcut = 1.25
highcut = 22.5
filter_order = 101

def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    filtered = lfilter(taps, 1.0, data, axis=0)
    return filtered[numtaps:]  # Delay kompensieren

filtered_data = bandpass_filter(raw_data, lowcut, highcut, fs, filter_order)

# === Plot Roh- vs. Gefilterte Daten (Ausschnitt) ===
plt.figure(figsize=(12, 5))
plt.plot(raw_data[:4000], label="Rohdaten (mV)", alpha=0.6)
plt.plot(np.arange(len(filtered_data[:4000])) + filter_order, filtered_data[:4000], label="Gefiltert (mV)", linewidth=2)
plt.title("EMG TXT: Rohdaten vs. Gefilterte Daten")
plt.xlabel("Sample")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Feature-Extraktion (MAV, WL) ===
WINDOW_SIZE = 250
STEP_SIZE = 100

mav_new = []
wl_new = []

for i in range(WINDOW_SIZE, len(filtered_data), STEP_SIZE):
    window = filtered_data[i - WINDOW_SIZE:i]
    mav_new.append(calculate_mav(window)[0])
    wl_new.append(calculate_wl(window)[0])

mav_new = np.array(mav_new)
wl_new = np.array(wl_new)
features_new = np.vstack((mav_new, wl_new)).T

# === Neue Daten skalieren + PCA anwenden ===
features_new_scaled = scaler.transform(features_new)
features_new_pca = pca.transform(features_new_scaled)

# === SVM Entscheidungsebene vorbereiten ===
x_min, x_max = features_train_pca[:, 0].min() - 1, features_train_pca[:, 0].max() + 1
y_min, y_max = features_train_pca[:, 1].min() - 1, features_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points).reshape(xx.shape)

# === PCA-Plot mit neuen Daten ===
unique_labels = np.unique(labels_train)
colors = plt.cm.get_cmap('tab10', len(unique_labels))

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)

for i, label in enumerate(unique_labels):
    idx = labels_train == label
    plt.scatter(features_train_pca[idx, 0], features_train_pca[idx, 1],
                color=colors(i), label=f"Klasse {label}", edgecolor='k', alpha=0.7)

plt.scatter(features_new_pca[:, 0], features_new_pca[:, 1],
            color='black', label='Neue Daten (TXT)', marker='x', alpha=0.8)

plt.xlabel('PCA Komponente 1')
plt.ylabel('PCA Komponente 2')
plt.title('SVM mit Trainingsdaten & neuen TXT-EMG-Daten')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
