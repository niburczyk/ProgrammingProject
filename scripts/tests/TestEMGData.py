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

# === Features auf Trainingsdaten skalieren und mit PCA transformieren ===
features_train_scaled = scaler.transform(features_train)
features_train_pca = pca.transform(features_train_scaled)

# === Funktionen für Features ===
def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

# === Neue Rohdaten laden und vorverarbeiten ===
raw_data = np.loadtxt('./sample/emg_data.txt').reshape(-1, 1)

# Umrechnung
adc_resolution = 1023
v_ref = 3.0
gain_total = 2848
raw_data = (raw_data / adc_resolution) * v_ref
raw_data = raw_data / gain_total
raw_data = raw_data * 1e3

# Bandpassfilter Parameter
fs = 2000
lowcut = 1.25
highcut = 22.5
order = 4

def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    filtered = lfilter(taps, 1.0, data, axis=0)
    return filtered[numtaps:]

filtered_data = bandpass_filter(raw_data, lowcut, highcut, fs, order)

# === MAV und WL aus gefilterten Daten berechnen ===
WINDOW_SIZE = 250
STEP_SIZE = 100

mav_new = []
wl_new = []

for i in range(WINDOW_SIZE, len(filtered_data), STEP_SIZE):
    window_np = filtered_data[i - WINDOW_SIZE:i]
    mav_new.append(calculate_mav(window_np)[0])
    wl_new.append(calculate_wl(window_np)[0])

mav_new = np.array(mav_new)
wl_new = np.array(wl_new)

# === Neue Features in die gleiche Form bringen wie Trainingsdaten (2D) ===
features_new = np.vstack((mav_new, wl_new)).T

# Skalieren + PCA auf neue Daten anwenden
features_new_scaled = scaler.transform(features_new)
features_new_pca = pca.transform(features_new_scaled)

# === Grenzen für Plot (auf PCA-dimensionen) ===
x_min, x_max = features_train_pca[:, 0].min() - 1, features_train_pca[:, 0].max() + 1
y_min, y_max = features_train_pca[:, 1].min() - 1, features_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Gitterpunkte für Vorhersage
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

# Farben für Klassen
unique_labels = np.unique(labels_train)
colors = plt.cm.get_cmap('tab10', len(unique_labels))

# === Plot ===
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)

# Trainingspunkte (PCA transformiert) plotten
for i, label in enumerate(unique_labels):
    idx = labels_train == label
    plt.scatter(features_train_pca[idx, 0], features_train_pca[idx, 1],
                color=colors(i), label=f"Klasse {label}", edgecolor='k', alpha=0.7)

# Neue Datenpunkte plotten
plt.scatter(features_new_pca[:, 0], features_new_pca[:, 1], color='black', label='Neue Daten', alpha=0.7, marker='x')

plt.xlabel('PCA Komponente 1')
plt.ylabel('PCA Komponente 2')
plt.title('SVM Entscheidungsflächen mit Trainings- und neuen Daten')
plt.legend()
plt.grid(True)
plt.show()
