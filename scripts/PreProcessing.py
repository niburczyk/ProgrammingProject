import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# === Filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# === Feature-Berechnungen
def calculate_mav(data):
    return np.mean(np.abs(data), axis=0)

def calculate_wl(data):
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0)

# === Parameter
base_folder_path = './sample'
fs = 2000  # Abtastfrequenz in Hz
lowcut = 1.25
highcut = 22.5
order = 4

remove_time = 3  # Sekunden vorne und hinten
remove_samples = remove_time * fs  # Anzahl Samples zum Entfernen

window_size = 250  # 250 Samples = 125 ms
step_size = 125    # 50% Überlappung = 62.5 ms pro Schritt

# === Daten-Container
features = []
labels = []

# === Verarbeitung
for condition_folder in os.listdir(base_folder_path):
    folder_path = os.path.join(base_folder_path, condition_folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.mat'):
                file_path = os.path.join(folder_path, filename)
                mat_data = loadmat(file_path)
                data = mat_data['data']

                # Bandpassfilter
                filtered = bandpass_filter(data, lowcut, highcut, fs, order)

                # Gekürzter Bereich
                trimmed = filtered[remove_samples:-remove_samples]
                total_len = trimmed.shape[0]

                # Heuristische Segmentierung
                open_front = trimmed[:7 * fs]
                movement = trimmed[7 * fs:17 * fs]
                open_back = trimmed[17 * fs:]

                # Bewegungstyp bestimmen
                if 'F' in condition_folder:
                    movement_label = 'Fist'
                elif 'P' in condition_folder:
                    movement_label = 'Pinch'
                elif 'O' in condition_folder:
                    movement_label = 'Open'
                else:
                    print(f"⚠️  Unbekannter Bewegungstyp in: {filename}")
                    continue

                # Fensterung + Feature-Extraktion
                def extract_features(data_segment, label):
                    for start in range(0, len(data_segment) - window_size + 1, step_size):
                        window = data_segment[start:start + window_size]
                        mav = calculate_mav(window)
                        wl = calculate_wl(window)
                        vec = np.concatenate((mav, wl))
                        features.append(vec)
                        labels.append(label)

                extract_features(open_front, 'Open')
                extract_features(movement, movement_label)
                extract_features(open_back, 'Open')

# === In DataFrame umwandeln
features = np.array(features)
labels = np.array(labels)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

df = pd.DataFrame(features)
num_channels = features.shape[1] // 2
df.columns = [f'MAV_{i+1}' for i in range(num_channels)] + [f'WL_{i+1}' for i in range(num_channels)]
df['label'] = encoded_labels

# === Speichern
df.to_csv('training_dataset_windowed.csv', index=False)
print("✅ Fenster-basiertes Dataset gespeichert als 'training_dataset_windowed.csv'.")
