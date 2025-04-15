import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Funktion zum Einlesen der .mat-Datei
def read_mat_file(filename):
    data = loadmat(filename)
    return data

# Funktion zum Anwenden eines Bandpassfilters
def bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

# Funktion zur Berechnung des Mean Absolute Value (MAV)
def calculate_mav(data):
    return np.mean(np.abs(data), axis=0)

# Funktion zur Berechnung der Waveform Length (WL)
def calculate_wl(data):
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0)

# Funktion zum Darstellen des Signals
def plot_signal(original_data, filtered_data, title):
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(original_data, label='Original Signal')
    plt.title(f'{title} - Original Signal')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(filtered_data, label='Filtered Signal', color='orange')
    plt.title(f'{title} - Filtered Signal')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Hauptskript
if __name__ == "__main__":
    # Parameter
    base_folder_path = './sample'  # Basispfad zum Ordner mit den Unterordnern
    lowcut = 1.25  # untere Grenzfrequenz des Bandpassfilters
    highcut = 22.5  # obere Grenzfrequenz des Bandpassfilters
    fs = 2000  # Abtastfrequenz in Hz
    order = 4  # Filterordnung

    # Zeit in Sekunden, die entfernt werden soll
    remove_time = 3
    remove_samples = remove_time * fs  # Anzahl der zu entfernenden Abtastwerte

    # Listen für Features und Labels
    features = []
    labels = []

    # Durch alle Unterordner im Basisordner iterieren
    for condition_folder in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, condition_folder)
        if os.path.isdir(folder_path):  # Prüfen, ob es ein Verzeichnis ist
            # Durch alle Dateien im Unterordner iterieren
            for filename in os.listdir(folder_path):
                if filename.endswith('.mat'):
                    file_path = os.path.join(folder_path, filename)

                    # Einlesen der .mat-Datei
                    mat_data = read_mat_file(file_path)

                    # Angenommen, die .mat-Datei enthält eine Variable namens 'data'
                    data = mat_data['data']

                    # Anwenden des Bandpassfilters auf jeden Vektor (jede Spalte)
                    filtered_data = bandpass_filter(data, lowcut, highcut, fs, order)

                    # Kürzen der Werte 3 Sekunden vorne und 3 Sekunden hinten
                    filtered_data = filtered_data[remove_samples:-remove_samples]

                    # Berechnen der Merkmale (MAV und WL) für jeden Kanal
                    mav_vector = calculate_mav(filtered_data)
                    wl_vector = calculate_wl(filtered_data)

                    # Kombinieren der Merkmalsvektoren
                    feature_vector = np.concatenate((mav_vector, wl_vector))

                    # Bestimmen der Bewegung aus dem Dateinamen oder Ordnernamen
                    if 'F' in condition_folder:
                        label = 'Fist'
                    elif 'O' in condition_folder:
                        label = 'Open'
                    elif 'P' in condition_folder:
                        label = 'Pinch'
                    else:
                        print(f"Unbekannter Bewegungstyp in {filename}")
                        continue

                    # Hinzufügen der Features und Labels zur Liste
                    features.append(feature_vector)
                    labels.append(label)

    # Konvertieren der Listen in NumPy-Arrays
    features = np.array(features)
    labels = np.array(labels)

    # Kodierung der Labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Erstellen eines DataFrames für die Features und Labels
    df = pd.DataFrame(features)
    # Benennen der Spalten für bessere Lesbarkeit
    num_channels = len(mav_vector)  # Anzahl der EMG-Kanäle
    column_names = [f'MAV_{i+1}' for i in range(num_channels)] + [f'WL_{i+1}' for i in range(num_channels)]
    df.columns = column_names
    df['label'] = encoded_labels

    # Speichern der Features und Labels in eine CSV-Datei
    df.to_csv('training_dataset.csv', index=False)
    print("Training dataset with MAV and WL features saved as 'training_dataset.csv'.")

    # Optional: Beispielhafte Darstellung eines Signals
    if len(features) > 0:
        example_filename = os.listdir(folder_path)[0]
        example_file_path = os.path.join(folder_path, example_filename)
        mat_data = read_mat_file(example_file_path)
        data = mat_data['data']
        filtered_data = bandpass_filter(data, lowcut, highcut, fs, order)
        filtered_data = filtered_data[remove_samples:-remove_samples]
        plot_signal(data, filtered_data, os.path.splitext(example_filename)[0])