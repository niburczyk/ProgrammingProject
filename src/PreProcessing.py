import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

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

# Funktion zum Berechnen des Mittelwerts
def calculate_mean(filtered_data):
    return np.mean(filtered_data, axis=0)

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
    folder_path = '.\sample\Condition_F'  # Pfad zum Ordner mit den .mat-Dateien
    lowcut = 0.5  # untere Grenzfrequenz des Bandpassfilters
    highcut = 10.0  # obere Grenzfrequenz des Bandpassfilters
    fs = 2000  # Abtastfrequenz in Hz
    order = 1  # Filterordnung

    # Durch alle Dateien im Ordner iterieren
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(folder_path, filename)

            # Einlesen der .mat-Datei
            mat_data = read_mat_file(file_path)

            # Angenommen, die .mat-Datei enthält eine Variable namens 'data'
            data = mat_data['data']

            # Anwenden des Bandpassfilters auf jeden Vektor (jede Spalte)
            filtered_data = bandpass_filter(data, lowcut, highcut, fs, order)

            # Berechnen des Mittelwerts für jeden Vektor
            mean_vector = calculate_mean(filtered_data)

            # Optional: Den Mittelwertvektor speichern oder weiter verarbeiten
            print(f"Mean Vector for {filename}:", mean_vector)

            # Optional: Den Mittelwertvektor in eine Datei speichern
            output_filename = f'mean_vector_{os.path.splitext(filename)[0]}.npy'
            #np.save(os.path.join(folder_path, output_filename), mean_vector)

            # Darstellen des Original- und des gefilterten Signals
            plot_signal(data, filtered_data, os.path.splitext(filename)[0])
