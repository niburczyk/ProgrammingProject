import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from PreProcessing import read_mat_file

# Beispielhafte Funktion zur Signalfilterung
def bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

# Funktion zur Berechnung des Signal-Rausch-Verhältnisses (SNR)
def calculate_snr(signal, noise):
    signal_power = np.mean(np.square(signal / np.max(np.abs(signal))))
    noise_power = np.mean(np.square(noise / np.max(np.abs(noise))))
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

# Beispielhafte Daten (Ersetze durch deine Daten)
fs = 2000  # Abtastfrequenz in Hz
time = np.arange(0, 10, 1/fs)
mat_data = read_mat_file(r'.\sample\Condition-F\F0_10.mat')
original_signal = mat_data['data']

# Suche nach optimalen Filterparametern
best_snr = -np.inf
best_params = None

# Grid Search über verschiedene Parameter
for lowcut in [0.5, 1.0, 2.0]:
    for highcut in [15.0, 20.0, 30.0]:
        for order in [2, 3, 4]:
            filtered_signal = bandpass_filter(original_signal, lowcut, highcut, fs, order)
            
            # Signal-Rausch-Verhältnis (SNR) berechnen
            snr = calculate_snr(filtered_signal, original_signal - filtered_signal)
            
            # Überprüfen, ob die aktuelle Kombination die beste ist
            if snr > best_snr:
                best_snr = snr
                best_params = (lowcut, highcut, order)

# Beste Parameter ausgeben
print(f"Beste Parameterkombination: Lowcut = {best_params[0]}, Highcut = {best_params[1]}, Order = {best_params[2]} mit SNR = {best_snr:.2f} dB")
