import numpy as np
from scipy.signal import firwin, lfilter
from scipy.io import loadmat
import matplotlib.pyplot as plt

# === Filterkonfiguration (wie im Live-System) ===
fs = 2000  # Abtastfrequenz in Hz
lowcut = 1.25
highcut = 22.5
numtaps = 101
nyquist = 0.5 * fs
taps = firwin(numtaps, [lowcut / nyquist, highcut / nyquist], pass_zero=False)

# === Funktionen ===

def read_mat_file(file_path):
    mat = loadmat(file_path)
    return mat['data']  # ggf. Schlüssel anpassen, z. B. ['EMG'] o.ä.

def bandpass_filter(data):
    # FIR-Filter ohne Abschneiden – vollständiges Signal
    return lfilter(taps, 1.0, data, axis=0)

def calculate_snr(signal, noise):
    signal_power = np.mean(np.square(signal / np.max(np.abs(signal))))
    noise_power = np.mean(np.square(noise / np.max(np.abs(noise))))
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

# === Daten laden ===
original_signal = read_mat_file(r'.\sample\Condition-P\P46.mat')

# Wenn Signal 2D ist (z. B. [samples, channels]), nimm z. B. nur Kanal 0
if original_signal.ndim > 1:
    original_signal = original_signal[:, 0]  # Kanalwahl ggf. anpassen

# === FIR-Filterung ===
filtered_signal = bandpass_filter(original_signal)

# === SNR-Berechnung ===
noise = original_signal - filtered_signal
snr = calculate_snr(filtered_signal, noise)

# === Ausgabe ===
print(f"✅ FIR-Filter angewendet: numtaps = {numtaps}, Lowcut = {lowcut}, Highcut = {highcut}")
print(f"📈 SNR = {snr:.2f} dB")

# === Plot (komplettes Signal!) ===
plt.figure(figsize=(16, 5))
plt.plot(original_signal, label='Originalsignal', alpha=0.6)
plt.plot(filtered_signal, label='Gefiltert (FIR)', linestyle='--', linewidth=1.2)
plt.title("Komplette Signalansicht – Original vs. FIR-gefiltert")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
