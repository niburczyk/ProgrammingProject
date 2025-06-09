import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import firwin, lfilter

# === Lade EMG-Daten aus TXT (ADC-Werte → Volt)===
def load_txt_emg(filename, adc_resolution=1023, v_ref=3.0, gain_total=2848):
    raw = np.loadtxt(filename).reshape(-1, 1)
    v_out = (raw / adc_resolution) * v_ref
    v_out = v_out / gain_total
    return v_out * 1e3

# === Lade EMG-Daten aus MAT (bereits Volt) ===
def load_mat_emg(filename, variable_name='data'):
    mat = loadmat(filename)
    if variable_name not in mat:
        raise KeyError(f"Variable '{variable_name}' nicht in MAT-Datei gefunden.")
    data = mat[variable_name]
    if data.ndim > 1:
        data = data[:, 0].reshape(-1,1)
    else:
        data = data.reshape(-1,1)
    return data

def bandpass_filter(data, lowcut=1.25, highcut=22.5, fs=2000, numtaps=101):
    nyq = 0.5 * fs
    taps = firwin(numtaps, [lowcut/nyq, highcut/nyq], pass_zero=False)
    filtered = lfilter(taps, 1.0, data, axis=0)
    return filtered[numtaps:]

def calculate_mav(sig):
    return np.mean(np.abs(sig), axis=0)

def calculate_wl(sig):
    return np.sum(np.abs(np.diff(sig, axis=0)), axis=0)

def extract_features(data, window_size=250, step_size=100):
    mavs, wls = [], []
    for start in range(0, len(data) - window_size, step_size):
        win = data[start:start + window_size]
        mavs.append(calculate_mav(win))
        wls.append(calculate_wl(win))
    return np.array(mavs).squeeze(), np.array(wls).squeeze()

def zscore(feature):
    return (feature - np.mean(feature)) / np.std(feature)

def compare_emg_datasets(txt_file, mat_file, mat_variable='data'):
    # Daten laden
    txt = load_txt_emg(txt_file)
    mat = load_mat_emg(mat_file, variable_name=mat_variable)
    print(f"Rohbereiche: TXT {txt.min():.2e} bis {txt.max():.2e} V | MAT {mat.min():.2e} bis {mat.max():.2e} V")

    # Filterung
    txt_f = bandpass_filter(txt)
    mat_f = bandpass_filter(mat)

    # Feature extraction
    txt_mav, txt_wl = extract_features(txt_f)
    mat_mav, mat_wl = extract_features(mat_f)

    # Z‑Score‑Normierung
    txt_m = zscore(txt_mav)
    mat_m = zscore(mat_mav)
    txt_w = zscore(txt_wl)
    mat_w = zscore(mat_wl)

    # Statistik nach Normierung
    print(f"\nNach Z‑Score – MAV: TXT µ≈{txt_m.mean():.2f}, σ≈{txt_m.std():.2f} | MAT µ≈{mat_m.mean():.2f}, σ≈{mat_m.std():.2f}")
    print(f"Nach Z‑Score – WL: TXT µ≈{txt_w.mean():.2f}, σ≈{txt_w.std():.2f} | MAT µ≈{mat_w.mean():.2f}, σ≈{mat_w.std():.2f}")

    # Histogrammplot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(txt_m, bins=30, alpha=0.7, label='TXT MAV (norm)')
    plt.hist(mat_m, bins=30, alpha=0.7, label='MAT MAV (norm)')
    plt.legend(); plt.title("MAV Z‑Score")
    plt.subplot(1,2,2)
    plt.hist(txt_w, bins=30, alpha=0.7, label='TXT WL (norm)')
    plt.hist(mat_w, bins=30, alpha=0.7, label='MAT WL (norm)')
    plt.legend(); plt.title("WL Z‑Score")
    plt.tight_layout(); plt.show()



# === Beispiel-Aufruf ===
compare_emg_datasets('./sample/emg_data.txt', './sample/Condition-F/F11_10.mat', 'data')
