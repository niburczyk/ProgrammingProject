import serial
import time
import os
from scipy.io import savemat
import numpy as np

# === Einstellungen ===
SERIAL_PORT = 'COM3'  # z.B. 'COM3' unter Windows oder '/dev/ttyUSB0' unter Linux
BAUD_RATE = 230400
SAMPLING_RATE = 2000  # Hz
RECORD_SECONDS = 30
SAMPLES_PER_FILE = SAMPLING_RATE * RECORD_SECONDS

OUTPUT_DIR = r'C:\Users\Niklas\sciebo - Burczyk, Niklas (nibur001@fh-dortmund.de)@fh-dortmund.sciebo.de2\Master\Semester 1\PA\ProgrammingProject\sample\Condition-P'
START_FILE_NUM = 31 # Startzahl festlegen
FILE_SUFFIX = '.mat' # Suffix festlegen
FILE_PREFIX = 'P' # Anpassen 

# === ADC-Konstanten ===
ADC_RESOLUTION = 1023
V_REF = 3.0         # Referenzspannung in V
GAIN_TOTAL = 2848   # Gesamtverstärkung

# === Ordner anlegen ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

def open_serial_port():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    return ser


def record_data(ser):
    data = []
    while len(data) < SAMPLES_PER_FILE:
        line = ser.readline()
        if not line:
            continue
        try:
            value = float(line.decode().strip())
            data.append(value)
            if len(data) % SAMPLING_RATE == 0:
                sec = len(data) // SAMPLING_RATE
                print(f"   Aufnahmezeit: {sec} Sekunden", end='\r')
        except ValueError:
            continue
    print()

    raw_data = np.array(data).reshape(-1, 1)  # Spaltenvektor

    # === Umrechnung in mV ===
    raw_data = (raw_data / ADC_RESOLUTION) * V_REF
    raw_data = raw_data / GAIN_TOTAL
    raw_data = raw_data * 1e3  # in mV

    return raw_data

def main():
    file_num = START_FILE_NUM

    while True:
        input(f"\n🔁 Drücke [Enter], um mit der nächsten Aufnahme zu starten ({FILE_PREFIX}{file_num}{FILE_SUFFIX}) …")

        print(f"📡 Starte Aufnahme: 30 Sekunden ({SAMPLES_PER_FILE} Samples)...")
        data = record_data(ser)

        filename = f'{FILE_PREFIX}{file_num}{FILE_SUFFIX}'
        filepath = os.path.join(OUTPUT_DIR, filename)
        savemat(filepath, {'data': data})

        print(f"💾 Gespeichert: {filename} mit {data.shape[0]} Werten.")
        file_num += 1

if __name__ == '__main__':
    try:
        ser = open_serial_port()
        main()
    except KeyboardInterrupt:
        print("\n🚫 Aufnahme abgebrochen.")
    finally:
        if 'ser' in globals() and ser.is_open:
            ser.close()
        print("🔌 Serieller Port geschlossen.")
