import serial
import time
import os
import threading

# === Konfiguration ===
SERIAL_PORT = '/dev/ttyACM0'  # Passe ggf. an
BAUD_RATE = 230400
DATA_SAVE_DIR = os.path.expanduser('./data')
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

# === Globale Variablen ===
recording = False
recording_save = []
filename = f"raw_emg_data_{int(time.time()*1000)}.txt"
filepath = os.path.join(DATA_SAVE_DIR, filename)

# === Eingabe-Thread für START/STOP ===
def input_thread():
    global recording, recording_save
    print("Eingabe-Thread gestartet (START/STOP).")
    while True:
        cmd = input("Eingabe (START/STOP): ").strip().upper()
        if cmd == "START":
            recording = True
            recording_save = []
            print("Aufnahme gestartet.")
        elif cmd == "STOP":
            recording = False
            print("Aufnahme gestoppt.")
            save_to_file(recording_save)

# === Datei speichern ===
def save_to_file(data):
    if not data:
        print("Keine Daten zum Speichern.")
        return
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(','.join(map(str, row)) + '\n')
    print(f"Daten gespeichert in {filepath}")

# === Serielle Daten lesen ===
def serial_reader(arduino):
    print("Warte auf serielle Daten...")
    while True:
        try:
            line_bytes = arduino.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            parts = line.split(',')
            sample = [float(x) for x in parts]
            timestamp_ms = int(time.time() * 1000)
            if recording:
                recording_save.append([timestamp_ms] + sample)
        except Exception as e:
            print("Fehler beim Lesen:", e)

# === Hauptfunktion ===
def main():
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Arduino verbunden an {SERIAL_PORT}")
    except Exception as e:
        print(f"Verbindung fehlgeschlagen: {e}")
        return

    threading.Thread(target=input_thread, daemon=True).start()
    try:
        serial_reader(arduino)
    except KeyboardInterrupt:
        print("\nProgramm beendet durch Benutzer.")
    finally:
        if arduino and arduino.is_open:
            arduino.close()
        print("Serielle Verbindung geschlossen.")

if __name__ == "__main__":
    main()
