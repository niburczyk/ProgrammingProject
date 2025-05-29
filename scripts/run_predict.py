import serial
import csv
import time
from datetime import datetime

PORT = "/dev/ttyUSB0"
BAUD = 9600

def get_filename():
    return f"data/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    ser.flush()

    recording = False
    data = []

    print("📡 Warte auf START/STOP Befehl über serielle Schnittstelle...")

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"📥 Empfangen: {line}")

            if line == "START":
                print("▶️ Aufnahme gestartet.")
                recording = True
                data = []
                ser.write(b"ACK_START\n")

            elif line == "STOP":
                if recording:
                    print("⏹️ Aufnahme gestoppt.")
                    filename = get_filename()
                    with open(filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['timestamp', 'value'])  # ggf. anpassen
                        for row in data:
                            writer.writerow(row)
                    print(f"💾 Gespeichert: {filename}")
                    recording = False
                    ser.write(b"ACK_STOP\n")

            elif recording:
                ts = datetime.now().isoformat()
                data.append([ts, line])

if __name__ == "__main__":
    main()
