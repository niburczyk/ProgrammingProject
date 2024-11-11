import scipy.io as sio
import numpy as np
import os

# Dateipfad zu den .mat-Dateien (Anpassen!)
input_dir = './sample/Condition-O'
output_dir = './sample/Condition-O'

# Abtastfrequenz und Segmentdauer
sampling_rate = 2000  # Hz
segment_duration = 10  # Sekunden
segment_samples = segment_duration * sampling_rate  # Anzahl der Datenpunkte pro Segment

# Durchlaufen aller .mat-Dateien im Verzeichnis, die auf _long enden
for filename in os.listdir(input_dir):
    if filename.endswith('_long.mat'):
        filepath = os.path.join(input_dir, filename)
        
        # Laden der .mat-Datei
        mat_data = sio.loadmat(filepath)
        
        # Annahme: Die Daten sind in einer Variablen namens 'data' gespeichert
        data = mat_data.get('data')
        
        if data is None:
            print(f"Keine Daten in {filename} gefunden")
            continue
        
        # Segmentierung der Daten
        num_segments = data.shape[0] // segment_samples
        
        for i in range(num_segments):
            segment_data = data[i * segment_samples: (i + 1) * segment_samples]
            
            # Neue .mat-Datei speichern, mit passendem Segment-Index
            segment_filename = f"{os.path.splitext(filename)[0]}_segment_{i + 1}.mat"
            segment_filepath = os.path.join(output_dir, segment_filename)
            
            # Speichern des Segments in einer neuen .mat-Datei
            sio.savemat(segment_filepath, {'data': segment_data})
            print(f"Gespeichert: {segment_filepath}")

print("Segmentierung abgeschlossen!")
