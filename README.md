# ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten

Dieses Projekt enthält den Quellcode und die Dokumentation zur Entwicklung einer Support Vector Machine (SVM), die Elektromyographie-(EMG)-Daten des Unterarms verwendet, um Handpositionen zu bestimmen. Das finale Modell wird exportiert und kann mit einem Arduino Uno sowie einem Raspberry Pi genutzt werden.

## Inhaltsverzeichnis

- [ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten](#programmingproject-svm-zur-bestimmung-von-handpositionen-mittels-emg-daten)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Installation](#installation)
  - [Datenbeschreibung](#datenbeschreibung)
  - [Modellentwicklung](#modellentwicklung)
  - [Modell-Speicherung](#modell-speicherung)
  - [Verwendung auf dem Arduino](#verwendung-auf-dem-arduino)
  - [Autoren](#autoren)

## Installation

Das Projekt basiert darauf, dass ein Raspberry Pi mit einem Arduino über die serielle Schnittstelle verbunden ist. Der Zugriff auf den Raspberry Pi erfolgt z. B. via SSH. Die folgenden Schritte richten die Umgebung ein:

1. Repository klonen:  
   ```bash
   git clone https://github.com/niburczyk/ProgrammingProject.git
   cd ProgrammingProject
   ```

2. In das Projektverzeichnis wechseln:  
   ```bash
   cd ProgrammingProject/
   ```

3. Ausführungsrechte für die Skripte vergeben:  
   ```bash
   chmod +x scripts/*.sh
   ```

4. Docker im Hintergrund starten:  
   ```bash
   docker compose up --build -d
   ```

5. Überprüfen, ob der Container läuft:  
   ```bash
   docker ps
   ```

6. Seriellen Monitor des Arduinos mit `minicom` öffnen:  
   ```bash
   minicom -b 9600 -D /dev/ttyACM0
   ```

7. Echo aktivieren, um Befehle einzugeben:  
   Drücke `STRG + A`, dann `E` und bestätige mit `ENTER`.

8. Aufnahme oder Vorhersage starten bzw. stoppen:  
   ```text
   START<ENTER>
   STOP<ENTER>
   ```

## Datenbeschreibung

Die EMG-Daten stammen von Elektroden, die am Unterarm angebracht sind und die elektrische Muskelaktivität messen. Diese Daten dienen zur Klassifikation verschiedener Handpositionen.

Struktur der Datensätze:

- `./sample`: Ordner mit vorab klassifizierten Rohdaten in Unterordnern  
  - `./sample/Condition-F`: Rohdaten für geschlossene Faust (F = Fist)  
  - `./sample/Condition-O`: Rohdaten für offene Faust (O = Open)  
  - `./sample/Condition-P`: Rohdaten für Pinzettengriff (P = Pinch)  

- `./data`: Enthält die .csv-Datei mit dem Datensatz  
  - `training_dataset_windowed.csv`: CSV mit Trainingsdaten inklusive Features und Labels  

## Modellentwicklung

Die Modellentwicklung umfasst folgende Schritte:

1. **Datenvorverarbeitung:**  
   - Normalisierung der EMG-Daten  
   - Segmentierung in Zeitfenster  
   - Bandpassfilterung (inklusive Optimierung der Filtereinstellungen)  

2. **Merkmalextraktion:**  
   - Berechnung statistischer Merkmale (z. B. Mittelwert, Varianz) pro Fenster  

3. **Modelltraining:**  
   - Training der SVM mit den extrahierten Merkmalen und den Labels  

4. **Modellbewertung:**  
   - Evaluierung anhand von Metriken wie Genauigkeit, Präzision und Recall  

Der komplette Code zur Modellentwicklung befindet sich in `GenerateModel.py`.

## Modell-Speicherung

Das trainierte SVM-Modell wird mit `joblib` gespeichert, um es in anderen Python-Projekten oder für die spätere Verwendung laden zu können.

Installation von `joblib`:  
```bash
pip install joblib
```

Beispiel zum Speichern des Modells:  
```python
import joblib

model_filename = 'svm_model.pkl'
joblib.dump(svm_model, model_filename)
```

## Verwendung auf dem Arduino

Nach dem Training kann das Modell auf dem Raspberry Pi zur Vorhersage der EMG-Daten verwendet werden, die vom Arduino geliefert werden.

Die Datei `emg_transmitter` sammelt die EMG-Daten und sendet sie an den Raspberry Pi, wo die Vorhersage durchgeführt wird. Die Ergebnisse werden über die serielle Schnittstelle wieder an den Arduino zurückgegeben.

## Autoren

- **Niklas Burczyk** – [GitHub-Profil](https://github.com/niburczyk)

---

Viel Erfolg mit deinem Projekt! Bei Fragen kannst du gerne ein Issue eröffnen oder mich direkt kontaktieren.
