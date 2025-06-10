# ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten

Dieses Projekt enthält den Quellcode und die Dokumentation zur Entwicklung einer Support Vector Machine (SVM), die Elektromyographie-(EMG)-Daten des Unterarms verwendet, um Handpositionen zu bestimmen. Das finale Modell wird exportiert und kann mit einem Arduino Uno sowie einem Raspberry Pi genutzt werden.

## Inhaltsverzeichnis

- [ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten](#programmingproject-svm-zur-bestimmung-von-handpositionen-mittels-emg-daten)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Installation](#installation)
  - [Speicherung der Daten](#speicherung-der-daten)
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
   ```

2. In das Projektverzeichnis wechseln:  
   ```bash
   cd ProgrammingProject/
   ```

3. Ausführungsrechte für die Skripte vergeben:  
   ```bash
   chmod +x scripts/*.sh
   ```

4. Docker installation:  
   ```bash
   docker compose build
   ```   
   
5. Docker starten:  
   ```bash
   docker compose run --rm predictor
   ```

6. Aufnahme oder Vorhersage starten bzw. stoppen:  
   ```text
   START - Starten
   STOP - Beenden
   ```

## Speicherung der Daten

Die Daten werden im im Ordner `./data` als .txt Daten gespeichert. Sowohl die Prediction als auch die Aufgenommenen Daten werden gespeichert und abgelegt. Um diese über ssh herunterladen zu können, kann der Port 8080 genutzt werden. Dafür muss zunächst zu den Daten navigiert werden auf dem Raspberry Pi. Danach muss der Port freigegeben werden:
```python
python3 -m http.server 8080
```
Darauf hin ist sidn die Daten im Netzwerk unter der `IP-Adresse:8080` abrufbar. Sollten die Daten über diese Schnittstelle nicht abrufbar sein, so werden die Daten auch lokal auf dem Raspberry Pi gespeichert und "verwart".

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

Die Datei `emg_transmitter` sammelt die EMG-Daten und sendet sie an den Raspberry Pi, wo die Vorhersage durchgeführt wird. Die Ergebnisse werden ausgegeben. Dies kann kann mit `START` bzw. `STOP`gesteuert werden.

## Autoren

- **Niklas Burczyk** – [GitHub-Profil](https://github.com/niburczyk)

---

Viel Erfolg mit deinem Projekt! Bei Fragen kannst du gerne ein Issue eröffnen oder mich direkt kontaktieren.
