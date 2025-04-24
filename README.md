# ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten

Dieses Projekt beinhaltet den Quellcode und die Dokumentation zur Entwicklung einer Support Vector Machine (SVM), die Elektromyographie (EMG)-Daten des Unterarms verwendet, um Handpositionen zu bestimmen. Das finale Modell wird exportiert und kann auf einem Arduino verwendet werden.

## Inhaltsverzeichnis

- [ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten](#programmingproject-svm-zur-bestimmung-von-handpositionen-mittels-emg-daten)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Projektstruktur](#projektstruktur)
  - [Installation](#installation)
  - [Datenbeschreibung](#datenbeschreibung)
  - [Modellentwicklung](#modellentwicklung)
  - [Modell-Speicherung](#modell-speicherung)
  - [Verwendung auf dem Arduino](#verwendung-auf-dem-arduino)
  - [Autoren](#autoren)

## Projektstruktur
.
├── docs/                       # Arc42-Dokumentation
│   ├── architecture.md         # Systemarchitektur
│   └── decisions.md            # Entwurfsentscheidungen
├── model/
│   └── svm_model_optimized.pkl # Serialisiertes SVM-Modell
├── sample/                     # Roh-EMG-Daten
│   ├── Condition-F/            # Faust geschlossen
│   ├── Condition-O/            # Hand offen  
│   └── Condition-P/            # Pinzettengriff
├── data/
│   └── training_dataset.csv    # Aufbereiteter Trainingsdatensatz
├── scripts/                    # Python-Skripte
│   ├── GenerateSVM.py          # Haupttrainingsskript
│   ├── PreProcessing.py        # Datenvorverarbeitung
│   └── ConvertSvmToHeader.py   # Modell-Export für Arduino
├── include/
│   └── svm_predict_poly.cpp    # Exportierte SVM-Implementierung
├── src/                        # Hauptanwendungscode
├── tests/                      # Unit-Tests
└──

## Installation

Um das Projekt lokal auszuführen, folge diesen Schritten:

1. Klone das Repository:
   ```bash
   git clone https://github.com/niburczyk/ProgrammingProject.git
   cd ProgrammingProject
   ```

2. Erstelle eine virtuelle Umgebung und aktiviere sie:
   ```bash
   python -m venv env
   source env/bin/activate  # Auf Windows: env\Scripts\activate
   ```

## Datenbeschreibung

Die EMG-Daten stammen von Elektroden, die am Unterarm platziert sind und die elektrische Aktivität der Muskeln messen. Diese Daten werden verwendet, um verschiedene Handpositionen zu klassifizieren. Eine typische Datensatzstruktur könnte wie folgt aussehen:

- `.\sample`: in diesem Ordner befinden sich die vorweg klassifizierten Rohdaten in Unterordnern.
   - `.\sample\Condition-F`: In diesem Ordner befinden sich die Rohdaten der Kondition F, diese beschreibt Faust geschlossen (F = Fist) 
   - `.\sample\Condition-O`: In diesem Ordner befinden sich die Rohdaten der Kondition O, diese beschreibt Faust offen (O = Open) 
   - `.\sample\Condition-P`: In diesem Ordner befinden sich die Rohdaten der Kondition P, diese beschreibt den Pinzettengriff (P = Pinch) 
- `.\data`: in diesem Order befindet sich die .csv welche als Datensatz verwendet wird.
|  - `training_dataset.csv`: Eine CSV-Datei mit den entsprechenden Trainingsdaten, diese beinhaltet die features sowie die labels.

## Modellentwicklung

Der Prozess der Modellentwicklung umfasst die folgenden Schritte:

1. **Datenvorverarbeitung**:
   - Normalisierung der EMG-Daten
   - Segmentierung der Daten
   - Bandpassfilterung der Daten
      - Ermittlung der optimalen Bandpasseinstellung

2. **Merkmalextraktion**:
   - Berechnung von statistischen Merkmalen (z.B. Mittelwert, Varianz) für jedes Zeitfenster

3. **Modelltraining**:
   - Training der SVM mit den extrahierten Merkmalen und Labels

4. **Modellbewertung**:
   - Bewertung des Modells anhand von Metriken wie Genauigkeit, Präzision und Recall

Der vollständige Code für die Modellentwicklung befindet sich in der Datei `GenerateSVM.py`.

## Modell-Speicherung
Das trainierte SVM-Modell wird mit der Bibliothek `joblib` in einer Datei gespeichert, die zur weiteren Verarbeitung oder in anderen Python-Projekten verwendet werden kann.

```bash
pip install joblib
```
```python
import joblib

model_filename = 'svm_model.pkl'
joblib.dump(svm_model, model_filename)
````

## Verwendung auf dem Arduino

Nachdem das Modell in C-Code konvertiert wurde, kann es auf einem Arduino verwendet werden. Das ertselle Modell (`svm_model.h`) wird in das Arduino-Projekt eingebunden und kann zur Klassifizierung der Echtzeit-EMG-Daten genutzt werden.

Ein Beispiel-Sketch für den Arduino könnte wie folgt aussehen:

```cpp
#include "svm_model.h"

void setup() {
  Serial.begin(9600);
}

void loop() {
  // EMG-Daten erfassen
  float emg_data[NUM_FEATURES];
  // ... Erfassen der Daten und Befüllen von emg_data

  // Klassifikation
  int hand_position = predict(emg_data);

  // Ausgabe der Handposition
  Serial.println(hand_position);
}
```

## Autoren

- **Niklas Burczyk** - [GitHub-Profil](https://github.com/niburczyk)
---

Viel Erfolg mit deinem Projekt! Wenn du Fragen hast, zögere nicht, ein Issue zu eröffnen oder mich direkt zu kontaktieren.