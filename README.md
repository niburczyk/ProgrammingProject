# ProgrammingProject: SVM zur Bestimmung von Handpositionen mittels EMG-Daten

Dieses Projekt beinhaltet den Quellcode und die Dokumentation zur Entwicklung einer Support Vector Machine (SVM), die Elektromyographie (EMG)-Daten des Unterarms verwendet, um Handpositionen zu bestimmen. Das finale Modell wird exportiert und kann auf einem Arduino verwendet werden.

## Inhaltsverzeichnis

- [Installation](#installation)
- [Datenbeschreibung](#datenbeschreibung)
- [Modellentwicklung](#modellentwicklung)
- [Modell-Export](#modell-export)
- [Verwendung auf dem Arduino](#verwendung-auf-dem-arduino)
- [Autoren](#autoren)

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

3. Installiere die notwendigen Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

## Datenbeschreibung

Die EMG-Daten stammen von Elektroden, die am Unterarm platziert sind und die elektrische Aktivität der Muskeln messen. Diese Daten werden verwendet, um verschiedene Handpositionen zu klassifizieren. Eine typische Datensatzstruktur könnte wie folgt aussehen:

- `.\sample`: in diesem Ordner befinden sich die vorweg klassifizierten Rohdaten in Unterordnern.
   - `.\sample\Condition-F`: In diesem Ordner befinden sich die Rohdaten der Kondition F, diese beschreibt Faust geschlossen (F = Fist) 
   - `.\sample\Condition-O`: In diesem Ordner befinden sich die Rohdaten der Kondition O, diese beschreibt Faust offen (O = Open) 
   - `.\sample\Condition-P`: In diesem Ordner befinden sich die Rohdaten der Kondition P, diese beschreibt den Pinzettengriff (P = Pinch) 
- `training_dataset.csv`: Eine CSV-Datei mit den entsprechenden Trainingsdaten, diese beinhaltet die features sowie die labels.

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

## Modell-Export

Das trainierte SVM-Modell wird in einem format exportiert, das auf einem Arduino verwendet werden kann. Hierzu wird die Bibliothek `sklearn-porter` verwendet, um das Modell in C-Code zu konvertieren.

```
pip install sklearn-porter
```

```python
from sklearn_porter import Porter

# Annahme: 'model' ist das trainierte SVM-Modell
porter = Porter(model, language='c')
output = porter.export(embed_data=True)

with open('svm_model.h', 'w') as file:
    file.write(output)
```

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