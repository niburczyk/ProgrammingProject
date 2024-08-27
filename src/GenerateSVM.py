import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn_porter import Porter  # Zum Exportieren des Modells
import matplotlib.pyplot as plt
import seaborn as sns  # Für eine bessere Visualisierung der Konfusionsmatrix

# Pfad zur CSV-Datei
csv_file_path = './training_dataset.csv'

# Daten einlesen
data = pd.read_csv(csv_file_path)

# Die letzte Spalte ist das Label
features = data.iloc[:, :-1].values  # Alle Spalten außer der letzten
labels = data.iloc[:, -1].values    # Die letzte Spalte

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# SVM-Modell trainieren
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten
y_pred = svm_model.predict(X_test)

# Ergebnisse auswerten
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Berechnung der Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Konfusionsmatrix visualisieren
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(labels), yticklabels=set(labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Modell in eine .h-Datei exportieren
porter = Porter(svm_model, language='c')
output = porter.export(embed_data=True)

# Speichern in eine .h Datei
header_filename = 'svm_model.h'
with open(header_filename, 'w') as f:
    f.write(output)
print(f"SVM-Modell wurde als {header_filename} exportiert.")