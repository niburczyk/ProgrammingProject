import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # Für eine bessere Visualisierung der Konfusionsmatrix
import joblib  # Für das Speichern und Laden des Modells

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

# Modell mit joblib speichern
model_filename = 'svm_model.pkl'
joblib.dump(svm_model, model_filename)
print(f"SVM-Modell wurde als {model_filename} gespeichert.")

# Modell mit joblib laden (optional)
# svm_model_loaded = joblib.load(model_filename)