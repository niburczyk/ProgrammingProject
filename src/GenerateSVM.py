import os  # Für Dateiprüfungen
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Für das Speichern und Laden des Modells
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Pfad zur CSV-Datei
csv_file_path = './training_dataset.csv'

# Pfad zur gespeicherten Modelldatei
model_filename = 'svm_model_optimized.pkl'

# Prüfen, ob das Modell bereits gespeichert wurde
if os.path.exists(model_filename):
    # Modell laden
    svm_model_loaded = joblib.load(model_filename)
    print(f"Gespeichertes Modell '{model_filename}' geladen.")

    # Optional: Testdaten erneut einlesen, um Ergebnisse zu überprüfen
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :-1].values  # Alle Spalten außer der letzten
    labels = data.iloc[:, -1].values    # Die letzte Spalte

    # Standardisierung der Features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Optional: PCA zur Reduzierung der Dimensionen (falls notwendig)
    pca = PCA(n_components=2)  # Wir reduzieren auf 2 Dimensionen für die Visualisierung
    features = pca.fit_transform(features)

    # Aufteilen der Daten in Trainings- und Testsets
    _, X_test, _, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Vorhersagen auf den Testdaten mit dem geladenen Modell
    y_pred = svm_model_loaded.predict(X_test)

    # Ergebnisse auswerten
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Konfusionsmatrix visualisieren
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(labels), yticklabels=set(labels))
    plt.title('Confusion Matrix (Geladenes Modell)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

else:
    # Daten einlesen
    data = pd.read_csv(csv_file_path)

    # Die letzte Spalte ist das Label
    features = data.iloc[:, :-1].values  # Alle Spalten außer der letzten
    labels = data.iloc[:, -1].values    # Die letzte Spalte

    # Standardisierung der Features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Optional: PCA zur Reduzierung der Dimensionen (falls notwendig)
    pca = PCA(n_components=2)  # Wir reduzieren auf 2 Dimensionen für die Visualisierung
    features = pca.fit_transform(features)

    # Aufteilen der Daten in Trainings- und Testsets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Hyperparameter-Optimierung mit GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularisierungsparameter
        'gamma': ['scale', 'auto', 0.1, 1, 10],  # Einflussbereich des RBF-Kernels
        'kernel': ['rbf', 'poly'],  # Wählen zwischen RBF und Polynomial
        'degree': [3, 4, 5, 6]  # Nur für den Poly-Kernel relevant
    }

    # GridSearchCV für die Auswahl der besten Hyperparameter
    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Beste Parameter und Modell
    print(f"Beste Parameter: {grid_search.best_params_}")
    svm_model = grid_search.best_estimator_

    # Vorhersagen auf den Testdaten
    y_pred = svm_model.predict(X_test)

    # Ergebnisse auswerten
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Konfusionsmatrix visualisieren
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(labels), yticklabels=set(labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Modell mit joblib speichern
    joblib.dump(svm_model, model_filename)
    print(f"SVM-Modell wurde als {model_filename} gespeichert.")
