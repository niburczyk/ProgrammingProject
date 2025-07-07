import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Pfade ===
csv_file_path = './sample/data/training_dataset_windowed.csv'
model_filename = './model/svm_pipeline_model.pkl'

# === Daten laden ===
data = pd.read_csv(csv_file_path)
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# === Modell laden oder neu trainieren ===
if os.path.exists(model_filename):
    pipeline_model = joblib.load(model_filename)
    print(f"Gespeichertes Modell '{model_filename}' geladen.")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    y_pred = pipeline_model.predict(X_test)

    print(f"\nAccuracy (geladenes Modell): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
    plt.title('Confusion Matrix (Geladenes Modell)')
    plt.show()

else:
    # === Pipeline erstellen ===
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('svm', SVC(class_weight='balanced'))
    ])

    # === Parameter-Grid für GridSearch ===
    param_grid = [
        {
            'svm__kernel': ['rbf'],
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 0.01, 0.1],
            'svm__shrinking': [True]
        },
        {
            'svm__kernel': ['poly'],
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 0.01],
            'svm__degree': [2, 3],
            'svm__coef0': [0.0, 0.5],
            'svm__shrinking': [True]
        },
        {
            'svm__kernel': ['linear'],
            'svm__C': [0.01, 0.1, 1, 10],
            'svm__shrinking': [True]
        }
    ]

    # === Trainings-/Test-Split ===
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # === GridSearchCV ===
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    print("Starte GridSearchCV ...")
    grid_search.fit(X_train, y_train)
    print("GridSearch abgeschlossen.")

    print(f"\nBestes Ergebnis: {grid_search.best_score_:.4f}")
    print(f"Beste Parameter: {grid_search.best_params_}")

    # === Bestes Modell evaluieren ===
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    print(f"\nAccuracy (Testdaten): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # === Confusion-Matrix ===
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
    plt.title('Confusion Matrix')
    plt.show()

    # === Modell speichern ===
    joblib.dump(best_pipeline, model_filename)
    print(f"Pipeline-Modell als '{model_filename}' gespeichert.")
    
    # Nach erfolgreichem GridSearch
    best_pipeline = grid_search.best_estimator_

    # Einzelne Komponenten extrahieren
    scaler = best_pipeline.named_steps['scaler']
    pca = best_pipeline.named_steps['pca']
    svm_model = best_pipeline.named_steps['svm']

    # Einzelkomponenten speichern
    joblib.dump(scaler, './model/scaler.pkl')
    joblib.dump(pca, './model/pca_components.pkl')
    joblib.dump(svm_model, './model/svm_model_optimized.pkl')

    print("✅ Alle Komponenten einzeln gespeichert.")
