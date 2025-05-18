import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# === Datei- und Modellpfade ===
csv_file_path = './sample/data/training_dataset.csv'
model_filename = './model/svm_model_optimized.pkl'
scaler_filename = './model/scaler.pkl'
pca_filename = './model/pca_components.pkl'

# === Prüfen, ob Modell bereits existiert ===
if os.path.exists(model_filename):
    # Modell & Scaler laden
    svm_model_loaded = joblib.load(model_filename)
    scaler_loaded = joblib.load(scaler_filename) if os.path.exists(scaler_filename) else None
    print(f"Gespeichertes Modell '{model_filename}' und Scaler '{scaler_filename}' geladen.")

    # Daten laden
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Skalierung
    if scaler_loaded:
        features = scaler_loaded.transform(features)

    # PCA oder LDA
    use_lda_for_eval = False
    if use_lda_for_eval:
        lda = LDA(n_components=min(len(set(labels)) - 1, features.shape[1]))
        features = lda.fit_transform(features, labels)
    else:
        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

    # Testdaten-Split
    _, X_test, _, y_test = train_test_split(features res, labels, test_size=0.3, random_state=42)

    # Modell-Evaluation auf Testdaten
    y_pred = svm_model_loaded.predict(X_test)
    print(f"\nAccuracy (geladenes Modell): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Confusion-Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=set(labels), yticklabels=set(labels))
    plt.title('Confusion Matrix (Geladenes Modell)')
    plt.show()

    # ==== GESAMTDATENSATZ VALIDIEREN ====
    print("\n==== Klassenvorhersage auf dem gesamten Datensatz ====")
    full_predictions = svm_model_loaded.predict(features)
    for true_label, pred_label in zip(labels, full_predictions):
        print(f"Tatsächlich: {true_label} -> Vorhergesagt: {pred_label}")

else:
    # === Neues Training ===
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Skalierung
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # PCA oder LDA
    use_lda = False
    if use_lda:
        lda = LDA(n_components=min(len(set(labels)) - 1, features.shape[1]))
        features_reduced = lda.fit_transform(features, labels)
    else:
        pca = PCA(n_components=2)
        features_reduced = pca.fit_transform(features)

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.3, random_state=42)

    # Hyperparameter-Tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'degree': [3, 4, 5, 6]  # Nur für 'poly'
    }

    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Bestes Modell
    svm_model = grid_search.best_estimator_
    print(f"\nBeste Parameter: {grid_search.best_params_}")
    print("Modell-Typ:", type(svm_model))
    print("decision_function_shape:", getattr(svm_model, 'decision_function_shape', 'Nicht vorhanden'))
    print("dual_coef_.shape:", svm_model.dual_coef_.shape)
    print("Anzahl Klassen:", len(svm_model.classes_))

    # Evaluation
    y_pred = svm_model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Confusion-Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=set(labels), yticklabels=set(labels))
    plt.title('Confusion Matrix')
    plt.show()

    # Modell und Parameter speichern
    joblib.dump(svm_model, model_filename)
    print(f"Modell als '{model_filename}' gespeichert.")

    joblib.dump(scaler, scaler_filename)
    print(f"Scaler als '{scaler_filename}' gespeichert.")

    joblib.dump(pca, pca_filename)
    print(f"PCA-Komponenten als '{pca_filename}' gespeichert.")
