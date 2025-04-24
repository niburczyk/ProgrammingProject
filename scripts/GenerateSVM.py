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

# Pfade
csv_file_path = './training_dataset.csv'
model_filename = 'svm_model_optimized.pkl'

# Prüfen, ob das Modell existiert
if os.path.exists(model_filename):
    # Modell laden
    svm_model_loaded = joblib.load(model_filename)
    print(f"Gespeichertes Modell '{model_filename}' geladen.")

    # Daten erneut einlesen für Evaluation
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Standardisierung
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # ===== WÄHLE HIER PCA/LDA FÜR DIE EVALUATION =====
    use_lda_for_eval = True

    if use_lda_for_eval:
        lda = LDA(n_components=min(len(set(labels)) - 1, features.shape[1]))
        features = lda.fit_transform(features, labels)
    else:
        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

    # Testdaten (gleiche Aufteilung wie beim Training)
    _, X_test, _, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Evaluation
    y_pred = svm_model_loaded.predict(X_test)
    print(f"Accuracy (geladenes Modell): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=set(labels), yticklabels=set(labels))
    plt.title('Confusion Matrix (Geladenes Modell)')
    plt.show()

else:
    # Daten einlesen
    data = pd.read_csv(csv_file_path)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Standardisierung
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # ===== WÄHLE HIER PCA ODER LDA =====
    use_lda = False  # True für LDA, False für PCA

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
        'degree': [3, 4, 5, 6]  # Nur für poly relevant
    }

    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Bestes Modell
    svm_model = grid_search.best_estimator_
    print(f"Beste Parameter: {grid_search.best_params_}")

    # Evaluation
    y_pred = svm_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=set(labels), yticklabels=set(labels))
    plt.title('Confusion Matrix')
    plt.show()

    # Modell speichern
    joblib.dump(svm_model, model_filename)
    print(f"Modell als '{model_filename}' gespeichert.")