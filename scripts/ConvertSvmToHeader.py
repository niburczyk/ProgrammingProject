import joblib
import numpy as np

# === Pfad zum Modell ===
model_path = "svm_model_optimized.pkl"
scaler_path = "scaler.pkl"  # Optional, falls separat gespeichert

# === Modell und Preprocessing laden ===
model = joblib.load(model_path)
try:
    scaler = joblib.load(scaler_path)
    has_scaler = True
except:
    scaler = None
    has_scaler = False

# === Modellparameter extrahieren ===
support_vectors = model.support_vectors_
dual_coef = model.dual_coef_
intercept = model.intercept_
gamma = model._gamma  # Achtung: _gamma enthält den berechneten Wert
coef0 = model.coef0 if hasattr(model, "coef0") else 0.0
degree = model.degree if hasattr(model, "degree") else 3
class_labels = model.classes_

# === Optional: Scaler-Parameter ===
if has_scaler:
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

# === Hilfsfunktion zum Umwandeln von NumPy in C-Arrays ===
def array_to_c(name, array, dtype="float"):
    array = np.array(array)
    shape = array.shape
    if len(shape) == 1:
        return f"const {dtype} {name}[{shape[0]}] = {{ " + ", ".join(f"{v:.6f}" for v in array) + " };\n"
    elif len(shape) == 2:
        return f"const {dtype} {name}[{shape[0]}][{shape[1]}] = {{\n" + \
            ",\n".join("    {" + ", ".join(f"{v:.6f}" for v in row) + "}" for row in array) + "\n};\n"
    else:
        raise ValueError("Array zu hochdimensional")

# === Header-Datei zusammenbauen ===
header = "#ifndef SVM_MODEL_H\n#define SVM_MODEL_H\n\n"
header += "#include <Arduino.h>\n\n"

header += f"const int n_classes = {len(class_labels)};\n"
header += f"const int n_support_vectors = {support_vectors.shape[0]};\n"
header += f"const int vector_length = {support_vectors.shape[1]};\n\n"

header += array_to_c("support_vectors", support_vectors)
header += array_to_c("dual_coef", dual_coef)
header += array_to_c("intercept", intercept)
header += array_to_c("class_labels", class_labels.astype(int), dtype="int")

header += f"\nconst float gamma = {gamma:.10f};\n"
header += f"const float coef0 = {coef0:.6f};\n"
header += f"const int degree = {degree};\n"

if has_scaler:
    header += "\n// === Scaler Parameter ===\n"
    header += array_to_c("scaler_mean", scaler_mean)
    header += array_to_c("scaler_scale", scaler_scale)

header += "\n#endif // SVM_MODEL_H\n"

# === Speichern der Header-Datei ===
with open("svm_model.h", "w") as f:
    f.write(header)

print("✅ Header-Datei 'svm_model.h' wurde erfolgreich erstellt.")