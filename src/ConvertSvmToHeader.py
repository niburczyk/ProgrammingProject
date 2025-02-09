import joblib
import numpy as np

# Laden des gespeicherten SVM-Modells
model = joblib.load("svm_model_optimized.pkl")

# Extrahieren der Modellparameter
support_vectors = model.support_vectors_
dual_coef = model.dual_coef_
intercept = model.intercept_
gamma = model.gamma if model.gamma != "scale" else 1.0 / (support_vectors.shape[1] * support_vectors.var())
coef0 = model.coef0
degree = model.degree
n_classes = len(model.classes_)

# Funktion zur Umwandlung eines NumPy-Arrays in C++-Format
def array_to_c(name, array):
    array = np.array(array)
    shape = array.shape
    if len(shape) == 1:
        return f"const float {name}[{shape[0]}] = {{" + ", ".join(map(str, array)) + "};\n"
    else:
        return f"const float {name}[{shape[0]}][{shape[1]}] = {{\n" + \
               ",\n".join(["    {" + ", ".join(map(str, row)) + "}" for row in array]) + "\n};\n"

# Generierung der Header-Datei
header = "#ifndef SVM_MODEL_H\n#define SVM_MODEL_H\n\n"
header += "#include <Arduino.h>\n\n"

header += f"const int n_classes = {n_classes};\n"
header += array_to_c("support_vectors", support_vectors)
header += array_to_c("dual_coef", dual_coef)
header += array_to_c("intercept", intercept)
header += f"const float gamma = {gamma};\n"
header += f"const float coef0 = {coef0};\n"
header += f"const int degree = {degree};\n\n"

header += "#endif // SVM_MODEL_H"

# Speichern der Datei
with open("svm_model.h", "w") as f:
    f.write(header)

print("Header-Datei 'svm_model.h' wurde erfolgreich erstellt.")
