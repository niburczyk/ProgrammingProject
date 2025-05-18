import numpy as np
import joblib

model = joblib.load("./model/svm_model_optimized.pkl")

# Parameter aus deiner Header-Datei
gamma = 10.0
coef0 = 0.0
degree = 6

scaler_mean = np.array([0.000160, 0.097490])
scaler_scale = np.array([0.000484, 0.303631])

support_vectors = np.array([
    [-0.446748, 0.013669],
    [-0.459677, 0.006412],
    [-0.459683, 0.006415],
    [-0.459669, 0.006427],
    [-0.459683, 0.006416],
    [-0.458928, 0.006736],
    [-0.459061, 0.006683],
    [-0.453842, 0.009702]
])

dual_coef = np.array([
    [0.879183, -0.000000, -0.000000, -0.879183, -0.000000, -0.000000, -0.000000, -3.123187],  # Klasse 0 vs Rest
    [3.123187, 63.725490, 61.565937, 63.725490, 63.725490, -125.291427, -127.450980, -0.000000] # Klasse 1 vs Rest
])

intercepts = np.array([11.223850, 20.309426])  # nur 2 Intercepts für 2 Klassen (Klasse 2 implizit)

pca_components = np.array([
    [0.707107, 0.707107],
    [-0.707107, 0.707107]
])
pca_mean = np.array([-0.000000, -0.000000])

# Kernelfunktion (Polynom)
def polynomial_kernel(x, sv):
    return (gamma * np.dot(x, sv) + coef0) ** degree

def svm_predict(x_raw):
    # 1. Scalen
    x_scaled = (x_raw - scaler_mean) / scaler_scale
    # 2. PCA-Transformation
    x_pca = np.dot(pca_components, (x_scaled - pca_mean))

    # 3. OvR-Entscheidungswerte berechnen (für Klasse 0 und 1)
    decision_values = []
    for i in range(2):  # zwei Klassen mit dual_coef & intercept
        sv_contrib = 0
        for j in range(support_vectors.shape[0]):
            sv_contrib += dual_coef[i, j] * polynomial_kernel(x_pca, support_vectors[j])
        decision_values.append(sv_contrib + intercepts[i])

    # 4. Score für Klasse 2 = 0 (keine explizite Entscheidungsfunktion)
    decision_values.append(0.0)

    # 5. Maximalwert suchen und Index (Klasse) zurückgeben
    predicted_class = np.argmax(decision_values)
    return predicted_class

# Testbeispiele
test_inputs = [
    [2.31762189034315e-06, 0.004736648602727831],
    [0.001111451906441353,0.6001055854177187],
    [5.062407057867585e-07, 0.001150456687647776]
]

for x in test_inputs:
    pred = svm_predict(np.array(x))
    predictions = model.predict(test_inputs)
    print(f"Input {x} => vorhergesagte Klasse: {pred}")
