#include "svm_model.h"
#include <math.h>

// Beispielinput (muss dieselbe Länge wie vector_length haben!)
float x_input[vector_length] = {
    0.5, 1.2, -0.3, 0.9  // <-- Ersetze durch reale Eingabewerte
};

// Optional: Standardisierung (falls vorhanden)
void standardize_input(float* input) {
#ifdef scaler_mean
    for (int i = 0; i < vector_length; ++i) {
        input[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];
    }
#endif
}

// Polynomieller Kernel
float kernel_poly(const float* x, const float* sv) {
    float dot_product = 0.0;
    for (int i = 0; i < vector_length; ++i) {
        dot_product += x[i] * sv[i];
    }
    return powf(gamma * dot_product + coef0, degree);
}

// SVM-Vorhersage
int svm_predict(const float* input) {
    float decision = 0.0;

    for (int i = 0; i < n_support_vectors; ++i) {
        float k = kernel_poly(input, support_vectors[i]);
        decision += dual_coef[0][i] * k;
    }

    decision += intercept[0];

    // Binäre Klassifikation
    return (decision >= 0) ? class_labels[1] : class_labels[0];
}

void setup() {
    Serial.begin(9600);
    delay(1000); // Warten auf den Serial Monitor

    // Optional skalieren
    standardize_input(x_input);

    // Vorhersage treffen
    int prediction = svm_predict(x_input);

    // Ausgabe
    Serial.print("Vorhergesagte Klasse: ");
    Serial.println(prediction);
}

void loop() {
    // Optional: kontinuierliche Vorhersage oder Sensorintegration
}
