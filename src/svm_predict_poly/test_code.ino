#include "svm_model.h"  
#include <math.h>

const int analogInPin = A0; // Analoger Eingabepin
const unsigned long restDuration = 10000; // Ruhephase in ms
const unsigned long sampleInterval = 1;   // 1000 Hz Samplingrate
const int windowSize = 100;  // Fenstergröße für den gleitenden Mittelwert

unsigned long startTime;
unsigned long lastSampleTime = 0;

bool measuring = false;
bool paused = false;

unsigned long restSum = 0;
unsigned int restCount = 0;
float restAverage = 0;

// Gleitender Mittelwert
float sensorData[windowSize];  // Array für den gleitenden Mittelwert
int dataIndex = 0;  // Zeiger für das aktuelle Element im Array
float sum = 0.0;  // Summe der letzten windowSize Werte für MAV

float mav = 0.0;  // Berechneter MAV-Wert
float wl = 0.0;   // Berechneter WL-Wert

// Berechnung des Mean Absolute Value (MAV)
float calculate_mav() {
    return sum / windowSize;
}

// Berechnung der Waveform Length (WL)
float calculate_wl() {
    float sum_wl = 0.0;
    for (int i = 1; i < windowSize; i++) {
        sum_wl += abs(sensorData[i] - sensorData[i - 1]);
    }
    return sum_wl;
}

// Polynomieller Kernel
float kernel_poly(const float* x, const float* sv) {
    float dot_product = 0.0;
    for (int i = 0; i < vector_length; ++i) {
        dot_product += x[i] * sv[i];
    }
    return powf(gamma * dot_product + coef0, degree);  // Polynomieller Kernel
}

// One-vs-One SVM-Vorhersage
int svm_predict(const float* input) {
    int votes[n_classes] = {0};
    int vote_index = 0;
    Serial.print("Votes: ");

    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            float sum = 0.0;
            Serial.print(votes[i]); Serial.print(" ");

            for (int k = 0; k < n_support_vectors; ++k) {
                float k_val = kernel_poly(input, support_vectors[k]);
                sum += dual_coef[vote_index][k] * k_val;
            }

            sum += intercept[vote_index];

            if (sum > 0)
                votes[i]++;
            else
                votes[j]++;

            vote_index++;
        }
    }

    // Bestimme Klasse mit den meisten Stimmen
    int predicted_class;
    int max_votes = votes[0];
    for (int i = 1; i < n_classes; ++i) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

// Funktion zur Normalisierung der Eingabedaten
float* normalize_input(float input[]) {
    static float normalized[2]; // static für Rückgabe als Zeiger
    for (int i = 0; i < 2; i++) {
        normalized[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];
    }
    return normalized;
}

// Funktion zur PCA-Transformation (Dummy-Implementierung - Ersetze mit deinen PCA-Komponenten)
void pca_transform(float* input) {
    // Hier sollte die PCA-Transformation angewendet werden
    // Beispielsweise mit vordefinierten Komponenten für die Reduktion auf 2 Dimensionen
    float transformed[2] = {input[0] * 1.0 + input[1] * 0.5, input[0] * -0.5 + input[1] * 1.0}; // Beispielhafte Transformation
    input[0] = transformed[0];
    input[1] = transformed[1];
}

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Warten bis der serielle Monitor verbunden ist

  Serial.println("Starte SVM-Prediction-Test...");

  // Testdaten: {MAV, WL}
  const float test_data[][2] = {
    {0.000150, 0.0125},   // Klasse 0
    {0.000155, 0.0130},   // Klasse 0
    {0.000160, 0.0135},   // Klasse 0
    {0.000165, 0.0140},   // Klasse 0
    {0.000170, 0.0200},   // Klasse 1
    {0.000180, 0.0210},   // Klasse 1
    {0.000190, 0.0220},   // Klasse 1
    {0.000195, 0.0230},   // Klasse 1
    {0.000200, 0.0250},   // Klasse 2
    {0.000210, 0.0260},   // Klasse 2
    {0.000220, 0.0270},   // Klasse 2
    {0.000230, 0.0280},   // Klasse 2
    {0.000240, 0.0300},   // Klasse 2
  };

  const int num_samples = sizeof(test_data) / sizeof(test_data[0]);

  for (int i = 0; i < num_samples; i++) {
    float input[2] = {test_data[i][0], test_data[i][1]};
    float* norm_ptr = normalize_input(input);
    float normalized[2] = { norm_ptr[0], norm_ptr[1] };

    // PCA-Transformation auf die normalisierten Eingabedaten anwenden
    pca_transform(normalized);

    int pred = svm_predict(normalized);
    
    Serial.print("Sample ");
    Serial.print(i);
    Serial.print(" | Normalized Input: [");
    Serial.print(normalized[0], 6);
    Serial.print(", ");
    Serial.print(normalized[1], 6);
    Serial.print("] | Predicted Class: ");
    Serial.println(pred);
  }
}

void loop() {
    // Deine Hauptlogik für die Messung bleibt unverändert
}
