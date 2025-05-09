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

    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            float sum = 0.0;

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
    int predicted_class = 0;
    int max_votes = votes[0];
    for (int i = 1; i < n_classes; ++i) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

void setup() {
    Serial.begin(9600);
    while (!Serial); // Warten auf seriellen Monitor
    Serial.println("Start der Messung...");
    startTime = millis();

    // Initialisieren
    for (int i = 0; i < windowSize; i++) {
        sensorData[i] = 0;
    }
    Serial.println(">> Ruhemessung gestartet (10sek.)");
}

void loop() {
    unsigned long currentTime = millis();

    // Serielle Steuerung
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        if (input.equalsIgnoreCase("stop")) {
            paused = true;
            Serial.println(">> Ausgabe gestoppt.");
        } else if (input.equalsIgnoreCase("start")) {
            paused = false;
            Serial.println(">> Ausgabe fortgesetzt.");
        }
    }

    // Ruhemessung
    if (!measuring && (currentTime - startTime <= restDuration)) {
        if (currentTime - lastSampleTime >= sampleInterval) {
            lastSampleTime = currentTime;
            int value = analogRead(analogInPin);
            restSum += value;
            restCount++;
        }
    }

    // Ruhephase abgeschlossen
    else if (!measuring) {
        restAverage = (float)restSum / restCount;
        measuring = true;
        Serial.println("Messung beginnt...");
    }

    // Hauptmessung
    else if (!paused && (currentTime - lastSampleTime >= sampleInterval)) {
        lastSampleTime = currentTime;

        // Lese aktuellen Sensorwert
        int rawValue = analogRead(analogInPin);
        float diff = rawValue - restAverage;

        // Gleitender Mittelwert
        sum += diff - sensorData[dataIndex];
        sensorData[dataIndex] = diff;
        dataIndex = (dataIndex + 1) % windowSize;

        mav = calculate_mav();
        wl = calculate_wl();

        float x_input[vector_length] = {mav, wl};

        int prediction = svm_predict(x_input);

        // Ausgabe
        Serial.print("Klasse: ");
        Serial.print(prediction);
        Serial.print(" | MAV: ");
        Serial.print(mav);
        Serial.print(" | WL: ");
        Serial.println(wl);
    }
}
