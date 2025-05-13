#include <svm_model.h>
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

// Standardisierung der Eingabewerte basierend auf der Ruhephase
float* normalize_input(float input[]) {
    static float normalized[2]; // static für Rückgabe als Zeiger
    for (int i = 0; i < 2; i++) {
        normalized[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];
    }
    return normalized;
}

// Polynomieller Kernel
float kernel_poly(const float* x, const float* sv) {
    float dot_product = 0.0;
    for (int i = 0; i < vector_length; ++i) {
        dot_product += x[i] * sv[i];
    }
    return powf(gamma * dot_product + coef0, degree);  // Polynomieller Kernel
}

// Funktion zur Durchführung der Vorhersage
int svm_predict(const float* input) {
    float decision[n_classes] = {0.0};  // Array für die Entscheidung jeder Klasse

    // Berechnung der Entscheidungswerte für jede Klasse
    for (int i = 0; i < n_classes; ++i) {
        for (int j = 0; j < n_support_vectors; ++j) {
            float k = kernel_poly(input, support_vectors[j]);
            decision[i] += dual_coef[i][j] * k;
        }
        decision[i] += intercept[i];  // Intercept hinzufügen
    }

    // Bestimme die Klasse mit der höchsten Entscheidung
    int predicted_class = 0;
    float max_decision = decision[0];

    for (int i = 1; i < n_classes; ++i) {
        if (decision[i] > max_decision) {
            max_decision = decision[i];
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

    // Initialisieren der ersten Werte
    for (int i = 0; i < windowSize; i++) {
        sensorData[i] = 0;
    }
}

void loop() {
    unsigned long currentTime = millis();

    // Serielle Eingabe prüfen (Start/Pause)
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
            Serial.println(">> Ruhemessung gestartet (10sek.)")
            int value = analogRead(analogInPin);
            restSum += value;
            restCount++;
        }
    }
    // Ruhephase abgeschlossen
    else if (!measuring) {
        restAverage = (float)restSum / restCount; // Berechne den Mittelwert der Ruhephase
        measuring = true;
        Serial.println("Messung beginnt...");
    }

    // Hauptmessung (wenn nicht pausiert)
    else if (!paused && (currentTime - lastSampleTime >= sampleInterval)) {
        lastSampleTime = currentTime;

        // Lese den aktuellen Wert vom Sensor
        int rawValue = analogRead(analogInPin);
        float diff = rawValue - restAverage;  // Subtrahiere den Ruhemittelwert (Normalisierung)

        // Berechnung des gleitenden Mittelwerts
        sum += diff - sensorData[dataIndex];  // Entferne den ältesten Wert und füge den neuen hinzu
        sensorData[dataIndex] = diff;  // Speichere den neuen Wert
        dataIndex = (dataIndex + 1) % windowSize;  // Zirkulierung des Index

        // Berechne MAV und WL für den aktuellen Puffer
        mav = calculate_mav();
        wl = calculate_wl();

        // Eingabewerte für das SVM-Modell (MAV und WL)
        float x_input[vector_length] = {mav, wl};

        // Vorhersage treffen und vorher normalisierung der Eingabewerte basierend auf dem Ruhemittelwert
        int prediction = svm_predict( normalize_input(x_input););

        // Ausgabe der Vorhersage
        Serial.print("Vorhergesagte Klasse: ");
        Serial.println(prediction);
    }
}
