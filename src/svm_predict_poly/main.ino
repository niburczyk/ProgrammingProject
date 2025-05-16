#include "measurement.ino"

void setup() {
    Serial.begin(9600);
    while (!Serial);
    Serial.println("Start der Messung...");
    initMeasurement();
}

void loop() {
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

    performMeasurementStep();
}
