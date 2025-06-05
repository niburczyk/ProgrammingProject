const int emgPin = A0;
unsigned long lastSendTime = 0;
const unsigned int sampleInterval = 1;  // 1 ms = 1000 Hz
const unsigned int driftSamples = 100;  // Anzahl der Samples zur Drift-Berechnung

String command = "";
bool recording = true;

bool offsetCalculated = false;
unsigned int sampleCount = 0;
long offsetSum = 0;
float offset = 0;

void setup() {
  Serial.begin(230400);
  while (!Serial);
}

void loop() {
  unsigned long now = millis();

  if (recording && (now - lastSendTime >= sampleInterval)) {
    int emg = analogRead(emgPin);

    // 1. Phase: Drift/Offset berechnen
    if (!offsetCalculated) {
      offsetSum += emg;
      sampleCount++;

      if (sampleCount >= driftSamples) {
        offset = (float)offsetSum / driftSamples;
        offsetCalculated = true;
      }
    } else {
      // 2. Phase: Bereinigten EMG-Wert ausgeben
      float cleanedEMG = emg - offset;
      Serial.println(cleanedEMG);
    }

    lastSendTime = now;
  }
}
