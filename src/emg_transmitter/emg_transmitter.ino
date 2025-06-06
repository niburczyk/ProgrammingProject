const int emgPin = A0;

const unsigned long sampleIntervalMicros = 500;  // 0,5 ms = 2000 Hz
unsigned long lastSampleTime = 0;

const unsigned int driftSamples = 100;

bool offsetCalculated = false;
unsigned int sampleCount = 0;
unsigned long offsetSum = 0;
float offset = 0;

void setup() {
  Serial.begin(230400);
  while (!Serial);
}

void loop() {
  unsigned long now = micros();

  if (now - lastSampleTime >= sampleIntervalMicros) {
    int emg = analogRead(emgPin);

    if (!offsetCalculated) {
      offsetSum += emg;
      sampleCount++;

      if (sampleCount >= driftSamples) {
        offset = (float)offsetSum / driftSamples;
        offsetCalculated = true;
      }
    } else {
      float cleanedEMG = emg - offset;
      Serial.println(cleanedEMG);
    }

    lastSampleTime = now;
  }
}
