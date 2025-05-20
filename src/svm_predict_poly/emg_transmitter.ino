const int emgPin = A0;
unsigned long lastSendTime = 0;
const unsigned int sampleInterval = 5;  // 5 ms -> 200 Hz Samplingrate
String prediction = "";

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Für Leonardo/Micro Boards
  delay(2000);
  Serial.println("Arduino bereit");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSendTime >= sampleInterval) {
    int emg = analogRead(emgPin);
    Serial.println(emg);  // EMG Wert senden
    lastSendTime = now;
  }

  // Prediction vom Pi lesen, falls vorhanden
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      Serial.print("Vorhersage vom Pi: ");
      Serial.println(prediction);
      prediction = "";
    } else {
      prediction += c;
    }
  }
}
