const int emgPin = A0;
unsigned long lastSendTime = 0;
const unsigned int sampleInterval = 1;  // 1 ms -> 1000 Hz Samplingrate
String command = "";
bool recording = true;

void setup() {
  Serial.begin(9600);
  while (!Serial);
  delay(2000);
  Serial.println("Arduino bereit");
}

void loop() {
  unsigned long now = millis();

  if (recording && (now - lastSendTime >= sampleInterval)) {
    int emg = analogRead(emgPin);
    Serial.println(emg);  // EMG-Wert senden
    lastSendTime = now;
  }
}
