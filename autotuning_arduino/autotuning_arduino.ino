// Code to control the autotuning circuit with python.
//  cPins[] = {S1, S2, S3, S4, S5, T1, T2, T3, T4, T5, M1, M2, M3, M4, M5, TTL}
int cPins[] = {22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 42, 44, 46, 48, 50, 53};
int nPins = 16;

void setup() {
  // Start the Serial communication
  Serial.begin(115200);
  Serial.setTimeout(100);
  // Set the pins for tuning, matching and series capacitors
  for (int c=0; c<nPins; c++) {
    pinMode(cPins[c], OUTPUT);
  }
}

void loop() {
  // Wait until there are any data available into the serial port
  if (Serial.available()>0) {
    delay(10);
    String state = Serial.readString();
    for (int c=0; c<nPins; c++) {
      digitalWrite(cPins[c], String(state[c]).toInt());
    }
    Serial.write("Ready!");
  }
}
