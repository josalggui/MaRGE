/* Code to control the autotuning circuit with python.
We added two relays to control the VNA. First one selects if the autotuning is connected to the scanner or to the VNA.
Second one switches on or off the VNA to prevent noise coupling to the RF chain in the scanner.
*/
//  cPins[] = {S1, S2, S3, S4, S5, T1, T2, T3, T4, T5, M1, M2, M3, M4, M5, VNA, VNA}
int cPins[] = {34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 51, 53, 47, 49, 43, 45,  32};
int nPins = 17;

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
    Serial.write("Ready!\n");
  }

}
