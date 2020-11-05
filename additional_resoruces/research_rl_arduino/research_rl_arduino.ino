/**
 *
 * HX711 library for Arduino - example file
 * https://github.com/bogde/HX711
 *
 * MIT License
 * (c) 2018 Bogdan Necula
 *
**/
#include "HX711.h"


// HX711 circuit wiring
const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN = 3;


HX711 scale_base, scale_1, scale_2;

void setup() {
  Serial.begin(38400);
//  Serial.println("HX711 Demo");
//
//  Serial.println("Initializing the scale");

  // Initialize library with data output pin, clock input pin and gain factor.
  // Channel selection is made by passing the appropriate gain:
  // - With a gain factor of 64 or 128, channel A is selected
  // - With a gain factor of 32, channel B is selected
  // By omitting the gain factor parameter, the library
  // default "128" (Channel A) is used here.
  scale_base.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale_1.begin(4, 5);
  scale_2.begin(6, 7);

  scale_base.set_scale(434.f);                      // this value is obtained by calibrating the scale with known weights; see the README for details
  scale_base.tare();				        // reset the scale to 0

  scale_1.set_scale(434.f);                      // this value is obtained by calibrating the scale with known weights; see the README for details
  scale_1.tare();               // reset the scale to 0
  
  scale_2.set_scale(434.f);                      // this value is obtained by calibrating the scale with known weights; see the README for details
  scale_2.tare();               // reset the scale to 0

}

void loop() {
  Serial.print(scale_base.get_units(), 2);
  Serial.print(" ");
  Serial.print(scale_1.get_units(), 2);
  Serial.print(" ");
  Serial.println(scale_2.get_units(), 2);
}
