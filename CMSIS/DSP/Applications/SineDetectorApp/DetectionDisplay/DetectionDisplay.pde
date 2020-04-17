  /*
 * Copyright (c) 2020 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * This example reads audio data from the on-board PDM microphones
 * and try to detect a 1kHz sine signal using a SVM predictor.
 *
 * Circuit:
 *   - Arduino Nano 33 BLE board
 */

import processing.serial.*;

Serial myPort;

// Color opacity for the test display
int n = 0;

// Decay of color opacity
int decay = 5;

void setup()
{
  size(380, 150);
  myPort = new Serial(this, "COM6", 115200);
 
  textSize(72); // set text size
  myPort.clear();
}

void draw() {
  // If some data is available on the serial port then some signal was detected.
  if (myPort.available() > 0) 
  {
    // Opacity is set to maximum.
    n=255;
    myPort.clear();
  }
  background(255);
  textAlign(CENTER);

  // Define a green color with some oapcity
  fill(0, 255, 0, n);

  // Decrease the opacity until it is 0.
  if (n >= decay)
  {
     n = n - decay;
  }
  else
  {
    n = 0;
  }

  // Display the word "Detected"
  text("DETECTED", 190, 95);
}

 
