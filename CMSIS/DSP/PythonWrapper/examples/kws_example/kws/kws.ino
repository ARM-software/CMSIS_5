/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        kws.ino
 * Description:  Very simple Yes detector
 *
 * $Date:        16 March 2022
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2022 ARM Limited or its affiliates. All rights reserved.
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
 */

#include <PDM.h>
#include "arm_math.h"
#include "coef.h"
#include "scheduler.h"

// default number of output channels
static const char channels = 1;

// default PCM output frequency
static const int frequency = 16000;

// Buffer to read samples into, each sample is 16-bits
short sampleBuffer[AUDIOBUFFER_LENGTH];

// Number of audio samples read
volatile int samplesRead=0;


void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("Starting ...");

  // Configure the data receive callback
  PDM.onReceive(onPDMdata);

  // Optionally set the gain
  // Defaults to 20 on the BLE Sense and 24 on the Portenta Vision Shield
  // PDM.setGain(30);

  // Initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate for the Arduino Nano 33 BLE Sense
  // - a 32 kHz or 64 kHz sample rate for the Arduino Portenta Vision Shield
  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }


}

void loop() {
  int error;
  uint32_t nb;

  // Start the CMSIS-DSP generated synchronous scheduling

  Serial.println("Scheduling ...");
  nb = scheduler(&error,window,coef_q15,coef_shift,intercept_q15,intercept_shift);
  Serial.println("End of scheduling");
  Serial.print("Error status = ");
  Serial.println(error);
  Serial.print("Nb iterations");
  Serial.println(nb);
}

/**
 * Callback function to process the data from the PDM microphone.
 * NOTE: This callback is executed as part of an ISR.
 * Therefore using `Serial` to print messages inside this function isn't supported.
 * */
void onPDMdata() {
  // Query the number of available bytes
  int bytesAvailable = PDM.available();

  // Get free remaining bytes in the buffer
  // If real time is respected, there should never be
  // any overflow.
  int remainingFreeBytes = AUDIOBUFFER_LENGTH*2 - samplesRead*2;
  
  if (remainingFreeBytes >= bytesAvailable)
  {
     // Read into the sample buffer
     int nbReadBytes = PDM.read(sampleBuffer+samplesRead, bytesAvailable);

     // 16-bit, 2 bytes per sample
     samplesRead += nbReadBytes / 2;
  }
  else if (remainingFreeBytes > 0)
  {
    // Read into the sample buffer
     int nbReadBytes = PDM.read(sampleBuffer+samplesRead, remainingFreeBytes);

     // 16-bit, 2 bytes per sample
     samplesRead += nbReadBytes / 2;
  }

 
}
