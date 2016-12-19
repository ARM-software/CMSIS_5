/*
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
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
 * ----------------------------------------------------------------------
 *
 * $Date:        20. May 2015
 * $Revision:    V1.10
 *
 * Project:      CMSIS-DAP Template MDK5
 * Title:        main.c RTX CMSIS-DAP Main module
 *
 *---------------------------------------------------------------------------*/

#include "cmsis_os.h"
#include "osObjects.h"
#include "rl_usb.h"
#include "DAP_config.h"
#include "DAP.h"

// Main program
int main (void) {

  DAP_Setup();                          // DAP Setup 

  USBD_Initialize(0U);                  // USB Device Initialization
  USBD_Connect(0U);                     // USB Device Connect

  while (!USBD_Configured(0U));         // Wait for USB Device to configure

  LED_CONNECTED_OUT(1U);                // Turn on  Debugger Connected LED
  LED_RUNNING_OUT(1U);                  // Turn on  Target Running LED
  Delayms(500U);                        // Wait for 500ms
  LED_RUNNING_OUT(0U);                  // Turn off Target Running LED
  LED_CONNECTED_OUT(0U);                // Turn off Debugger Connected LED

  // Create HID Thread
  HID0_ThreadId = osThreadCreate(osThread(HID0_Thread), NULL);

  osThreadSetPriority(osThreadGetId(), osPriorityIdle);
  for (;;);                             // Endless Loop
}
