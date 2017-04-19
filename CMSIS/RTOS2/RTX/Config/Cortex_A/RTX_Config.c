/*
 * Copyright (c) 2013-2017 ARM Limited. All rights reserved.
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
 * -----------------------------------------------------------------------------
 *
 * $Revision:   V5.1.0
 *
 * Project:     CMSIS-RTOS RTX
 * Title:       RTX Configuration
 *
 * -----------------------------------------------------------------------------
 */
 
#include "RTE_Components.h"
#include CMSIS_device_header

#include "rtx_os.h"


// Setup System Timer.
// \return system timer IRQ number.
int32_t osRtxSysTimerSetup (void) {
  // ...
  return (0);
}

// Enable System Timer.
void osRtxSysTimerEnable (void) {
  // ...
}

// Disable System Timer.
void osRtxSysTimerDisable (void) {
  // ...
}

// Acknowledge System Timer IRQ.
void osRtxSysTimerAckIRQ (void) {
  // ...
}

// Get System Timer count.
// \return system timer count.
uint32_t osRtxSysTimerGetCount (void) {
  // ...
  return (0U);
}

// Get System Timer frequency.
// \return system timer frequency.
uint32_t osRtxSysTimerGetFreq (void) {
  // ...
  return (1000000U);
}


// OS Idle Thread
__WEAK __NO_RETURN void osRtxIdleThread (void *argument) {
  (void)argument;

  for (;;) {}
}
 
// OS Error Callback function
__WEAK uint32_t osRtxErrorNotify (uint32_t code, void *object_id) {
  (void)object_id;

  switch (code) {
    case osRtxErrorStackUnderflow:
      // Stack underflow detected for thread (thread_id=object_id)
      break;
    case osRtxErrorISRQueueOverflow:
      // ISR Queue overflow detected when inserting object (object_id)
      break;
    case osRtxErrorTimerQueueOverflow:
      // User Timer Callback Queue overflow detected for timer (timer_id=object_id)
      break;
    case osRtxErrorClibSpace:
      // Standard C/C++ library libspace not available: increase OS_THREAD_LIBSPACE_NUM
      break;
    case osRtxErrorClibMutex:
      // Standard C/C++ library mutex initialization failed
      break;
    default:
      break;
  }
  for (;;) {}
//return 0U;
}
