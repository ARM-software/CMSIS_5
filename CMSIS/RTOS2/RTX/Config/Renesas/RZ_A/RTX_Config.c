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


// Define OS Timer channel and interrupt number
#define OSTM                        OSTM0
#define OSTM_IRQn                   OSTMI0TINT_IRQn

static uint32_t ExtTim_Cnt;         // Timer count used for overflow detection
static uint32_t ExtTim_Freq;        // Timer frequency


// Get OS Timer current count value
__STATIC_INLINE uint32_t OSTM_GetCount(void) {
  ExtTim_Cnt = OSTM.OSTMnCNT;
  return  (OSTM.OSTMnCMP - ExtTim_Cnt);
}

// Check if OS Timer counter was reloaded
__STATIC_INLINE uint32_t OSTM_GetOverflow(void) {
  return ((OSTM.OSTMnCNT > ExtTim_Cnt) ? (1U) : (0U));
}

// Get OS Timer period
__STATIC_INLINE uint32_t OSTM_GetPeriod(void) {
  return (OSTM.OSTMnCMP + 1U);
}


// Setup System Timer.
// \return system timer IRQ number.
int32_t osRtxSysTimerSetup (void) {
  uint32_t freq;

  // Get CPG.FRQCR[IFC] bits
  freq = (CPG.FRQCR >> 8) & 0x03;

  // Determine Divider 2 output clock by using SystemCoreClock
  if (freq == 0x03U) {
    freq = (SystemCoreClock * 3U);
  }
  else if (freq == 0x01U) {
    freq = (SystemCoreClock * 3U)/2U;
  }
  else {
    freq = SystemCoreClock;
  }
  // Peripheral clock 0C = (Divider 2 clock * 1/12)
  freq = freq / 12U;

  // Determine tick frequency
  freq = freq / osRtxConfig.tick_freq;

  // Save frequency for later
  ExtTim_Freq = freq;

  // Enable OSTM clock
  CPG.STBCR5 &= ~(CPG_STBCR5_BIT_MSTP51);

  // Stop the OSTM counter
  OSTM.OSTMnTT  = 0x01U;

  // Set interval timer mode and disable interrupts when counting starts
  OSTM.OSTMnCTL = 0x00U;

  // Set compare value
  OSTM.OSTMnCMP = freq - 1U;

  return (OSTM_IRQn);
}

// Enable System Timer.
void osRtxSysTimerEnable (void) {
  /* Start the OSTM counter */
  OSTM.OSTMnTS = 0x01U;
}

// Disable System Timer.
void osRtxSysTimerDisable (void) {
  // Stop the OSTM counter
  OSTM.OSTMnTT = 0x01U;
}

// Acknowledge System Timer IRQ.
void osRtxSysTimerAckIRQ (void) {
  // Acknowledge OSTM interrupt
  GIC_ClearPendingIRQ (OSTM_IRQn);
}

// Get System Timer count.
// \return system timer count.
uint32_t osRtxSysTimerGetCount (void) {
  uint32_t tick;
  uint32_t val;

  tick = (uint32_t)osRtxInfo.kernel.tick;
  val  = OSTM_GetCount();
  if (OSTM_GetOverflow()) {
    val = OSTM_GetCount();
    tick++;
  }
  val += tick * OSTM_GetPeriod();

  return val;
}

// Get System Timer frequency.
// \return system timer frequency.
uint32_t osRtxSysTimerGetFreq (void) {
  return ExtTim_Freq;
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
