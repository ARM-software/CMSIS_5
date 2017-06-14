/**************************************************************************//**
 * @file     os_tick_ptim.c
 * @brief    CMSIS OS Tick implementation for Private Timer
 * @version  V1.0.0
 * @date     13. June 2017
 ******************************************************************************/
/*
 * Copyright (c) 2017-2017 ARM Limited. All rights reserved.
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

#include "os_tick.h"

#include "RTE_Components.h"
#include CMSIS_device_header

#if defined(PTIM) && defined (__GIC_PRESENT) && (__GIC_PRESENT != 0U)

#ifndef PTIM_IRQ_PRIORITY
#define PTIM_IRQ_PRIORITY           0xFFU
#endif

static uint8_t PTIM_PendIRQ;        // Timer interrupt pending flag

// Setup OS Tick.
int32_t  OS_Tick_Setup (uint32_t freq, IRQHandler_t handler) {
  uint32_t load;
  uint32_t prio;
  uint32_t bits;

  if (freq == 0U) {
    return (-1);
  }

  PTIM_PendIRQ = 0U;

  // Private Timer runs with the system frequency
  load = (SystemCoreClock / freq) - 1U;

  // Disable Private Timer and set load value
  PTIM_SetControl   (0U);
  PTIM_SetLoadValue (load);

  // Disable corresponding IRQ
  GIC_DisableIRQ     (PrivTimer_IRQn);
  GIC_ClearPendingIRQ(PrivTimer_IRQn);

  // Determine number of implemented priority bits
  GIC_SetPriority (PrivTimer_IRQn, 0xFFU);

  prio = GIC_GetPriority (PrivTimer_IRQn);

  // At least bits [7:4] must be implemented
  if ((prio & 0xF0U) == 0U) {
    return (-1);
  }

  for (bits = 0; bits < 4; bits++) {
    if ((prio & 0x01) != 0) {
      break;
    }
    prio >>= 1;
  }
  
  // Adjust configured priority to the number of implemented priority bits
  prio = (PTIM_IRQ_PRIORITY << bits) & 0xFFUL;

  // Set Private Timer interrupt priority
  GIC_SetPriority(PrivTimer_IRQn, prio-1U);

  // Set edge-triggered and 1-N model bits
  GIC_SetLevelModel(PrivTimer_IRQn, 1, 1);

  // Register tick interrupt handler function
  InterruptHandlerRegister(PrivTimer_IRQn, (IRQHandler)handler);

  // Enable corresponding IRQ
  GIC_EnableIRQ (PrivTimer_IRQn);

  // Set bits: IRQ enable and Auto reload
  PTIM_SetControl (0x06U);

  return (0);
}

/// Enable OS Tick.
int32_t  OS_Tick_Enable (void) {
  uint32_t ctrl;

  // Set pending interrupt if flag set
  if (PTIM_PendIRQ != 0U) {
    PTIM_PendIRQ = 0U;
    GIC_SetPendingIRQ (PrivTimer_IRQn);
  }

  // Start the Private Timer
  ctrl  = PTIM_GetControl();
  // Set bit: Timer enable
  ctrl |= 1U;
  PTIM_SetControl (ctrl);

  return (0);
}

/// Disable OS Tick.
int32_t  OS_Tick_Disable (void) {
  uint32_t ctrl;
  
  // Stop the Private Timer
  ctrl  = PTIM_GetControl();
  // Clear bit: Timer enable
  ctrl &= ~1U;
  PTIM_SetControl (ctrl);

  // Remember pending interrupt flag
  if ((GIC_GetIRQStatus (PrivTimer_IRQn) >> 1) != 0) {
    GIC_ClearPendingIRQ (PrivTimer_IRQn);
    PTIM_PendIRQ = 1U;
  }

  return (0);
}

// Acknowledge OS Tick IRQ.
int32_t  OS_Tick_AcknowledgeIRQ (void) {
  PTIM_ClearEventFlag();
  return (0);
}

// Get OS Tick IRQ number.
int32_t  OS_Tick_GetIRQn (void) {
  return (PrivTimer_IRQn);
}

// Get OS Tick clock.
uint32_t OS_Tick_GetClock (void) {
  return (SystemCoreClock);
}

// Get OS Tick interval.
uint32_t OS_Tick_GetInterval (void) {
  return (PTIM_GetLoadValue() + 1U);
}

// Get OS Tick count value.
uint32_t OS_Tick_GetCount (void) {
  uint32_t load = PTIM_GetLoadValue();
  return  (load - PTIM_GetCurrentValue());
}

// Get OS Tick overflow status.
uint32_t OS_Tick_GetOverflow (void) {
  return (PTIM->ISR & 1);
}

#endif  // PTIM
