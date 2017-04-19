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
 * Project:     CMSIS-RTOS RTX
 * Title:       RTX GIC functions
 *
 * -----------------------------------------------------------------------------
 */

#include "RTE_Components.h"
#include CMSIS_device_header

#include "rtx_lib.h"

#if ((__ARM_ARCH_7A__ == 1U) && (__GIC_PRESENT == 1U))

extern const uint32_t irqRtxGicBase[];
       const uint32_t irqRtxGicBase[2] = {
  GIC_DISTRIBUTOR_BASE,
  GIC_INTERFACE_BASE
};


static IRQn_Type PendSV_IRQn;
static uint8_t   PendSV_Flag = 0U;


// Pending supervisor call interface
// =================================

/// Get Pending SV (Service Call) Flag
/// \return    Pending SV Flag
uint8_t GetPendSV (void) {
  uint32_t pend;

  pend = GIC_GetIRQStatus(PendSV_IRQn);

  return ((uint8_t)(pend & 1U));
}

/// Clear Pending SV (Service Call) Flag
void ClrPendSV (void) {
  GIC_ClearPendingIRQ(PendSV_IRQn);
  PendSV_Flag = 0U;
}

/// Set Pending SV (Service Call) Flag
void SetPendSV (void) {
  PendSV_Flag = 1U;
  GIC_SetPendingIRQ(PendSV_IRQn);
}

/// Set Pending Flags
/// \param[in] flags  Flags to set
void SetPendFlags (uint8_t flags) {
  if ((flags & 1U) != 0U) {
    PendSV_Flag = 1U;
    GIC_SetPendingIRQ(PendSV_IRQn);
  }
}


// External IRQ handling interface
// =================================

/// Enable RTX interrupts
void osRtxIrqUnlock (void) {
  GIC_EnableIRQ(PendSV_IRQn);
}

/// Disable RTX interrupts
void osRtxIrqLock (void) {
  GIC_DisableIRQ(PendSV_IRQn);
}

/// Timer/PendSV interrupt handler
void osRtxIrqHandler (void) {

  if (PendSV_Flag == 0U) {
    osRtxTick_Handler();
  } else {
    ClrPendSV();
    osRtxPendSV_Handler();
  }
}


// External tick timer IRQ interface
// =================================

/// Setup External Tick Timer Interrupt
/// \param[in] irqn  Interrupt number
void ExtTick_SetupIRQ (int32_t irqn) {
  IRQn_Type irq = (IRQn_Type)irqn;
  uint32_t prio;

  PendSV_IRQn = irq;

  // Disable corresponding IRQ first
  GIC_DisableIRQ     (irq);
  GIC_ClearPendingIRQ(irq);

  // Write 0xFF to determine priority level
  GIC_SetPriority(irq, 0xFFU);

  // Read back the number of priority bits
  prio = GIC_GetPriority(irq);

  // Set lowest possible priority
  GIC_SetPriority(irq, prio - 1);

  // Set edge-triggered and 1-N model bits
  GIC_SetLevelModel(irq, 1, 1);

  InterruptHandlerRegister(irq, osRtxIrqHandler);
}

/// Enable External Tick Timer Interrupt
/// \param[in] irqn  Interrupt number
void ExtTick_EnableIRQ (int32_t irqn) {
  GIC_EnableIRQ((IRQn_Type)irqn);
}

/// Disable External Tick Timer Interrupt
/// \param[in] irqn  Interrupt number
void ExtTick_DisableIRQ (int32_t irqn) {
  GIC_DisableIRQ((IRQn_Type)irqn);
}

#endif
