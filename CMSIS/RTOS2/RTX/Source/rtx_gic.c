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

#include "rtx_os.h"


#if ((__ARM_ARCH_7A__ == 1U) && (__GIC_PRESENT == 1U))

extern IRQHandler IRQTable[];
extern uint32_t   IRQCount;

static uint32_t ID0_Active;

int32_t osRtxIrqGetId (void) {
  IRQn_Type irq;
  int32_t id;
  uint32_t prio;

  /* Dummy read to avoid GIC 390 errata 801120 */
  GIC_GetHighPendingIRQ();

  irq = GIC_AcknowledgePending();

  __DSB();

  /* Workaround GIC 390 errata 733075 (GIC-390_Errata_Notice_v6.pdf, 09-Jul-2014)  */
  /* The following workaround code is for a single-core system.  It would be       */
  /* different in a multi-core system.                                             */
  /* If the ID is 0 or 0x3FE or 0x3FF, then the GIC CPU interface may be locked-up */
  /* so unlock it, otherwise service the interrupt as normal.                      */
  /* Special IDs 1020=0x3FC and 1021=0x3FD are reserved values in GICv1 and GICv2  */
  /* so will not occur here.                                                       */
  id = (int32_t)irq;

  if ((irq == 0U) || (irq >= 0x3FEU)) {
    /* Unlock the CPU interface with a dummy write to Interrupt Priority Register */
    prio = GIC_GetPriority((IRQn_Type)0);
    GIC_SetPriority ((IRQn_Type)0, prio);

    __DSB();

    if (id != 0U) {
      /* Not 0 (spurious interrupt) */
      id = -1;
    }
    else if ((GIC_GetIRQStatus (irq) & 1U) == 0U) {
      /* Not active (spurious interrupt) */
      id = -1;
    }
    else if (ID0_Active == 1U) {
      /* Already seen (spurious interrupt) */
      id = -1;
    }
    else {
      ID0_Active = 1U;
    }
    /* End of Workaround GIC 390 errata 733075 */
  }

  return (id);
}

uint32_t osRtxIrqGetHandler (int32_t id) {
  IRQHandler h;

  if (id < IRQCount) {
    h = IRQTable[id];
  } else {
    h = NULL;
  }

  return ((uint32_t)h);
}

void osRtxIrqSetEnd (int32_t id) {

  GIC_EndInterrupt ((IRQn_Type)id);

  if (id == 0) {
    ID0_Active = 0U;
  }
}

void osRtxIrqEnableTick  (void) {
  GIC_EnableIRQ((IRQn_Type)osRtxInfo.tick_irqn);
}

void osRtxIrqDisableTick (void) {
  GIC_DisableIRQ((IRQn_Type)osRtxInfo.tick_irqn);
}

#endif

