/******************************************************************************
 * @file     system_ARMCA9.c
 * @brief    CMSIS Device System Source File for ARM Cortex-A9 Device Series
 * @version  V1.00
 * @date     22 Feb 2017
 *
 * @note
 *
 ******************************************************************************/
/*
 * Copyright (c) 2009-2017 ARM Limited. All rights reserved.
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

#include <ARMCA9.h>

#define  SYSTEM_CLOCK  12000000U

/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
uint32_t SystemCoreClock = SYSTEM_CLOCK;

/*----------------------------------------------------------------------------
  System Core Clock update function
 *----------------------------------------------------------------------------*/
void SystemCoreClockUpdate (void)
{
  SystemCoreClock = SYSTEM_CLOCK;
}

/*----------------------------------------------------------------------------
  IRQ Handler Register/Unregister
 *----------------------------------------------------------------------------*/
IRQHandler IRQTable[40U] = { 0U };

uint32_t IRQCount = sizeof IRQTable / 4U;

uint32_t InterruptHandlerRegister (IRQn_Type irq, IRQHandler handler)
{
  if (irq < IRQCount) {
    IRQTable[irq] = handler;
    return 0U;
  }
  else {
    return 1U;
  }
}

uint32_t InterruptHandlerUnregister (IRQn_Type irq)
{
  if (irq < IRQCount) {
    IRQTable[irq] = 0U;
    return 0U;
  }
  else {
    return 1U;
  }
}

/*----------------------------------------------------------------------------
  System Initialization
 *----------------------------------------------------------------------------*/
void SystemInit (void)
{
/* do not use global variables because this function is called before
   reaching pre-main. RW section may be overwritten afterwards.          */
  GIC_Enable();
  L1C_EnableCaches();
  L1C_EnableBTAC();
  __FPU_Enable();
}
