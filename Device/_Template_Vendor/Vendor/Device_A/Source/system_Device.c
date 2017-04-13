/******************************************************************************
 * @file     system_<Device>.c
 * @brief    CMSIS Cortex-A Device Peripheral Access Layer 
 * @version  V1.00
 * @date     30. March 2017
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

#include <stdint.h>
#include "<Device>.h" /* ToDo: replace '<Device>' with your device name */


/*----------------------------------------------------------------------------
  Define clocks
 *----------------------------------------------------------------------------*/
/* ToDo: add here your necessary defines for device initialization
         following is an example for different system frequencies */
#define XTAL            (12000000U)       /* Oscillator frequency             */

#define SYSTEM_CLOCK    (5 * XTAL)


/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
/* ToDo: initialize SystemCoreClock with the system core clock frequency value
         achieved after system intitialization.
         This means system core clock frequency after call to SystemInit() */
uint32_t SystemCoreClock = SYSTEM_CLOCK;  /* System Clock Frequency (Core Clock)*/



/*----------------------------------------------------------------------------
  Clock functions
 *----------------------------------------------------------------------------*/

void SystemCoreClockUpdate (void)            /* Get Core Clock Frequency      */
{
/* ToDo: add code to calculate the system frequency based upon the current
         register settings.
         This function can be used to retrieve the system core clock frequeny
         after user changed register sittings. */
  SystemCoreClock = SYSTEM_CLOCK;
}
/*----------------------------------------------------------------------------
  IRQ Handler Register/Unregister
 *----------------------------------------------------------------------------*/
 /* ToDo: add here your device specific number of interrupt handlers */
 IRQHandler IRQTable[<Device>_IRQ_MAX] = { 0U };

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
/* ToDo: add code to initialize the system
   Do not use global variables because this function is called before
   reaching pre-main. RW section may be overwritten afterwards.          */
  SystemCoreClock = SYSTEM_CLOCK;

  // Invalidate entire Unified TLB
  __set_TLBIALL(0);

  // Invalidate entire branch predictor array
  __set_BPIALL(0);
  __DSB();
  __ISB();

  //  Invalidate instruction cache and flush branch target cache
  __set_ICIALLU(0);
  __DSB();
  __ISB();

  //  Invalidate data cache
  L1C_InvalidateDCacheAll();
  
  // Create Translation Table
  MMU_CreateTranslationTable();

  // Enable MMU
  MMU_Enable();

  // Enable Caches
  L1C_EnableCaches();
  L1C_EnableBTAC();

#if (__L2C_PRESENT == 1) 
  // Enable GIC
  L2C_Enable();
#endif

#if (__GIC_PRESENT == 1) 
  // Enable GIC
  GIC_Enable();
#endif

#if ((__FPU_PRESENT == 1) && (__FPU_USED == 1))
  // Enable FPU
  __FPU_Enable();
#endif
}
