/**************************************************************************//**
 * @file     system_ARMCA53.c
 * @brief    CMSIS Device System Source File for
 *           ARMCA53 Device
 * @version  V1.0.1
 * @date     20. August 2020
 ******************************************************************************/
/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
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

#if defined (ARMCA53)
  #include "ARMCA53.h"
#elif defined (ARMCA53_FP)
  #include "ARMCA53_FP.h"
#elif defined (ARMCA53_DSP_DP_FP)
  #include "ARMCA53_DSP_DP_FP.h"
#else
  #error device not specified!
#endif

extern void MMU_CreateTranslationTable(void);

/*----------------------------------------------------------------------------
  Define clocks
 *----------------------------------------------------------------------------*/
#define  SYSTEM_CLOCK  12000000U

/*----------------------------------------------------------------------------
  External References
 *----------------------------------------------------------------------------*/
extern uint64_t __VECTOR_TABLE_EL3;
extern uint64_t __VECTOR_TABLE_EL2;
extern uint64_t __VECTOR_TABLE_EL1;

/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
uint32_t SystemCoreClock = SYSTEM_CLOCK; /* System Core Clock Frequency */

/*----------------------------------------------------------------------------
  System Core Clock update function
 *----------------------------------------------------------------------------*/
void SystemCoreClockUpdate (void)
{
  SystemCoreClock = SYSTEM_CLOCK;
}

/*----------------------------------------------------------------------------
  System initialization function
 *----------------------------------------------------------------------------*/
void SystemInit (void)
{

  __set_VBAR_EL3((uint64_t)&__VECTOR_TABLE_EL3);
  __set_VBAR_EL2((uint64_t)&__VECTOR_TABLE_EL2);
  __set_VBAR_EL1((uint64_t)&__VECTOR_TABLE_EL1);

  // Invalidate entire Unified TLB
  MMU_InvalidateTLB();

  // Invalidate entire branch predictor array
  L1C_InvalidateBTAC();

  // Invalidate instruction cache and flush branch target cache
  L1C_InvalidateICacheAll();

  // Invalidate data cache
  L1C_InvalidateDCacheAll();

#if ((__FPU_PRESENT == 1) && (__FPU_USED == 1))
  // Enable FPU
  __FPU_Enable();
#endif

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

  SystemCoreClock = SYSTEM_CLOCK;
}
