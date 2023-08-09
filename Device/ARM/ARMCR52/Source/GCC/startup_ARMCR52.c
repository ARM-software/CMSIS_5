/******************************************************************************
 * @file     startup_ARMCR52.c
 * @brief    CMSIS Core Device Startup File for Cortex-R52 Device
 * @version  V1.0.0
 * @date     15. June 2020
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

#if defined (ARMCR52)
  #include "ARMCR52.h"
#elif defined (ARMCR52_FP)
  #include "ARMCR52_FP.h"
#elif defined (ARMCR52_DSP_DP_FP)
  #include "ARMCR52_DSP_DP_FP.h"
#else
  #error device not specified!
#endif

/*----------------------------------------------------------------------------
  External References
 *----------------------------------------------------------------------------*/
extern uint32_t __INITIAL_SP;

/*----------------------------------------------------------------------------
  Internal References
 *----------------------------------------------------------------------------*/
void EL2_Default_Handler(void);
void EL1_Default_Handler(void);
void __NAKED __NO_RETURN EL2_Reset_Handler(void);

/*----------------------------------------------------------------------------
  Exception / Interrupt Handler
 *----------------------------------------------------------------------------*/
void EL2_Undef_Handler    (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void EL2_SVC_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void EL2_Prefetch_Handler (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void EL2_Abort_Handler    (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void EL2_Reserved_Handler (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void EL2_IRQ_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void EL2_FIQ_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));

void EL1_Reset_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_Undef_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_SVC_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_Prefetch_Handler (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_Abort_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_Reserved_Handler (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_IRQ_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void EL1_FIQ_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));

/*----------------------------------------------------------------------------
  Exception / Interrupt Vector table
 *----------------------------------------------------------------------------*/

#if defined ( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

void __VECTOR_TABLE_ATTRIBUTE __VECTOR_TABLE_EL2(void)
{
  __ASM volatile(
  "LDR    PC,=EL2_Reset_Handler  \n"
  "B.W    EL2_Undef_Handler      \n"
  "B.W    EL2_SVC_Handler        \n"
  "B.W    EL2_Prefetch_Handler   \n"
  "B.W    EL2_Abort_Handler      \n"
  "B.W    EL2_Reserved_Handler   \n"
  "B.W    EL2_IRQ_Handler        \n"
  "B.W    EL2_FIQ_Handler        \n"
  );
}

void __VECTOR_TABLE_ATTRIBUTE __VECTOR_TABLE_EL1(void)
{
  __ASM volatile(
  "B.W    EL1_Reset_Handler      \n"
  "B.W    EL1_Undef_Handler      \n"
  "B.W    EL1_SVC_Handler        \n"
  "B.W    EL1_Prefetch_Handler   \n"
  "B.W    EL1_Abort_Handler      \n"
  "B.W    EL1_Reserved_Handler   \n"
  "B.W    EL1_IRQ_Handler        \n"
  "B.W    EL1_FIQ_Handler        \n"
  );
}

#if defined ( __GNUC__ )
#pragma GCC diagnostic pop
#endif

/*----------------------------------------------------------------------------
  Switch from EL2 to EL1 mode
 *----------------------------------------------------------------------------*/
void __NAKED __NO_RETURN SwitchEL2toEL1(uint32_t func)
{
  register CPSR_Type spsr_hyp;

  __set_ELR_Hyp(func);

  spsr_hyp.w = __get_SPSR_Hyp();
#if defined(__thumb2__) && (__thumb2__==1)
  /* Thumb instruction set and supervisor mode */
  spsr_hyp.b.M = CPSR_M_SVC;
  spsr_hyp.b.T = 1;
#else
  /* ARM instruction set and supervisor mode */
  spsr_hyp.b.M = CPSR_M_SVC;
#endif
  __set_SPSR_Hyp(spsr_hyp.w);

  __ERET();

}

/*----------------------------------------------------------------------------
  Reset Handler called on controller reset
 *----------------------------------------------------------------------------*/
void __NAKED __NO_RETURN EL2_Reset_Handler(void)
{
  __EARLY_INIT();
  __set_SP((uint32_t)&__INITIAL_SP);
  SystemInit(); /* CMSIS System Initialization */

  /* ARMv8-R cores are in EL2 (hypervisor mode) after reset, and we need
     to first descend to EL1 (supervisor mode) before the traditional SP
     setting code can be run */
  SwitchEL2toEL1((uint32_t)&__PROGRAM_START); /* Enter PreMain (C library entry point) */
}

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void EL2_Default_Handler(void)
{
  while(1);
}

void EL1_Default_Handler(void)
{
  while(1);
}
