/******************************************************************************
 * @file     startup_ARMCA53.c
 * @brief    CMSIS Core Device Startup File for Cortex-A53 Device
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

/*----------------------------------------------------------------------------
  External References
 *----------------------------------------------------------------------------*/
extern uint32_t __INITIAL_SP;

extern __NO_RETURN void __PROGRAM_START(void);

/*----------------------------------------------------------------------------
  Internal References
 *----------------------------------------------------------------------------*/
void __NO_RETURN EL3_Default_Handler(void);
void __NO_RETURN EL2_Default_Handler(void);
void __NO_RETURN EL1_Default_Handler(void);
void __NO_RETURN Reset_Handler(void);
void __NO_RETURN Reset_Handler_C(void);


/*----------------------------------------------------------------------------
  Exception / Interrupt Handler
 *----------------------------------------------------------------------------*/
void __NO_RETURN Current_EL3_SP0_Sync_Handler     (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SP0_IRQ_Handler      (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SP0_FIQ_Handler      (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SP0_SError_Handler   (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SPx_Sync_Handler     (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SPx_IRQ_Handler      (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SPx_FIQ_Handler      (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Current_EL3_SPx_SError_Handler   (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch64_Sync_Handler   (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch64_IRQ_Handler    (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch64_FIQ_Handler    (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch64_SError_Handler (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch32_Sync_Handler   (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch32_IRQ_Handler    (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch32_FIQ_Handler    (void) __attribute__ ((weak, alias("EL3_Default_Handler")));
void __NO_RETURN Lower_EL3_AArch32_SError_Handler (void) __attribute__ ((weak, alias("EL3_Default_Handler")));

void __NO_RETURN Current_EL2_SP0_Sync_Handler     (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SP0_IRQ_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SP0_FIQ_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SP0_SError_Handler   (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SPx_Sync_Handler     (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SPx_IRQ_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SPx_FIQ_Handler      (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Current_EL2_SPx_SError_Handler   (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch64_Sync_Handler   (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch64_IRQ_Handler    (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch64_FIQ_Handler    (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch64_SError_Handler (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch32_Sync_Handler   (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch32_IRQ_Handler    (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch32_FIQ_Handler    (void) __attribute__ ((weak, alias("EL2_Default_Handler")));
void __NO_RETURN Lower_EL2_AArch32_SError_Handler (void) __attribute__ ((weak, alias("EL2_Default_Handler")));

void __NO_RETURN Current_EL1_SP0_Sync_Handler     (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SP0_IRQ_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SP0_FIQ_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SP0_SError_Handler   (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SPx_Sync_Handler     (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SPx_IRQ_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SPx_FIQ_Handler      (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Current_EL1_SPx_SError_Handler   (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch64_Sync_Handler   (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch64_IRQ_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch64_FIQ_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch64_SError_Handler (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch32_Sync_Handler   (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch32_IRQ_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch32_FIQ_Handler    (void) __attribute__ ((weak, alias("EL1_Default_Handler")));
void __NO_RETURN Lower_EL1_AArch32_SError_Handler (void) __attribute__ ((weak, alias("EL1_Default_Handler")));

__ASM("\n .macro ventry label \n .align 7 \n b \\label \n .endm");

/*----------------------------------------------------------------------------
  Exception / Interrupt Vector table
 *----------------------------------------------------------------------------*/

#if defined ( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

void __VECTOR_TABLE_ATTRIBUTE __VECTOR_TABLE_EL3(void)
{
  __ASM volatile(
  "ventry Current_EL3_SP0_Sync_Handler     \n"
  "ventry Current_EL3_SP0_IRQ_Handler      \n"
  "ventry Current_EL3_SP0_FIQ_Handler      \n"
  "ventry Current_EL3_SP0_SError_Handler   \n"
  "ventry Current_EL3_SPx_Sync_Handler     \n"
  "ventry Current_EL3_SPx_IRQ_Handler      \n"
  "ventry Current_EL3_SPx_FIQ_Handler      \n"
  "ventry Current_EL3_SPx_SError_Handler   \n"
  "ventry Lower_EL3_AArch64_Sync_Handler   \n"
  "ventry Lower_EL3_AArch64_IRQ_Handler    \n"
  "ventry Lower_EL3_AArch64_FIQ_Handler    \n"
  "ventry Lower_EL3_AArch64_SError_Handler \n"
  "ventry Lower_EL3_AArch32_Sync_Handler   \n"
  "ventry Lower_EL3_AArch32_IRQ_Handler    \n"
  "ventry Lower_EL3_AArch32_FIQ_Handler    \n"
  "ventry Lower_EL3_AArch32_SError_Handler \n"
  );
}
 
void __VECTOR_TABLE_ATTRIBUTE __VECTOR_TABLE_EL2(void)
{
  __ASM volatile(
  "ventry Current_EL2_SP0_Sync_Handler     \n"
  "ventry Current_EL2_SP0_IRQ_Handler      \n"
  "ventry Current_EL2_SP0_FIQ_Handler      \n"
  "ventry Current_EL2_SP0_SError_Handler   \n"
  "ventry Current_EL2_SPx_Sync_Handler     \n"
  "ventry Current_EL2_SPx_IRQ_Handler      \n"
  "ventry Current_EL2_SPx_FIQ_Handler      \n"
  "ventry Current_EL2_SPx_SError_Handler   \n"
  "ventry Lower_EL2_AArch64_Sync_Handler   \n"
  "ventry Lower_EL2_AArch64_IRQ_Handler    \n"
  "ventry Lower_EL2_AArch64_FIQ_Handler    \n"
  "ventry Lower_EL2_AArch64_SError_Handler \n"
  "ventry Lower_EL2_AArch32_Sync_Handler   \n"
  "ventry Lower_EL2_AArch32_IRQ_Handler    \n"
  "ventry Lower_EL2_AArch32_FIQ_Handler    \n"
  "ventry Lower_EL2_AArch32_SError_Handler \n"
  );
}

void __VECTOR_TABLE_ATTRIBUTE __VECTOR_TABLE_EL1(void)
{
  __ASM volatile(
  "ventry Current_EL1_SP0_Sync_Handler     \n"
  "ventry Current_EL1_SP0_IRQ_Handler      \n"
  "ventry Current_EL1_SP0_FIQ_Handler      \n"
  "ventry Current_EL1_SP0_SError_Handler   \n"
  "ventry Current_EL1_SPx_Sync_Handler     \n"
  "ventry Current_EL1_SPx_IRQ_Handler      \n"
  "ventry Current_EL1_SPx_FIQ_Handler      \n"
  "ventry Current_EL1_SPx_SError_Handler   \n"
  "ventry Lower_EL1_AArch64_Sync_Handler   \n"
  "ventry Lower_EL1_AArch64_IRQ_Handler    \n"
  "ventry Lower_EL1_AArch64_FIQ_Handler    \n"
  "ventry Lower_EL1_AArch64_SError_Handler \n"
  "ventry Lower_EL1_AArch32_Sync_Handler   \n"
  "ventry Lower_EL1_AArch32_IRQ_Handler    \n"
  "ventry Lower_EL1_AArch32_FIQ_Handler    \n"
  "ventry Lower_EL1_AArch32_SError_Handler \n"
  );
}

#if defined ( __GNUC__ )
#pragma GCC diagnostic pop
#endif

/*----------------------------------------------------------------------------
  __EARLY_INIT routine custom version
 *----------------------------------------------------------------------------*/
#undef __EARLY_INIT
#define __EARLY_INIT

/*----------------------------------------------------------------------------
  Reset Handler called on controller reset
 *----------------------------------------------------------------------------*/
__ASM(
  "\t.section	.startup,\"ax\",@progbits \n"
  "\t.align	2 \n"
  "\t.globl Reset_Handler \n"
  "\t.type Reset_Handler, %function \n"
  "Reset_Handler: \n"

  __EARLY_INIT

  "#__set_SP((uint64)&__INITIAL_SP) \n" \
  "\tadrp x0, :pg_hi21:__EL3StackTop \n"
  "\tadd	x0, x0, :lo12:__EL3StackTop \n"
  "\tmov	sp, x0 \n"
  "\tb Reset_Handler_C \n"
  "\t.size	Reset_Handler, .-Reset_Handler \n"
);

/*----------------------------------------------------------------------------
  Reset Handler C version called by "naked" Reset Handler
 *----------------------------------------------------------------------------*/
void __NO_RETURN __attribute__((section(".startup"))) Reset_Handler_C(void)
{
  SystemInit();                      /* CMSIS System Initialization */

  __PROGRAM_START();                 /* Enter PreMain (C library entry point) */
}

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void EL3_Default_Handler(void)
{
  while(1);
}

void EL2_Default_Handler(void)
{
  while(1);
}

void EL1_Default_Handler(void)
{
  while(1);
}
