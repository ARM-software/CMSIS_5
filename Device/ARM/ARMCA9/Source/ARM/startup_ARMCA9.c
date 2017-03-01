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

/*----------------------------------------------------------------------------
  Definitions
 *----------------------------------------------------------------------------*/
#define USR_MODE 0x10            // User mode
#define FIQ_MODE 0x11            // Fast Interrupt Request mode
#define IRQ_MODE 0x12            // Interrupt Request mode
#define SVC_MODE 0x13            // Supervisor mode
#define ABT_MODE 0x17            // Abort mode
#define UND_MODE 0x1B            // Undefined Instruction mode
#define SYS_MODE 0x1F            // System mode

/*----------------------------------------------------------------------------
  Linker generated Symbols
 *----------------------------------------------------------------------------*/
extern uint32_t Image$$FIQ_STACK$$ZI$$Limit;
extern uint32_t Image$$IRQ_STACK$$ZI$$Limit;
extern uint32_t Image$$SVC_STACK$$ZI$$Limit;
extern uint32_t Image$$ABT_STACK$$ZI$$Limit;
extern uint32_t Image$$UND_STACK$$ZI$$Limit;
extern uint32_t Image$$ARM_LIB_STACK$$ZI$$Limit;

/*----------------------------------------------------------------------------
  Internal References
 *----------------------------------------------------------------------------*/
void Reset_Handler(void);

/*----------------------------------------------------------------------------
  Exception / Interrupt Handler
 *----------------------------------------------------------------------------*/
void Undef_Handler (void) __attribute__ ((weak, alias("Default_Handler")));
void SVC_Handler   (void) __attribute__ ((weak, alias("Default_Handler")));
void PAbt_Handler  (void) __attribute__ ((weak, alias("Default_Handler")));
void DAbt_Handler  (void) __attribute__ ((weak, alias("Default_Handler")));
void IRQ_Handler   (void) __attribute__ ((weak, alias("Default_Handler")));
void FIQ_Handler   (void) __attribute__ ((weak, alias("Default_Handler")));

/*----------------------------------------------------------------------------
  Exception / Interrupt Vector Table
 *----------------------------------------------------------------------------*/
void Vectors(void) __attribute__ ((section("RESET")));
__ASM void Vectors(void) {
  IMPORT Reset_Handler
  IMPORT Undef_Handler
  IMPORT SVC_Handler
  IMPORT PAbt_Handler
  IMPORT DAbt_Handler
  IMPORT IRQ_Handler
  IMPORT FIQ_Handler
  LDR    PC, =Reset_Handler
  LDR    PC, =Undef_Handler
  LDR    PC, =SVC_Handler
  LDR    PC, =PAbt_Handler
  LDR    PC, =DAbt_Handler
  NOP
  LDR    PC, =IRQ_Handler
  LDR    PC, =FIQ_Handler
}

/*----------------------------------------------------------------------------
  Reset Handler called on controller reset
 *----------------------------------------------------------------------------*/
void Reset_Handler(void) {
uint32_t reg;

  // Put any cores other than 0 to sleep
  if ((__get_MPIDR()&3U)!=0) __WFI();

  reg  = __get_SCTLR();  // Read CP15 System Control register
  reg &= ~(0x1 << 12);   // Clear I bit 12 to disable I Cache
  reg &= ~(0x1 <<  2);   // Clear C bit  2 to disable D Cache
  reg &= ~(0x1 <<  0);   // Clear M bit  0 to disable MMU
  reg &= ~(0x1 << 11);   // Clear Z bit 11 to disable branch prediction
  reg &= ~(0x1 << 13);   // Clear V bit 13 to disable hivecs
  __set_SCTLR(reg);      // Write value back to CP15 System Control register
  __ISB();

  reg  = __get_ACTRL();  // Read CP15 Auxiliary Control Register
  reg |= (0x1 <<  1);    // Enable L2 prefetch hint (UNK/WI since r4p1)
  __set_ACTRL(reg);      // Write CP15 Auxiliary Control Register

  __set_VBAR((uint32_t)((uint32_t*)&Vectors));

  // Setup Stack for each exceptional mode
  __set_mode(FIQ_MODE);
  __set_SP((uint32_t)&Image$$FIQ_STACK$$ZI$$Limit);
  __set_mode(IRQ_MODE);
  __set_SP((uint32_t)&Image$$IRQ_STACK$$ZI$$Limit);
  __set_mode(SVC_MODE);
  __set_SP((uint32_t)&Image$$SVC_STACK$$ZI$$Limit);
  __set_mode(ABT_MODE);
  __set_SP((uint32_t)&Image$$ABT_STACK$$ZI$$Limit);
  __set_mode(UND_MODE);
  __set_SP((uint32_t)&Image$$UND_STACK$$ZI$$Limit);
  __set_mode(SYS_MODE);
  __set_SP((uint32_t)&Image$$ARM_LIB_STACK$$ZI$$Limit);

  // Create Translation Table
  MMU_CreateTranslationTable();

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
  __L1C_CleanInvalidateCache(0);

  // Invalidate entire Unified TLB
  __set_TLBIALL(0);
  // Invalidate entire branch predictor array
  __set_BPIALL(0);
  __DSB();
  __ISB();
  // Invalidate instruction cache and flush branch target cache
  __set_ICIALLU(0);
  __DSB();
  __ISB();

  // Enable MMU, but leave caches disabled (they will be enabled later)
  reg  = __get_SCTLR();  // Read CP15 System Control register
  reg |=  (0x1 << 29);   // Set AFE bit 29 to enable simplified access permissions model
  reg &= ~(0x1 << 28);   // Clear TRE bit 28 to disable TEX remap
  reg &= ~(0x1 << 12);   // Clear I bit 12 to disable I Cache
  reg &= ~(0x1 <<  2);   // Clear C bit  2 to disable D Cache
  reg &= ~(0x1 <<  1);   // Clear A bit  1 to disable strict alignment fault checking
  reg |=  (0x1 <<  0);	 // Set M bit 0 to enable MMU
  __set_SCTLR(reg);      // Write CP15 System Control register

  SystemInit();

  extern void __main(void);
  __main();
}

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void Default_Handler(void) {
	while(1);
}
