/******************************************************************************
 * @file     startup_ARMCA7.c
 * @brief    CMSIS Device System Source File for ARM Cortex-A7 Device Series
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

#include <ARMCA7.h>

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
  Internal References
 *----------------------------------------------------------------------------*/
void Vectors       (void) __attribute__ ((section("RESET")));
void Reset_Handler (void);

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
__ASM void Vectors(void) {
  PRESERVE8
  
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
__ASM void Reset_Handler(void) {
  PRESERVE8

  // Mask interrupts
  CPSID  if                           

  // Put any cores other than 0 to sleep
  MRC    p15, 0, R0, c0, c0, 5       // Read MPIDR
  ANDS   R0, R0, #3
goToSleep
  WFINE
  BNE    goToSleep

  // Reset SCTLR Settings
  MRC    p15, 0, R0, c1, c0, 0       // Read CP15 System Control register
  BIC    R0, R0, #(0x1 << 12)        // Clear I bit 12 to disable I Cache
  BIC    R0, R0, #(0x1 <<  2)        // Clear C bit  2 to disable D Cache
  BIC    R0, R0, #0x1                // Clear M bit  0 to disable MMU
  BIC    R0, R0, #(0x1 << 11)        // Clear Z bit 11 to disable branch prediction
  BIC    R0, R0, #(0x1 << 13)        // Clear V bit 13 to disable hivecs
  MCR    p15, 0, R0, c1, c0, 0       // Write value back to CP15 System Control register
  ISB

  // Configure ACTLR
  MRC    p15, 0, r0, c1, c0, 1       // Read CP15 Auxiliary Control Register
  ORR    r0, r0, #(1 <<  1)          // Enable L2 prefetch hint (UNK/WI since r4p1)
  MCR    p15, 0, r0, c1, c0, 1       // Write CP15 Auxiliary Control Register

  // Set Vector Base Address Register (VBAR) to point to this application's vector table
  LDR    R0, =Vectors
  MCR    p15, 0, R0, c12, c0, 0

  // Setup Stack for each exceptional mode
  IMPORT |Image$$FIQ_STACK$$ZI$$Limit|
  IMPORT |Image$$IRQ_STACK$$ZI$$Limit|
  IMPORT |Image$$SVC_STACK$$ZI$$Limit|
  IMPORT |Image$$ABT_STACK$$ZI$$Limit|
  IMPORT |Image$$UND_STACK$$ZI$$Limit|
  IMPORT |Image$$ARM_LIB_STACK$$ZI$$Limit|
  CPS    #0x11
  LDR    SP, =|Image$$FIQ_STACK$$ZI$$Limit|
  CPS    #0x12
  LDR    SP, =|Image$$IRQ_STACK$$ZI$$Limit|
  CPS    #0x13
  LDR    SP, =|Image$$SVC_STACK$$ZI$$Limit|
  CPS    #0x17
  LDR    SP, =|Image$$ABT_STACK$$ZI$$Limit|
  CPS    #0x1B
  LDR    SP, =|Image$$UND_STACK$$ZI$$Limit|
  CPS    #0x1F
  LDR    SP, =|Image$$ARM_LIB_STACK$$ZI$$Limit|

  // Call SystemInit
  IMPORT SystemInit
  BL     SystemInit

  // Unmask interrupts
  CPSIE  if

  // Call __main
  IMPORT __main
  BL     __main
}

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void Default_Handler(void) {
  while(1);
}
