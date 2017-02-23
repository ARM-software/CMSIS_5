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
#include <stdint.h>

extern void $Super$$main(void);
__asm void __FPU_Enable(void);

// Flag indicates whether inside an ISR, and the depth of nesting.  0 = not in ISR.
uint32_t IRQNestLevel = 0;
// Flag used to workaround GIC 390 errata 733075
uint32_t seen_id0_active = 0;


/**
 * Initialize the memory subsystem.
 *
 * @param  none
 * @return none
 *
 * @brief Initialize the memory subsystem, including enabling the cache and BTAC. Requires PL1, so implemented as an SVC in case threads are USR mode.
 */
#pragma push
#pragma arm
void __svc(1) EnableCaches(void);
void __SVC_1(void) {

/* Before enabling the caches, the instruction cache, the data cache, TLB, and BTAC must have been invalidated.
 * You are not required to invalidate the main TLB, even though it is recommended for safety
 * reasons. This ensures compatibility with future revisions of the processor. */

//  unsigned int l2_id;

  /* After MMU is enabled and data has been invalidated, enable caches and BTAC */
  L1C_EnableCaches();
  L1C_EnableBTAC();

  /* If L2C-310 is present, Invalidate and Enable L2 cache here */
//  l2_id = L2C_GetID();
//  if (l2_id)
//  {
//     L2C_InvAllByWay();
//     L2C_Enable();
//  }
}
#pragma pop

IRQHandler IRQTable[] = {
    0, //IRQ 0
    0, //IRQ 1
    0, //IRQ 2
    0, //IRQ 3
    0, //IRQ 4
    0, //IRQ 5
    0, //IRQ 6
    0, //IRQ 7
    0, //IRQ 8
    0, //IRQ 9
    0, //IRQ 10
    0, //IRQ 11
    0, //IRQ 12
    0, //IRQ 13
    0, //IRQ 14
    0, //IRQ 15
    0, //IRQ 16
    0, //IRQ 17
    0, //IRQ 18
    0, //IRQ 19
    0, //IRQ 20
    0, //IRQ 21
    0, //IRQ 22
    0, //IRQ 23
    0, //IRQ 24
    0, //IRQ 25
    0, //IRQ 26
    0, //IRQ 27
    0, //IRQ 28
    0, //IRQ 29
    0, //IRQ 30
    0, //IRQ 31
    0, //IRQ 32
    0, //IRQ 33
    0, //IRQ 34
    0, //IRQ 35
    0, //IRQ 36
    0, //IRQ 37
    0, //IRQ 38
    0, //IRQ 39
    0  //IRQ 40
};
uint32_t IRQCount = sizeof IRQTable / 4;

uint32_t InterruptHandlerRegister (IRQn_Type irq, IRQHandler handler)
{
    if (irq < IRQCount) {
        IRQTable[irq] = handler;
        return 0;
    }
    else {
        return 1;
    }
}

uint32_t InterruptHandlerUnregister (IRQn_Type irq)
{
    if (irq < IRQCount) {
        IRQTable[irq] = 0;
        return 0;
    }
    else {
        return 1;
    }
}

/**
 * Initialize the system
 *
 * @param  none
 * @return none
 *
 * @brief  Setup the microcontroller system.
 *         Initialize the System.
 */
void SystemInit (void)
{
/*       do not use global variables because this function is called before
         reaching pre-main. RW section may be overwritten afterwards.          */
    GIC_Enable();
}

void $Sub$$main(void)
{
#ifdef __CMSIS_RTOS
  extern void PendSV_Handler(uint32_t);
  extern void OS_Tick_Handler(uint32_t);
  InterruptHandlerRegister(SGI0_IRQn     , PendSV_Handler);
  InterruptHandlerRegister(PrivTimer_IRQn, OS_Tick_Handler);
  EnableCaches();
#endif
  
  $Super$$main(); //Call main
}

//Fault Status Register (IFSR/DFSR) definitions
#define FSR_ALIGNMENT_FAULT                  0x01   //DFSR only. Fault on first lookup
#define FSR_INSTRUCTION_CACHE_MAINTENANCE    0x04   //DFSR only - async/external
#define FSR_SYNC_EXT_TTB_WALK_FIRST          0x0c   //sync/external
#define FSR_SYNC_EXT_TTB_WALK_SECOND         0x0e   //sync/external
#define FSR_SYNC_PARITY_TTB_WALK_FIRST       0x1c   //sync/external
#define FSR_SYNC_PARITY_TTB_WALK_SECOND      0x1e   //sync/external
#define FSR_TRANSLATION_FAULT_FIRST          0x05   //MMU Fault - internal
#define FSR_TRANSLATION_FAULT_SECOND         0x07   //MMU Fault - internal
#define FSR_ACCESS_FLAG_FAULT_FIRST          0x03   //MMU Fault - internal
#define FSR_ACCESS_FLAG_FAULT_SECOND         0x06   //MMU Fault - internal
#define FSR_DOMAIN_FAULT_FIRST               0x09   //MMU Fault - internal
#define FSR_DOMAIN_FAULT_SECOND              0x0b   //MMU Fault - internal
#define FSR_PERMISSION_FAULT_FIRST           0x0f   //MMU Fault - internal
#define FSR_PERMISSION_FAULT_SECOND          0x0d   //MMU Fault - internal
#define FSR_DEBUG_EVENT                      0x02   //internal
#define FSR_SYNC_EXT_ABORT                   0x08   //sync/external
#define FSR_TLB_CONFLICT_ABORT               0x10   //sync/external
#define FSR_LOCKDOWN                         0x14   //internal
#define FSR_COPROCESSOR_ABORT                0x1a   //internal
#define FSR_SYNC_PARITY_ERROR                0x19   //sync/external
#define FSR_ASYNC_EXTERNAL_ABORT             0x16   //DFSR only - async/external
#define FSR_ASYNC_PARITY_ERROR               0x18   //DFSR only - async/external

void CDAbtHandler(uint32_t DFSR, uint32_t DFAR, uint32_t LR) {
  uint32_t FS = (DFSR & (1 << 10)) >> 6 | (DFSR & 0x0f); //Store Fault Status

  switch(FS) {
    //Synchronous parity errors - retry
    case FSR_SYNC_PARITY_ERROR:
    case FSR_SYNC_PARITY_TTB_WALK_FIRST:
    case FSR_SYNC_PARITY_TTB_WALK_SECOND:
        return;

    //Your code here. Value in DFAR is invalid for some fault statuses.
    case FSR_ALIGNMENT_FAULT:
    case FSR_INSTRUCTION_CACHE_MAINTENANCE:
    case FSR_SYNC_EXT_TTB_WALK_FIRST:
    case FSR_SYNC_EXT_TTB_WALK_SECOND:
    case FSR_TRANSLATION_FAULT_FIRST:
    case FSR_TRANSLATION_FAULT_SECOND:
    case FSR_ACCESS_FLAG_FAULT_FIRST:
    case FSR_ACCESS_FLAG_FAULT_SECOND:
    case FSR_DOMAIN_FAULT_FIRST:
    case FSR_DOMAIN_FAULT_SECOND:
    case FSR_PERMISSION_FAULT_FIRST:
    case FSR_PERMISSION_FAULT_SECOND:
    case FSR_DEBUG_EVENT:
    case FSR_SYNC_EXT_ABORT:
    case FSR_TLB_CONFLICT_ABORT:
    case FSR_LOCKDOWN:
    case FSR_COPROCESSOR_ABORT:
    case FSR_ASYNC_EXTERNAL_ABORT: //DFAR invalid
    case FSR_ASYNC_PARITY_ERROR:   //DFAR invalid
    default:
      while(1);
  }
}

void CPAbtHandler(uint32_t IFSR, uint32_t IFAR, uint32_t LR) {
  uint32_t FS = (IFSR & (1 << 10)) >> 6 | (IFSR & 0x0f); //Store Fault Status

  switch(FS) {
    //Synchronous parity errors - retry
    case FSR_SYNC_PARITY_ERROR:
    case FSR_SYNC_PARITY_TTB_WALK_FIRST:
    case FSR_SYNC_PARITY_TTB_WALK_SECOND:
      return;

    //Your code here. Value in IFAR is invalid for some fault statuses.
    case FSR_SYNC_EXT_TTB_WALK_FIRST:
    case FSR_SYNC_EXT_TTB_WALK_SECOND:
    case FSR_TRANSLATION_FAULT_FIRST:
    case FSR_TRANSLATION_FAULT_SECOND:
    case FSR_ACCESS_FLAG_FAULT_FIRST:
    case FSR_ACCESS_FLAG_FAULT_SECOND:
    case FSR_DOMAIN_FAULT_FIRST:
    case FSR_DOMAIN_FAULT_SECOND:
    case FSR_PERMISSION_FAULT_FIRST:
    case FSR_PERMISSION_FAULT_SECOND:
    case FSR_DEBUG_EVENT: //IFAR invalid
    case FSR_SYNC_EXT_ABORT:
    case FSR_TLB_CONFLICT_ABORT:
    case FSR_LOCKDOWN:
    case FSR_COPROCESSOR_ABORT:
    default:
      while(1);
  }
}

//returns amount to decrement lr by
//this will be 0 when we have emulated the instruction and want to execute the next instruction
//this will be 2 when we have performed some maintenance and want to retry the instruction in Thumb (state == 2)
//this will be 4 when we have performed some maintenance and want to retry the instruction in ARM   (state == 4)
uint32_t CUndefHandler(uint32_t opcode, uint32_t state, uint32_t LR) {
  const int THUMB = 2;
  const int ARM = 4;
  //Lazy VFP/NEON initialisation and switching

  // (ARM ARM section A7.5) VFP data processing instruction?
  // (ARM ARM section A7.6) VFP/NEON register load/store instruction?
  // (ARM ARM section A7.8) VFP/NEON register data transfer instruction?
  // (ARM ARM section A7.9) VFP/NEON 64-bit register data transfer instruction?
  if ((state == ARM   && ((opcode & 0x0C000000) >> 26 == 0x03)) ||
      (state == THUMB && ((opcode & 0xEC000000) >> 26 == 0x3B))) {
    if (((opcode & 0x00000E00) >> 9) == 5) {
      __FPU_Enable();
      return state;
    }
  }

  // (ARM ARM section A7.4) NEON data processing instruction?
  if ((state == ARM   && ((opcode & 0xFE000000) >> 24 == 0xF2)) ||
      (state == THUMB && ((opcode & 0xEF000000) >> 24 == 0xEF)) ||
      // (ARM ARM section A7.7) NEON load/store instruction?
      (state == ARM   && ((opcode >> 24) == 0xF4)) ||
      (state == THUMB && ((opcode >> 24) == 0xF9))) {
    __FPU_Enable(); 
    return state;
  }

  //Add code here for other Undef cases
  while(1);
}
