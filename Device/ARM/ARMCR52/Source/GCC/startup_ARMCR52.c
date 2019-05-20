/**************************************************************************//**
 * @file     startup_ARMCR52.c
 * @brief    CMSIS Core Device Startup File for
 *           ARMCR52 Device
 * @version
 * @date
 ******************************************************************************/
/*
 * Copyright (c) 2019 Arm Limited. All rights reserved.
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

/*
 //-------- <<< Use Configuration Wizard in Context Menu >>> ------------------
*/

#if defined (ARMCR52)
  #include "ARMCR52.h"
#else
  #error device not specified!
#endif


/*----------------------------------------------------------------------------
  Linker generated Symbols
 *----------------------------------------------------------------------------*/
extern uint32_t __etext;
extern uint32_t __data_start__;
extern uint32_t __data_end__;
extern uint32_t __copy_table_start__;
extern uint32_t __copy_table_end__;
extern uint32_t __zero_table_start__;
extern uint32_t __zero_table_end__;
extern uint32_t __bss_start__;
extern uint32_t __bss_end__;
extern uint32_t __StackTop;
extern uint32_t __StackLimit;
extern uint32_t __ecc_init_start__, __ecc_init_end__;

/*----------------------------------------------------------------------------
  External References
 *----------------------------------------------------------------------------*/
extern void _start     (void) __attribute__((noreturn)); /* PreeMain (C library entry point) */

/*----------------------------------------------------------------------------
  Hypervisor stack, User Initial Stack & Heap
 *----------------------------------------------------------------------------*/
//<h> Hypervisor Stack Configuration
//  <o> Hypervisor Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//</h>
#define  __HYP_STACK_SIZE  0x00000100
static uint8_t hyp_stack[__HYP_STACK_SIZE] __attribute__ ((aligned(8), used, section(".hyp_stack")));

//<h> Stack Configuration
//  <o> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//</h>
#define  __STACK_SIZE  0x0001D000
static uint8_t stack[__STACK_SIZE] __attribute__ ((aligned(8), used, section(".stack")));

//<h> Heap Configuration
//  <o>  Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
//</h>
#define  __HEAP_SIZE   0x00000C00
#if __HEAP_SIZE > 0
static uint8_t heap[__HEAP_SIZE]   __attribute__ ((aligned(8), used, section(".heap")));
#endif

/*----------------------------------------------------------------------------
  Exception / Interrupt Handler
 *----------------------------------------------------------------------------*/
void EL2_Reset_Handler    (void) __attribute__((naked,__noreturn__));
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
  Exception / Interrupt Vector Table
 *----------------------------------------------------------------------------*/
void __attribute__ ((naked, aligned(32), used, section(".vectors"))) __EL2_Vectors(void) {
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

void __attribute__ ((naked, aligned(32), used, section(".vectors"))) __EL1_Vectors(void) {
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

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void EL2_Default_Handler(void) {
  while(1);
}

void EL1_Default_Handler(void) {
  while(1);
}

/*----------------------------------------------------------------------------
  Switch from EL2 to EL1 mode
 *----------------------------------------------------------------------------*/
void __attribute__((naked,__noreturn__)) SwitchEL2toEL1(uint32_t func) {
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

  __builtin_unreachable(); /* No return here */
}

/*----------------------------------------------------------------------------
  Reset Handler called on controller reset
 *----------------------------------------------------------------------------*/
void __attribute__((naked,__noreturn__)) EL2_Reset_Handler(void) {
  register uint32_t *pSrc, *pDest;
  register uint64_t *pDest64;
  register uint32_t *pTable __attribute__((unused));

  /* ECC initialization:
   * .bss, heap and stack must be initialized before usage */
  for (pDest64 = (uint64_t*)&__ecc_init_start__; pDest64 < (uint64_t*)&__ecc_init_end__; pDest64++) {
    *pDest64 = 0xDEADBEEFFEEDCAFEUL;
  }
  __set_SP(__HypStackTop);

  SystemInit(); /* System Initialization */

/* Firstly it copies data from read only memory to RAM.
 * There are two schemes to copy. One can copy more than one sections.
 * Another can copy only one section. The former scheme needs more
 * instructions and read-only data to implement than the latter.
 * Macro __STARTUP_COPY_MULTIPLE is used to choose between two schemes.
 */

#ifdef __STARTUP_COPY_MULTIPLE
/* Multiple sections scheme.
 *
 * Between symbol address __copy_table_start__ and __copy_table_end__,
 * there are array of triplets, each of which specify:
 *   offset 0: LMA of start of a section to copy from
 *   offset 4: VMA of start of a section to copy to
 *   offset 8: size of the section to copy. Must be multiply of 4
 *
 * All addresses must be aligned to 4 bytes boundary.
 */
  pTable = &__copy_table_start__;

  for (; pTable < &__copy_table_end__; pTable = pTable + 3) {
    pSrc  = (uint32_t*)*(pTable + 0);
    pDest = (uint32_t*)*(pTable + 1);
    for (; pDest < (uint32_t*)(*(pTable + 1) + *(pTable + 2)) ; ) {
      *pDest++ = *pSrc++;
    }
  }
#else
/* Single section scheme.
 *
 * The ranges of copy from/to are specified by following symbols
 *   __etext: LMA of start of the section to copy from. Usually end of text
 *   __data_start__: VMA of start of the section to copy to
 *   __data_end__: VMA of end of the section to copy to
 *
 * All addresses must be aligned to 4 bytes boundary.
 */
  pSrc  = &__etext;
  pDest = &__data_start__;

  for ( ; pDest < &__data_end__ ; ) {
    *pDest++ = *pSrc++;
  }
#endif /*__STARTUP_COPY_MULTIPLE */

/* This part of work usually is done in C library startup code.
 * Otherwise, define this macro to enable it in this startup.
 *
 * There are two schemes too.
 * One can clear multiple BSS sections. Another can only clear one section.
 * The former is more size expensive than the latter.
 *
 * Define macro __STARTUP_CLEAR_BSS_MULTIPLE to choose the former.
 * Otherwise define macro __STARTUP_CLEAR_BSS to choose the later.
 */
#ifdef __STARTUP_CLEAR_BSS_MULTIPLE
/* Multiple sections scheme.
 *
 * Between symbol address __copy_table_start__ and __copy_table_end__,
 * there are array of tuples specifying:
 *   offset 0: Start of a BSS section
 *   offset 4: Size of this BSS section. Must be multiply of 4
 */
  pTable = &__zero_table_start__;

  for (; pTable < &__zero_table_end__; pTable = pTable + 2) {
    pDest = (uint32_t*)*(pTable + 0);
    for (; pDest < (uint32_t*)(*(pTable + 0) + *(pTable + 1)) ; ) {
      *pDest++ = 0;
    }
  }
#elif defined (__STARTUP_CLEAR_BSS)
/* Single BSS section scheme.
 *
 * The BSS section is specified by following symbols
 *   __bss_start__: start of the BSS section.
 *   __bss_end__: end of the BSS section.
 *
 * Both addresses must be aligned to 4 bytes boundary.
 */
  pDest = &__bss_start__;

  for ( ; pDest < &__bss_end__ ; ) {
    *pDest++ = 0UL;
  }
#endif /* __STARTUP_CLEAR_BSS_MULTIPLE || __STARTUP_CLEAR_BSS */

  /* ARMv8-R cores are in EL2 (hypervisor mode) after reset, and we need
     to first descend to EL1 (supervisor mode) before the traditional SP
     setting code can be run */
  SwitchEL2toEL1((uint32_t)_start); /* Enter PreeMain (C library entry point) */

  __builtin_unreachable(); /* No return here */
}
