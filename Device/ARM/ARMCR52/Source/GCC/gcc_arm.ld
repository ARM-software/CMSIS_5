/******************************************************************************
 * @file     gcc_arm.ld
 * @brief    GNU Linker Script for Cortex-R based device
 * @version  V1.0.1
 * @date     08. Aug 2019
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
 *-------- <<< Use Configuration Wizard in Context Menu >>> -------------------
 */

/*---------------------- Flash Configuration ----------------------------------
  <h> Flash Configuration
    <o0> Flash Base Address <0x0-0xFFFFFFFF:8>
    <o1> Flash Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
  -----------------------------------------------------------------------------*/
__ROM_BASE = 0x10000000;
__ROM_SIZE = 0x00080000;

/*--------------------- Embedded RAM Configuration ----------------------------
  <h> RAM Configuration
    <o0> RAM Base Address    <0x0-0xFFFFFFFF:8>
    <o1> RAM Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
 -----------------------------------------------------------------------------*/
__RAM_BASE = 0x40000000;
__RAM_SIZE = 0x00080000;

/*--------------------- Stack / Heap Configuration ----------------------------
  <h> Stack / Heap Configuration
    <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
    <o1> Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
  -----------------------------------------------------------------------------*/
__STACK_SIZE = 0x0001D000; /* Newlib allocates stack for each CPSR mode */
__HEAP_SIZE  = 0x00000C00;

/*--------------------- Hypervisor Stack Configuration -----------------------
  <h> Hypervisor Stack Configuration
    <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
  -----------------------------------------------------------------------------*/
__HYP_STACK_SIZE = 0x00000100;


/*
 *-------------------- <<< end of configuration section >>> -------------------
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = __ROM_BASE, LENGTH = __ROM_SIZE
  RAM   (rwx) : ORIGIN = __RAM_BASE, LENGTH = __RAM_SIZE
}

/* Linker script to place sections and symbol values. Should be used together
 * with other linker script that defines memory regions FLASH and RAM.
 * It references following symbols, which must be defined in code:
 *   Reset_Handler : Entry of reset handler
 *
 * It defines following symbols, which code can use without definition:
 *   __exidx_start
 *   __exidx_end
 *   __ecc_table_start__
 *   __ecc_table_end__
 *   __copy_table_start__
 *   __copy_table_end__
 *   __zero_table_start__
 *   __zero_table_end__
 *   __etext
 *   __data_start__
 *   __preinit_array_start
 *   __preinit_array_end
 *   __init_array_start
 *   __init_array_end
 *   __fini_array_start
 *   __fini_array_end
 *   __data_end__
 *   __bss_start__
 *   __bss_end__
 *   __end__
 *   end
 *   __HeapLimit
 *   __StackLimit
 *   __StackTop
 *   __stack
 *   __HypStackLimit
 *   __HypStackTop
 */
ENTRY(__EL2_Vectors)

SECTIONS
{
  .text :
  {
    KEEP(*(.vectors))
    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)

    KEEP(*(.eh_frame*))
  } > FLASH

  /*
   * SG veneers:
   * All SG veneers are placed in the special output section .gnu.sgstubs. Its start address
   * must be set, either with the command line option ‘--section-start’ or in a linker script,
   * to indicate where to place these veneers in memory.
   */
/*
  .gnu.sgstubs :
  {
    . = ALIGN(32);
  } > FLASH
*/

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > FLASH

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > FLASH
  __exidx_end = .;

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (__etext)
    LONG (__data_start__)
    LONG ((__data_end__ - __data_start__) / 4)
    /* Add each additional data section here */
/*
    LONG (__etext2)
    LONG (__data2_start__)
    LONG ((__data2_end__ - __data2_start__) / 4)
*/
    __copy_table_end__ = .;
  } > FLASH

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    /* Add each additional bss section here */
/*
    LONG (__bss2_start__)
    LONG ((__bss2_end__ - __bss2_start__) / 4)
*/
    __zero_table_end__ = .;
  } > FLASH

  /* ECC initialization is done by 64-bit writes thus used symbols
   * must be 8byte aligned */
  .ecc.table :
  {
    . = ALIGN(8);
    __ecc_table_start__ = .;

    QUAD (__data_start__)
    QUAD ((__data_end__ - __data_start__) / 8)

/*
    QUAD (__data2_start__)
    QUAD ((__data2_end__ - __data2_start__) / 8)
*/

    QUAD (__bss_start__)
    QUAD ((__bss_end__ - __bss_start__) / 8)

/*
    QUAD (__bss2_start__)
    QUAD ((__bss2_end__ - __bss2_start__) / 8)
*/

    QUAD (__HeapTop)
    QUAD ((__HeapLimit - __HeapTop) / 8)

    QUAD (__HypStackLimit)
    QUAD ((__HypStackLimit - __HypStackTop) / 8)

    QUAD (__StackLimit)
    QUAD ((__StackLimit - __StackTop) / 8)

    __ecc_table_end__ = .;
  } > FLASH

  /**
   * Location counter can end up 2byte aligned with narrow Thumb code but
   * __etext is assumed by startup code to be the LMA of a section in RAM
   * which must be 4byte aligned 
   */
  __etext = ALIGN (4);

  .data : AT (__etext)
  {
    . = ALIGN(8);
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    /* preinit data */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    /* init data */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);


    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(8);
    /* All data end */
    __data_end__ = .;

  } > RAM

  /*
   * Secondary data section, optional
   *
   * Remember to add each additional data section
   * to the .copy.table and .ecc.table above to asure proper
   * initialization during startup.
   */
/*
  __etext2 = ALIGN (4);

  .data2 : AT (__etext2)
  {
    . = ALIGN(8);
    __data2_start__ = .;
    *(.data2)
    *(.data2.*)
    . = ALIGN(8);
    __data2_end__ = .;

  } > RAM2
*/

  /* ECC initialization is done by 64-bit writes thus __bss_start__
   * and __bss_end__ must be 8byte aligned */
  .bss :
  {
    . = ALIGN(8);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end__ = .;
  } > RAM AT > RAM

  /*
   * Secondary bss section, optional
   *
   * Remember to add each additional bss section
   * to the .zero.table and .ecc.table above to asure proper
   * initialization during startup.
   */
/*
  .bss2 :
  {
    . = ALIGN(8);
    __bss2_start__ = .;
    *(.bss2)
    *(.bss2.*)
    . = ALIGN(8);
    __bss2_end__ = .;
  } > RAM2 AT > RAM2
*/

  .heap (COPY) :
  {
    . = ALIGN(8);
    __end__ = .;
    __HeapTop = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > RAM

  .stack (ORIGIN(RAM) + LENGTH(RAM) - __STACK_SIZE - __HYP_STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > RAM

  /* Set Hypevisor stack top to end of RAM. Then set regular stack top,
   * and stack limit move down by size of __STACK_SIZE section */
  .hyp_stack (ORIGIN(RAM) + LENGTH(RAM) - __HYP_STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __HypStackLimit = .;
    . = . + __HYP_STACK_SIZE;
    . = ALIGN(8);
    __HypStackTop = .;
  } > RAM

  PROVIDE(__stack = __StackTop);

  /* Check if data + heap + stack + hypstack exceeds RAM limit */
  ASSERT(__StackLimit >= __HeapLimit, "region RAM overflowed with stack")
}
