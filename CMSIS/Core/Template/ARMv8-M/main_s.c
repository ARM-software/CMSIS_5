/*
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ----------------------------------------------------------------------
 *
 * main_s.c      Code template for secure main function 
 *
 * Version 1.0
 *    Initial Release
 *---------------------------------------------------------------------------*/
 
/* Use CMSE intrinsics */
#include <arm_cmse.h>
 
#include "RTE_Components.h"
#include CMSIS_device_header
 
/* TZ_START_NS: Start address of non-secure application */
#ifndef TZ_START_NS
#define TZ_START_NS (0x200000U)
#endif
 
/* Default process stack size */
#ifndef PROCESS_STACK_SIZE
#define PROCESS_STACK_SIZE 256U
#endif
 
/* Default process stack */
static uint64_t ProcessStack[PROCESS_STACK_SIZE/8U];
 
/* Generate BLXNS instruction */
void NonSecure_Start (uint32_t addr) __attribute__((always_inline));
void NonSecure_Start (uint32_t addr) {
  __ASM volatile ("blxns %[addr]" : : [addr] "l" (addr));
}
 
 
/* Secure main() */
int main(void) {
  volatile uint32_t NonSecure_ResetHandler;
 
  /* Add user setup code for secure part here*/
 
  /* Set non-secure main stack (MSP_NS) */
  __TZ_set_MSP_NS(*((uint32_t *)(TZ_START_NS)));

  /* Set default PSP, PSPLIM and privileged Thread Mode using PSP */
  __set_PSPLIM((uint32_t)ProcessStack);
  __set_PSP   ((uint32_t)ProcessStack + PROCESS_STACK_SIZE);
  __set_CONTROL(0x02U);
 
  /* Get non-secure reset hanlder */
  NonSecure_ResetHandler = cmse_nsfptr_create(*((uint32_t *)((TZ_START_NS) + 4U)));
 
  /* Start non-secure state software application */
  NonSecure_Start(NonSecure_ResetHandler);
 
  /* Non-secure software does not return, this code is not executed */
  while (1) {
    __NOP();
  }
}
