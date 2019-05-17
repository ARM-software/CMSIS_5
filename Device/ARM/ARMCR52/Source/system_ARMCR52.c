/**************************************************************************//**
 * @file     system_ARMCR52.c
 * @brief    CMSIS Device System Source File for
 *           ARMCR52 Device
 * @version  V1.0.0
 * @date     21. May 2019
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

#if defined (ARMCR52)
  #include "ARMCR52.h"
#else
  #error device not specified!
#endif

/*----------------------------------------------------------------------------
  Define clocks
 *----------------------------------------------------------------------------*/
#define  XTAL            (50000000UL)     /* Oscillator frequency */

#define  SYSTEM_CLOCK    (XTAL / 2U)


/*----------------------------------------------------------------------------
  External References
 *----------------------------------------------------------------------------*/
extern uint32_t __VECTOR_TABLE_EL2;
extern uint32_t __VECTOR_TABLE_EL1;


/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
uint32_t SystemCoreClock = SYSTEM_CLOCK;  /* System Core Clock Frequency */


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
  uint32_t ctrl;

  __set_HVBAR((uint32_t)&__VECTOR_TABLE_EL2);
  __set_VBAR((uint32_t)&__VECTOR_TABLE_EL1);

  /* I-cache & D-cache initialization */
    /* EL2 */
  ctrl = __get_HSCTLR();
  ctrl |= 0x1004; /* Set I and C bits */
  __set_HSCTLR(ctrl);

    /* EL1/0 */
  ctrl = __get_SCTLR();
  ctrl |= 0x1004; /* Set I and C bits */
  __set_SCTLR(ctrl);

  SystemCoreClock = SYSTEM_CLOCK;
}
