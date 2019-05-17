/**************************************************************************//**
 * @file     system_ARMCR52.c
 * @brief    CMSIS Device System Source File for
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
  Linker generated Symbols
 *----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------
  Externals
 *----------------------------------------------------------------------------*/
extern uint32_t __EL1_Vectors, __EL2_Vectors;


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

  __set_VBAR(__EL1_Vectors);

  if(__get_mode() == CPSR_M_HYP)
    __set_HVBAR(__EL2_Vectors);

  SystemCoreClock = SYSTEM_CLOCK;

  /* I-cache & D-cache initialization */
    /* EL2 */
  ctrl = __get_HSCTRL();
  ctrl |= 0x1004; /* Set I and C bits */
  __set_HSCTRL(ctrl);

    /* EL1/0 */
  ctrl = __get_SCTRL();
  ctrl |= 0x1004; /* Set I and C bits */
  __set_SCTRL(ctrl);

}
