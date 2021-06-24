/******************************************************************************
 * @file     ARMCA32.h
 * @brief    CMSIS Cortex-A32 Core Peripheral Access Layer Header File 
 * @version  V1.1.0
 * @date     15. May 2019
 *
 * @note
 *
 ******************************************************************************/
/*
 * Copyright (c) 2009-2019 Arm Limited. All rights reserved.
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


None of above values have been checked !

*/


#ifndef __ARMCA32_H__
#define __ARMCA32_H__

#ifdef __cplusplus
extern "C" {
#endif


/******************************************************************************/
/*                         Peripheral memory map                              */
/******************************************************************************/

/* Peripheral and RAM base address */
#define VE_A32_NORMAL                  (0x00000000UL)                                /*!< (FLASH0     ) Base Address */
#define VE_A32_PERIPH                  (0x13000000UL)                                /*!< (FLASH0     ) Base Address */
#define VE_A32_NORMAL2                 (0x14000000UL) 

/* --------  Configuration of the Cortex-A32 Processor and Core Peripherals  ------- */
#define __CA_REV        0x0000U    /*!< Core revision r0p0                          */
#define __CORTEX_A           32U    /*!< Cortex-A32 Core                              */
#define __FPU_PRESENT        1U    /* FPU present                                   */
#define __GIC_PRESENT        1U    /* GIC present                                   */
#define __TIM_PRESENT        1U    /* TIM present                                   */
#define __L2C_PRESENT        0U    /* L2C present                                   */

/** Device specific Interrupt IDs */
typedef enum IRQn
{
/******  SGI Interrupts Numbers                 ****************************************/
  SGI0_IRQn            =  0,        /*!< Software Generated Interrupt 0 */
  SGI1_IRQn            =  1,        /*!< Software Generated Interrupt 1 */
  SGI2_IRQn            =  2,        /*!< Software Generated Interrupt 2 */
  SGI3_IRQn            =  3,        /*!< Software Generated Interrupt 3 */
  SGI4_IRQn            =  4,        /*!< Software Generated Interrupt 4 */
  SGI5_IRQn            =  5,        /*!< Software Generated Interrupt 5 */
  SGI6_IRQn            =  6,        /*!< Software Generated Interrupt 6 */
  SGI7_IRQn            =  7,        /*!< Software Generated Interrupt 7 */
  SGI8_IRQn            =  8,        /*!< Software Generated Interrupt 8 */
  SGI9_IRQn            =  9,        /*!< Software Generated Interrupt 9 */
  SGI10_IRQn           = 10,        /*!< Software Generated Interrupt 10 */
  SGI11_IRQn           = 11,        /*!< Software Generated Interrupt 11 */
  SGI12_IRQn           = 12,        /*!< Software Generated Interrupt 12 */
  SGI13_IRQn           = 13,        /*!< Software Generated Interrupt 13 */
  SGI14_IRQn           = 14,        /*!< Software Generated Interrupt 14 */
  SGI15_IRQn           = 15,        /*!< Software Generated Interrupt 15 */

/******  Cortex-A9 Processor Exceptions Numbers ****************************************/
  GlobalTimer_IRQn     = 27,        /*!< Global Timer Interrupt                        */
  PrivTimer_IRQn       = 29,        /*!< Private Timer Interrupt                       */
  PrivWatchdog_IRQn    = 30,        /*!< Private Watchdog Interrupt                    */

/******  Platform Exceptions Numbers ***************************************************/
  Watchdog_IRQn        = 32,        /*!< SP805 Interrupt        */
  Timer0_IRQn          = 34,        /*!< SP804 Interrupt        */
  Timer1_IRQn          = 35,        /*!< SP804 Interrupt        */
  RTClock_IRQn         = 36,        /*!< PL031 Interrupt        */
  UART0_IRQn           = 37,        /*!< PL011 Interrupt        */
  UART1_IRQn           = 38,        /*!< PL011 Interrupt        */
  UART2_IRQn           = 39,        /*!< PL011 Interrupt        */
  UART3_IRQn           = 40,        /*!< PL011 Interrupt        */
  MCI0_IRQn            = 41,        /*!< PL180 Interrupt (1st)  */
  MCI1_IRQn            = 42,        /*!< PL180 Interrupt (2nd)  */
  AACI_IRQn            = 43,        /*!< PL041 Interrupt        */
  Keyboard_IRQn        = 44,        /*!< PL050 Interrupt        */
  Mouse_IRQn           = 45,        /*!< PL050 Interrupt        */
  CLCD_IRQn            = 46,        /*!< PL111 Interrupt        */
  Ethernet_IRQn        = 47,        /*!< SMSC_91C111 Interrupt  */
  VFS2_IRQn            = 73,        /*!< VFS2 Interrupt         */
} IRQn_Type;

// To allow inclusion of core_ca.h but those symbols are not used in our code
#define GIC_DISTRIBUTOR_BASE                  0
#define GIC_INTERFACE_BASE                    0
#define TIMER_BASE                            0

#include "core_ca.h"
#include <system_ARMCA32.h>

#ifdef __cplusplus
}
#endif

#endif  // __ARMCA5_H__
