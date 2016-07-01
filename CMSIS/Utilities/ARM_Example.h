
/****************************************************************************************************//**
 * @file     ARM_Example.h
 *
 * @brief    CMSIS Cortex-M3 Peripheral Access Layer Header File for
 *           ARM_Example from ARM Ltd..
 *
 * @version  V1.2
 * @date     16. April 2014
 *
 * @note     Generated with SVDConv V2.81e 
 *           from CMSIS SVD File 'ARM_Example.svd' Version 1.2,
 *
 * @par      ARM Limited (ARM) is supplying this software for use with Cortex-M
 *           processor based microcontroller, but can be equally used for other
 *           suitable processor architectures. This file can be freely distributed.
 *           Modifications to this file shall be clearly marked.
 *           
 *           THIS SOFTWARE IS PROVIDED "AS IS". NO WARRANTIES, WHETHER EXPRESS, IMPLIED
 *           OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
 *           MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
 *           ARM SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR
 *           CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER. 
 *
 *******************************************************************************************************/



/** @addtogroup ARM Ltd.
  * @{
  */

/** @addtogroup ARM_Example
  * @{
  */

#ifndef ARM_EXAMPLE_H
#define ARM_EXAMPLE_H

#ifdef __cplusplus
extern "C" {
#endif


/* -------------------------  Interrupt Number Definition  ------------------------ */

typedef enum {
/* -------------------  Cortex-M3 Processor Exceptions Numbers  ------------------- */
  Reset_IRQn                    = -15,              /*!<   1  Reset Vector, invoked on Power up and warm reset                 */
  NonMaskableInt_IRQn           = -14,              /*!<   2  Non maskable Interrupt, cannot be stopped or preempted           */
  HardFault_IRQn                = -13,              /*!<   3  Hard Fault, all classes of Fault                                 */
  MemoryManagement_IRQn         = -12,              /*!<   4  Memory Management, MPU mismatch, including Access Violation
                                                         and No Match                                                          */
  BusFault_IRQn                 = -11,              /*!<   5  Bus Fault, Pre-Fetch-, Memory Access Fault, other address/memory
                                                         related Fault                                                         */
  UsageFault_IRQn               = -10,              /*!<   6  Usage Fault, i.e. Undef Instruction, Illegal State Transition    */
  SVCall_IRQn                   =  -5,              /*!<  11  System Service Call via SVC instruction                          */
  DebugMonitor_IRQn             =  -4,              /*!<  12  Debug Monitor                                                    */
  PendSV_IRQn                   =  -2,              /*!<  14  Pendable request for system service                              */
  SysTick_IRQn                  =  -1,              /*!<  15  System Tick Timer                                                */
/* -------------------  ARM_Example Specific Interrupt Numbers  ------------------- */
  TIMER0_IRQn                   =   0,              /*!<   0  TIMER0                                                           */
  TIMER1_IRQn                   =   4,              /*!<   4  TIMER1                                                           */
  TIMER2_IRQn                   =   6               /*!<   6  TIMER2                                                           */
} IRQn_Type;


/** @addtogroup Configuration_of_CMSIS
  * @{
  */


/* ================================================================================ */
/* ================      Processor and Core Peripheral Section     ================ */
/* ================================================================================ */

/* ----------------Configuration of the Cortex-M3 Processor and Core Peripherals---------------- */
#define __CM3_REV                 0x0100            /*!< Cortex-M3 Core Revision                                               */
#define __MPU_PRESENT                  1            /*!< MPU present or not                                                    */
#define __NVIC_PRIO_BITS               3            /*!< Number of Bits used for Priority Levels                               */
#define __Vendor_SysTickConfig         0            /*!< Set to 1 if different SysTick Config is used                          */
/** @} */ /* End of group Configuration_of_CMSIS */

#include "core_cm3.h"                               /*!< Cortex-M3 processor and core peripherals                              */
#include "system_ARMCM3.h"                          /*!< ARM_Example System                                                    */


/* ================================================================================ */
/* ================       Device Specific Peripheral Section       ================ */
/* ================================================================================ */


/** @addtogroup Device_Peripheral_Registers
  * @{
  */


/* -------------------  Start of section using anonymous unions  ------------------ */
#if defined(__CC_ARM)
  #pragma push
  #pragma anon_unions
#elif defined(__ICCARM__)
  #pragma language=extended
#elif defined(__GNUC__)
  /* anonymous unions are enabled by default */
#elif defined(__TMS470__)
/* anonymous unions are enabled by default */
#elif defined(__TASKING__)
  #pragma warning 586
#else
  #warning Not supported compiler type
#endif



/* ================================================================================ */
/* ================                     TIMER0                     ================ */
/* ================================================================================ */


/**
  * @brief 32 Timer / Counter, counting up or down from different sources (TIMER0)
  */

typedef struct {                                    /*!< TIMER0 Structure                                                      */
  __IO uint32_t  CR;                                /*!< Control Register                                                      */
  __IO uint16_t  SR;                                /*!< Status Register                                                       */
  __I  uint16_t  RESERVED0[5];
  __IO uint16_t  INT;                               /*!< Interrupt Register                                                    */
  __I  uint16_t  RESERVED1[7];
  __IO uint32_t  COUNT;                             /*!< The Counter Register reflects the actual Value of the Timer/Counter   */
  __IO uint32_t  MATCH;                             /*!< The Match Register stores the compare Value for the MATCH condition   */
  
  union {
    __O  uint32_t  PRESCALE_WR;                     /*!< The Prescale Register stores the Value for the prescaler. The
                                                         cont event gets divided by this value                                 */
    __I  uint32_t  PRESCALE_RD;                     /*!< The Prescale Register stores the Value for the prescaler. The
                                                         cont event gets divided by this value                                 */
  };
  __I  uint32_t  RESERVED2[9];
  __IO uint32_t  RELOAD[4];                         /*!< The Reload Register stores the Value the COUNT Register gets
                                                         reloaded on a when a condition was met.                               */
} TIMER0_Type;


/* --------------------  End of section using anonymous unions  ------------------- */
#if defined(__CC_ARM)
  #pragma pop
#elif defined(__ICCARM__)
  /* leave anonymous unions enabled */
#elif defined(__GNUC__)
  /* anonymous unions are enabled by default */
#elif defined(__TMS470__)
  /* anonymous unions are enabled by default */
#elif defined(__TASKING__)
  #pragma warning restore
#else
  #warning Not supported compiler type
#endif




/* ================================================================================ */
/* ================              Peripheral memory map             ================ */
/* ================================================================================ */

#define TIMER0_BASE                     0x40010000UL
#define TIMER1_BASE                     0x40010100UL
#define TIMER2_BASE                     0x40010200UL


/* ================================================================================ */
/* ================             Peripheral declaration             ================ */
/* ================================================================================ */

#define TIMER0                          ((TIMER0_Type             *) TIMER0_BASE)
#define TIMER1                          ((TIMER0_Type             *) TIMER1_BASE)
#define TIMER2                          ((TIMER0_Type             *) TIMER2_BASE)


/** @} */ /* End of group Device_Peripheral_Registers */
/** @} */ /* End of group ARM_Example */
/** @} */ /* End of group ARM Ltd. */

#ifdef __cplusplus
}
#endif


#endif  /* ARM_Example_H */

