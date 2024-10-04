/**************************************************************************//**
 * @file     core_cr52.h
 * @brief    CMSIS Cortex-R52 Core Peripheral Access Layer Header File
 * @version  V1.0.0
 * @date     15. June 2020
 ******************************************************************************/
/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
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

#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header   /* treat file as system include file */
#elif defined ( __GNUC__ )
  #pragma GCC diagnostic ignored "-Wpedantic"   /* disable pedantic warning due to unnamed structs/unions */
#endif

#ifndef __CORE_CR52_H_GENERIC
#define __CORE_CR52_H_GENERIC

#include <stdint.h>

#ifdef __cplusplus
 extern "C" {
#endif

/**
  \page CMSIS_MISRA_Exceptions  MISRA-C:2004 Compliance Exceptions
  CMSIS violates the following MISRA-C:2004 rules:

   \li Required Rule 8.5, object/function definition in header file.<br>
     Function definitions in header files are used to allow 'inlining'.

   \li Required Rule 18.4, declaration of union type or object of union type: '{...}'.<br>
     Unions are used for effective representation of core registers.

   \li Advisory Rule 19.7, Function-like macro defined.<br>
     Function-like macros are used to allow more efficient code.
 */


/*******************************************************************************
 *                 CMSIS definitions
 ******************************************************************************/
/**
  \ingroup Cortex_R52
  @{
 */

#include "cmsis_version.h"

/*  CMSIS CR52 definitions */
#define __CR52_CMSIS_VERSION_MAIN  (1U)                                      /*!< \deprecated [31:16] CMSIS HAL main version   */
#define __CR52_CMSIS_VERSION_SUB   (0U)                                      /*!< \deprecated [15:0]  CMSIS HAL sub version    */
#define __CR52_CMSIS_VERSION       ((__CR52_CMSIS_VERSION_MAIN << 16U) | \
                                   __CR52_CMSIS_VERSION_SUB          )       /*!< \deprecated CMSIS HAL version number         */

#define __CORTEX_R                 (52U)                                       /*!< Cortex-R Core */

/** __FPU_USED indicates whether an FPU is used or not.
    For this, __FPU_PRESENT has to be checked prior to making use of FPU specific registers and functions.
*/
#if defined ( __CC_ARM )
  #if defined (__TARGET_FPU_VFP)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1U)
    #if defined (__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  #if defined (__ARM_FP)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #warning "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1U)
    #if defined (__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined ( __GNUC__ )
  #if defined (__VFP_FP__) && !defined(__SOFTFP__)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if (defined(__ARM_NEON) && (__ARM_NEON == 1))
    #if defined (__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined ( __ICCARM__ )
  #if defined (__ARMVFP__)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

  #if defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1U)
    #if defined (__DSP_PRESENT) && (__DSP_PRESENT == 1U)
      #define __DSP_USED       1U
    #else
      #error "Compiler generates DSP (SIMD) instructions for a devices without DSP extensions (check __DSP_PRESENT)"
      #define __DSP_USED         0U
    #endif
  #else
    #define __DSP_USED         0U
  #endif

#elif defined ( __TI_ARM__ )
  #if defined (__TI_VFP_SUPPORT__)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

#elif defined ( __TASKING__ )
  #if defined (__FPU_VFP__)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

#elif defined ( __CSMC__ )
  #if ( __CSMC__ & 0x400U)
    #if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)
      #define __FPU_USED       1U
    #else
      #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
      #define __FPU_USED       0U
    #endif
  #else
    #define __FPU_USED         0U
  #endif

#endif

#include "cmsis_compiler.h"               /* CMSIS compiler specific defines */


#ifdef __cplusplus
}
#endif

#endif /* __CORE_CR52_H_GENERIC */

#ifndef __CMSIS_GENERIC

#ifndef __CORE_CR52_H_DEPENDANT
#define __CORE_CR52_H_DEPENDANT

#ifdef __cplusplus
 extern "C" {
#endif

 /* check device defines and use defaults */
#if defined __CHECK_DEVICE_DEFINES
  #ifndef __CR52_REV
    #define __CR52_REV              0x0000U
    #warning "__CR52_REV not defined in device header file; using default!"
  #endif

  #ifndef __FPU_PRESENT
    #define __FPU_PRESENT             0U
    #warning "__FPU_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __MPU_PRESENT
    #define __MPU_PRESENT             0U
    #warning "__MPU_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __ICACHE_PRESENT
    #define __ICACHE_PRESENT          0U
    #warning "__ICACHE_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __DCACHE_PRESENT
    #define __DCACHE_PRESENT          0U
    #warning "__DCACHE_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __DTCM_PRESENT
    #define __DTCM_PRESENT            0U
    #warning "__DTCM_PRESENT        not defined in device header file; using default!"
  #endif

  #ifndef __ECC_PRESENT
    #define __ECC_PRESENT             0U
    #warning "__ECC_PRESENT not defined in device header file; using default!"
  #endif

#endif

/* IO definitions (access restrictions to peripheral registers) */
/**
    \defgroup CMSIS_glob_defs CMSIS Global Defines

    <strong>IO Type Qualifiers</strong> are used
    \li to specify the access to peripheral variables.
    \li for automatic generation of peripheral register debug information.
*/
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

/* following defines should be used for structure members */
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */
#define RESERVED(N, T) T RESERVED##N;    // placeholder struct members used for "reserved" areas

/*@} end of group Cortex_R52 */



 /*******************************************************************************
  *                 Register Abstraction
   Core Register contain:
   - CPSR
  ******************************************************************************/

/* Core Register CPSR */
typedef union
{
  struct
  {
    uint32_t M:5;                        /*!< \brief bit:  0.. 4  Mode field */
    uint32_t T:1;                        /*!< \brief bit:      5  Thumb execution state bit */
    uint32_t F:1;                        /*!< \brief bit:      6  FIQ mask bit */
    uint32_t I:1;                        /*!< \brief bit:      7  IRQ mask bit */
    uint32_t A:1;                        /*!< \brief bit:      8  Asynchronous abort mask bit */
    uint32_t E:1;                        /*!< \brief bit:      9  Endianness execution state bit */
    uint32_t IT1:6;                      /*!< \brief bit: 10..15  If-Then execution state bits 2-7 */
    uint32_t GE:4;                       /*!< \brief bit: 16..19  Greater than or Equal flags */
    RESERVED(0:4, uint32_t)              
    uint32_t J:1;                        /*!< \brief bit:     24  Jazelle bit */
    uint32_t IT0:2;                      /*!< \brief bit: 25..26  If-Then execution state bits 0-1 */
    uint32_t Q:1;                        /*!< \brief bit:     27  Saturation condition flag */
    uint32_t V:1;                        /*!< \brief bit:     28  Overflow condition code flag */
    uint32_t C:1;                        /*!< \brief bit:     29  Carry condition code flag */
    uint32_t Z:1;                        /*!< \brief bit:     30  Zero condition code flag */
    uint32_t N:1;                        /*!< \brief bit:     31  Negative condition code flag */
  } b;                                   /*!< \brief Structure used for bit  access */
  uint32_t w;                            /*!< \brief Type      used for word access */
} CPSR_Type;



/* CPSR Register Definitions */
#define CPSR_N_Pos                       31U                                    /*!< \brief CPSR: N Position */
#define CPSR_N_Msk                       (1UL << CPSR_N_Pos)                    /*!< \brief CPSR: N Mask */

#define CPSR_Z_Pos                       30U                                    /*!< \brief CPSR: Z Position */
#define CPSR_Z_Msk                       (1UL << CPSR_Z_Pos)                    /*!< \brief CPSR: Z Mask */

#define CPSR_C_Pos                       29U                                    /*!< \brief CPSR: C Position */
#define CPSR_C_Msk                       (1UL << CPSR_C_Pos)                    /*!< \brief CPSR: C Mask */

#define CPSR_V_Pos                       28U                                    /*!< \brief CPSR: V Position */
#define CPSR_V_Msk                       (1UL << CPSR_V_Pos)                    /*!< \brief CPSR: V Mask */

#define CPSR_Q_Pos                       27U                                    /*!< \brief CPSR: Q Position */
#define CPSR_Q_Msk                       (1UL << CPSR_Q_Pos)                    /*!< \brief CPSR: Q Mask */

#define CPSR_IT0_Pos                     25U                                    /*!< \brief CPSR: IT0 Position */
#define CPSR_IT0_Msk                     (3UL << CPSR_IT0_Pos)                  /*!< \brief CPSR: IT0 Mask */

#define CPSR_J_Pos                       24U                                    /*!< \brief CPSR: J Position */
#define CPSR_J_Msk                       (1UL << CPSR_J_Pos)                    /*!< \brief CPSR: J Mask */

#define CPSR_GE_Pos                      16U                                    /*!< \brief CPSR: GE Position */
#define CPSR_GE_Msk                      (0xFUL << CPSR_GE_Pos)                 /*!< \brief CPSR: GE Mask */

#define CPSR_IT1_Pos                     10U                                    /*!< \brief CPSR: IT1 Position */
#define CPSR_IT1_Msk                     (0x3FUL << CPSR_IT1_Pos)               /*!< \brief CPSR: IT1 Mask */

#define CPSR_E_Pos                       9U                                     /*!< \brief CPSR: E Position */
#define CPSR_E_Msk                       (1UL << CPSR_E_Pos)                    /*!< \brief CPSR: E Mask */

#define CPSR_A_Pos                       8U                                     /*!< \brief CPSR: A Position */
#define CPSR_A_Msk                       (1UL << CPSR_A_Pos)                    /*!< \brief CPSR: A Mask */

#define CPSR_I_Pos                       7U                                     /*!< \brief CPSR: I Position */
#define CPSR_I_Msk                       (1UL << CPSR_I_Pos)                    /*!< \brief CPSR: I Mask */

#define CPSR_F_Pos                       6U                                     /*!< \brief CPSR: F Position */
#define CPSR_F_Msk                       (1UL << CPSR_F_Pos)                    /*!< \brief CPSR: F Mask */

#define CPSR_T_Pos                       5U                                     /*!< \brief CPSR: T Position */
#define CPSR_T_Msk                       (1UL << CPSR_T_Pos)                    /*!< \brief CPSR: T Mask */

#define CPSR_M_Pos                       0U                                     /*!< \brief CPSR: M Position */
#define CPSR_M_Msk                       (0x1FUL << CPSR_M_Pos)                 /*!< \brief CPSR: M Mask */

#define CPSR_M_USR                       0x10U                                  /*!< \brief CPSR: M User mode (PL0) */
#define CPSR_M_FIQ                       0x11U                                  /*!< \brief CPSR: M Fast Interrupt mode (PL1) */
#define CPSR_M_IRQ                       0x12U                                  /*!< \brief CPSR: M Interrupt mode (PL1) */
#define CPSR_M_SVC                       0x13U                                  /*!< \brief CPSR: M Supervisor mode (PL1) */
#define CPSR_M_MON                       0x16U                                  /*!< \brief CPSR: M Monitor mode (PL1) */
#define CPSR_M_ABT                       0x17U                                  /*!< \brief CPSR: M Abort mode (PL1) */
#define CPSR_M_HYP                       0x1AU                                  /*!< \brief CPSR: M Hypervisor mode (PL2) */
#define CPSR_M_UND                       0x1BU                                  /*!< \brief CPSR: M Undefined mode (PL1) */
#define CPSR_M_SYS                       0x1FU                                  /*!< \brief CPSR: M System mode (PL1) */


#ifdef __cplusplus
}
#endif

#endif /* __CORE_CR52_H_DEPENDANT */

#endif /* __CMSIS_GENERIC */
