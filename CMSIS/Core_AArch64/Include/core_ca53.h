/**************************************************************************//**
 * @file     core_ca53.h
 * @brief    CMSIS Cortex-A53 Core Peripheral Access Layer Header File
 * @version  V1.0.1
 * @date     20. August 2020
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

#ifndef __CORE_CA53_H_GENERIC
#define __CORE_CA53_H_GENERIC

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
  \ingroup Cortex_A53
  @{
 */

#include "cmsis_version.h"

/*  CMSIS CA53 definitions */
#define __CA53_CMSIS_VERSION_MAIN  (1U)                                      /*!< \brief [31:16] CMSIS-Core(A) main version   */
#define __CA53_CMSIS_VERSION_SUB   (0U)                                      /*!< \brief [15:0]  CMSIS-Core(A) sub version    */
#define __CA53_CMSIS_VERSION       ((__CA53_CMSIS_VERSION_MAIN << 16U) | \
                                   __CA53_CMSIS_VERSION_SUB          )       /*!< \brief CMSIS-Core(A) version number         */

#define __CORTEX_A                 (53U)                                       /*!< Cortex-A Core */

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
  #if defined (__ARM_FP) && (__ARM_FP==0xE)
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

#endif /* __CORE_CA53_H_GENERIC */

#ifndef __CMSIS_GENERIC

#ifndef __CORE_CA53_H_DEPENDANT
#define __CORE_CA53_H_DEPENDANT

#ifdef __cplusplus
 extern "C" {
#endif

 /* check device defines and use defaults */
#if defined __CHECK_DEVICE_DEFINES
  #ifndef __CA53_REV
    #define __CA53_REV              0x0000U
    #warning "__CA53_REV not defined in device header file; using default!"
  #endif
  
  #ifndef __FPU_PRESENT
    #define __FPU_PRESENT             0U
    #warning "__FPU_PRESENT not defined in device header file; using default!"
  #endif
    
  #ifndef __GIC_PRESENT
    #define __GIC_PRESENT             1U
    #warning "__GIC_PRESENT not defined in device header file; using default!"
  #endif
  
  #ifndef __TIM_PRESENT
    #define __TIM_PRESENT             1U
    #warning "__TIM_PRESENT not defined in device header file; using default!"
  #endif
  
  #ifndef __L2C_PRESENT
    #define __L2C_PRESENT             0U
    #warning "__L2C_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __L3C_PRESENT
    #define __L3C_PRESENT             0U
    #warning "__L3C_PRESENT not defined in device header file; using default!"
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

/*@} end of group Cortex_A53 */



 /*******************************************************************************
  *                 Register Abstraction
   Core Register contain:
   - CPSR
   - CP15 Registers
   - L2C-310 Cache Controller
   - Generic Interrupt Controller Distributor
   - Generic Interrupt Controller Interface
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
    RESERVED(0:6, uint32_t)              
    uint32_t GE:4;                       /*!< \brief bit: 16..19  Greater than or Equal flags */
    RESERVED(1:1, uint32_t)              
    uint32_t DIT:1;                      /*!< \brief bit:     21  Data Independent Timing */
    uint32_t PAN:1;                      /*!< \brief bit:     22  Privileged Access Never */
    RESERVED(2:4, uint32_t)              
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

#define CPSR_PAN_Pos                     22U                                    /*!< \brief CPSR: PAN Position */
#define CPSR_PAN_Msk                     (0x1UL << CPSR_PAN_Pos)                /*!< \brief CPSR: PAN Mask */

#define CPSR_DIT_Pos                     21U                                    /*!< \brief CPSR: DIT Position */
#define CPSR_DIT_Msk                     (0x1UL << CPSR_PAN_Pos)                /*!< \brief CPSR: DIT Mask */

#define CPSR_GE_Pos                      16U                                    /*!< \brief CPSR: GE Position */
#define CPSR_GE_Msk                      (0xFUL << CPSR_GE_Pos)                 /*!< \brief CPSR: GE Mask */

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

/* Register SCTLR */
typedef union
{
  struct
  {
    uint64_t M:1;                        /*!< \brief bit:     0  MMU enable */
    uint64_t A:1;                        /*!< \brief bit:     1  Alignment check enable */
    uint64_t C:1;                        /*!< \brief bit:     2  Cache enable */
    uint64_t SA:1;                       /*!< \brief bit:     3  SP Alignment check enable */
    RESERVED(1:2, uint64_t)              //[5:4]
    uint64_t nAA:1;                      /*!< \brief bit:     6  Non-aligned access */
    RESERVED(2:4, uint64_t)              //[10:7]
    uint64_t EOS:1;                      /*!< \brief bit:    11  Exception Exit is Context Synchronizing */
    uint64_t I:1;                        /*!< \brief bit:    12  Instruction cache enable */
    uint64_t EnDB:1;                     //13
    RESERVED(3:2, uint64_t)              //[15:14]
    RESERVED(4:1, uint64_t)              //[16]
    RESERVED(5:1, uint64_t)              //[17]
    RESERVED(6:1, uint64_t)              //[18]
    uint64_t WXN:1;                      /*!< \brief bit:    19  Write permission implies XN */
    RESERVED(7:1, uint64_t)              //[20]
    uint64_t IESB:1;                     //21
    uint64_t EIS:1;                      //22
    RESERVED(8:1, uint64_t)              //[23]
    RESERVED(9:1, uint64_t)              //[24]
    uint64_t EE:1;                       /*!< \brief bit:    25  Exception Endianness */
    RESERVED(10:1, uint64_t)             //[26]
    uint64_t EnDA:1;                     //27
    RESERVED(11:2, uint64_t)             //[29:28]
    uint64_t EnIB:1;                     //30
    uint64_t EnIA:1;                     //31
    RESERVED(12:4, uint64_t)             //[35:32]
    uint64_t BT:1;                       //36
    uint64_t ITFSB:1;                    //37
    RESERVED(13:2, uint64_t)             //[39:38]
    uint64_t TCF:2;                      //[41:40]
    RESERVED(14:1, uint64_t)             //[42]
    uint64_t ATA:1;                      //43
    uint64_t DSSBS:1;                    //44
    RESERVED(15:19, uint64_t)            //[63:45]
  } b;                                   /*!< \brief Structure used for bit  access */
  uint64_t w;                            /*!< \brief Type      used for word access */
} SCTLR_Type;

#define SCTLR_TE_Pos                     30U                                    /*!< \brief SCTLR: TE Position */
#define SCTLR_TE_Msk                     (1UL << SCTLR_TE_Pos)                  /*!< \brief SCTLR: TE Mask */

#define SCTLR_AFE_Pos                    29U                                    /*!< \brief SCTLR: AFE Position */
#define SCTLR_AFE_Msk                    (1UL << SCTLR_AFE_Pos)                 /*!< \brief SCTLR: AFE Mask */

#define SCTLR_TRE_Pos                    28U                                    /*!< \brief SCTLR: TRE Position */
#define SCTLR_TRE_Msk                    (1UL << SCTLR_TRE_Pos)                 /*!< \brief SCTLR: TRE Mask */

#define SCTLR_EE_Pos                     25U                                    /*!< \brief SCTLR: EE Position */
#define SCTLR_EE_Msk                     (1UL << SCTLR_EE_Pos)                  /*!< \brief SCTLR: EE Mask */

#define SCTLR_UWXN_Pos                   20U                                    /*!< \brief SCTLR: UWXN Position */
#define SCTLR_UWXN_Msk                   (1UL << SCTLR_UWXN_Pos)                /*!< \brief SCTLR: UWXN Mask */

#define SCTLR_WXN_Pos                    19U                                    /*!< \brief SCTLR: WXN Position */
#define SCTLR_WXN_Msk                    (1UL << SCTLR_WXN_Pos)                 /*!< \brief SCTLR: WXN Mask */

#define SCTLR_nTWE_Pos                   18U                                    /*!< \brief SCTLR: nTWE Position */
#define SCTLR_nTWE_Msk                   (1UL << SCTLR_nTWE_Pos)                /*!< \brief SCTLR: nTWE Mask */

#define SCTLR_nTWI_Pos                   16U                                    /*!< \brief SCTLR: nTWI Position */
#define SCTLR_nTWI_Msk                   (1UL << SCTLR_nTWI_Pos)                /*!< \brief SCTLR: nTWI Mask */

#define SCTLR_V_Pos                      13U                                    /*!< \brief SCTLR: V Position */
#define SCTLR_V_Msk                      (1UL << SCTLR_V_Pos)                   /*!< \brief SCTLR: V Mask */

#define SCTLR_I_Pos                      12U                                    /*!< \brief SCTLR: I Position */
#define SCTLR_I_Msk                      (1UL << SCTLR_I_Pos)                   /*!< \brief SCTLR: I Mask */

#define SCTLR_SED_Pos                    8U                                     /*!< \brief SCTLR: SED Position */
#define SCTLR_SED_Msk                    (1UL << SCTLR_SED_Pos)                 /*!< \brief SCTLR: SED Mask */

#define SCTLR_ITD_Pos                    7U                                     /*!< \brief SCTLR: ITD Position */
#define SCTLR_ITD_Msk                    (1UL << SCTLR_ITD_Pos)                 /*!< \brief SCTLR: ITD Mask */

#define SCTLR_THEE_Pos                   6U                                     /*!< \brief SCTLR: THEE Position */
#define SCTLR_THEE_Msk                   (1UL << SCTLR_THEE_Pos)                /*!< \brief SCTLR: THEE Mask */

#define SCTLR_CP15BEN_Pos                5U                                     /*!< \brief SCTLR: CP15BEN Position */
#define SCTLR_CP15BEN_Msk                (1UL << SCTLR_CP15BEN_Pos)             /*!< \brief SCTLR: CP15BEN Mask */

#define SCTLR_C_Pos                      2U                                     /*!< \brief SCTLR: C Position */
#define SCTLR_C_Msk                      (1UL << SCTLR_C_Pos)                   /*!< \brief SCTLR: C Mask */

#define SCTLR_A_Pos                      1U                                     /*!< \brief SCTLR: A Position */
#define SCTLR_A_Msk                      (1UL << SCTLR_A_Pos)                   /*!< \brief SCTLR: A Mask */

#define SCTLR_M_Pos                      0U                                     /*!< \brief SCTLR: M Position */
#define SCTLR_M_Msk                      (1UL << SCTLR_M_Pos)                   /*!< \brief SCTLR: M Mask */


/* Register TCR_EL3 */
typedef union
{
  struct
  {
    uint64_t T0SZ:6;                     //[5:0]
    RESERVED(1:2, uint64_t)              //[7:6]
    uint64_t IRGN0:2;                    //[9:8]
    uint64_t ORGN0:2;                    //[11:10]
    uint64_t SH0:2;                      //[13:12]
    uint64_t TG0:2;                      //[15:14]
    uint64_t PS:3;                       //[18:16]
    RESERVED(2:1, uint64_t)              //[19]
    uint64_t TBI:1;                      //[20]
    uint64_t HA:1;                       //[21]
    uint64_t HD:1;                       //[22]
    RESERVED(3:1, uint64_t)              //[23]
    uint64_t HPD:1;                      //[24]
    uint64_t HWU59:1;                    //[25]
    uint64_t HWU60:1;                    //[26]
    uint64_t HWU61:1;                    //[27]
    uint64_t HWU62:1;                    //[28]
    uint64_t TBID:1;                     //[29]
    uint64_t TCMA:1;                     //[30]
    RESERVED(4:1, uint64_t)              //[31]
    RESERVED(5:32, uint64_t)             //[63:32]
  } b;
  uint64_t w;                            /*!< \brief Type      used for word access */
} TCR_EL3_Type;


/* Register MPIDR_EL1 */
typedef union
{
  struct
  {
    uint64_t Aff0:8;
    uint64_t Aff1:8;
    uint64_t Aff2:8;
    uint64_t MT:1;
    RESERVED(0:5, uint64_t)
    uint64_t U:1;
    RESERVED(1:1, uint64_t)
    uint64_t Aff3:8;
    RESERVED(2:24, uint64_t)
  } b;                                   /*!< \brief Structure used for bit  access */
  uint64_t w;                            /*!< \brief Type      used for word access */
} MPIDR_EL1_Type;


 /*******************************************************************************
  *                Hardware Abstraction Layer
   Core Function Interface contains:
   - L1 Cache Functions
   - L2C-310 Cache Controller Functions 
   - PL1 Timer Functions
   - GIC Functions
   - MMU Functions
  ******************************************************************************/
 
/* ##########################  L1 Cache functions  ################################# */

/** \brief Enable Caches by setting I and C bits in SCTLR register.
*/
__STATIC_FORCEINLINE void L1C_EnableCaches(void) {
  __set_SCTLR_EL3( __get_SCTLR_EL3() | SCTLR_I_Msk | SCTLR_C_Msk);
  __ISB();
}

/** \brief Disable Caches by clearing I and C bits in SCTLR register.
*/
__STATIC_FORCEINLINE void L1C_DisableCaches(void) {
  __set_SCTLR_EL3( __get_SCTLR_EL3() & (~SCTLR_I_Msk) & (~SCTLR_C_Msk));
  __ISB();
}

/** \brief  Enable Branch Prediction by setting Z bit in SCTLR register.
*/
__STATIC_FORCEINLINE void L1C_EnableBTAC(void) {

}

/** \brief  Disable Branch Prediction by clearing Z bit in SCTLR register.
*/
__STATIC_FORCEINLINE void L1C_DisableBTAC(void) {

}

/** \brief  Invalidate entire branch predictor array
*/
__STATIC_FORCEINLINE void L1C_InvalidateBTAC(void) {

}

/** \brief  Invalidate the whole instruction cache
*/
__STATIC_FORCEINLINE void L1C_InvalidateICacheAll(void) {

}

/** \brief  Invalidate the whole data cache.
*/
__STATIC_FORCEINLINE void L1C_InvalidateDCacheAll(void) {

}


/* ##########################  L2 Cache functions  ################################# */
#if (__L2C_PRESENT == 1U) || defined(DOXYGEN)

/** \brief Enable Level 2 Cache
*/
__STATIC_INLINE void L2C_Enable(void)
{

}
#endif

/* ##########################  L3 Cache functions  ################################# */
#if (__L3C_PRESENT == 1U) || defined(DOXYGEN)

#endif

/* ##########################  GIC functions  ###################################### */
#if (__GIC_PRESENT == 1U) || defined(DOXYGEN)

#endif

/* ##########################  Generic Timer functions  ############################ */
#if (__TIM_PRESENT == 1U) || defined(DOXYGEN)

#endif

/* ##########################  MMU functions  ###################################### */

/** \brief  Enable MMU
*/
__STATIC_INLINE void MMU_Enable(void)
{
  __set_SCTLR_EL3( __get_SCTLR_EL3() | SCTLR_M_Msk);
  __ISB();
}

/** \brief  Disable MMU
*/
__STATIC_INLINE void MMU_Disable(void)
{
  __set_SCTLR_EL3( __get_SCTLR_EL3() & (~SCTLR_M_Msk));
  __ISB();
}

/** \brief  Invalidate entire unified TLB
*/

__STATIC_INLINE void MMU_InvalidateTLB(void)
{
  __DSB();
  __ASM volatile("tlbi vmalle1is");
  __DSB();
  __ISB();
}


#ifdef __cplusplus
}
#endif

#endif /* __CORE_CA53_H_DEPENDANT */

#endif /* __CMSIS_GENERIC */
