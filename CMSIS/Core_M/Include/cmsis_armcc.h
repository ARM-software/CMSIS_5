/**************************************************************************//**
 * @file     cmsis_armcc.h
 * @brief    CMSIS compiler ARMCC (Arm Compiler 5) header file
 * @version  V5.4.0
 * @date     04. December 2022
 ******************************************************************************/
/*
 * Copyright (c) 2009-2022 Arm Limited. All rights reserved.
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

#ifndef __CMSIS_ARMCC_H
#define __CMSIS_ARMCC_H

// Include the generic settigs:
#include "../../Core/cmsis_generic_armcc.h"

/* #########################  Startup and Lowlevel Init  ######################## */

#ifndef __PROGRAM_START
#define __PROGRAM_START           __main
#endif

#ifndef __INITIAL_SP
#define __INITIAL_SP              Image$$ARM_LIB_STACK$$ZI$$Limit
#endif

#ifndef __STACK_LIMIT
#define __STACK_LIMIT             Image$$ARM_LIB_STACK$$ZI$$Base
#endif

#ifndef __VECTOR_TABLE
#define __VECTOR_TABLE            __Vectors
#endif

#ifndef __VECTOR_TABLE_ATTRIBUTE
#define __VECTOR_TABLE_ATTRIBUTE  __attribute__((used, section("RESET")))
#endif



/**
  \brief   Get Control Register
  \details Returns the content of the Control Register.
  \return               Control Register value
 */
__STATIC_INLINE uint32_t __get_CONTROL(void)
{
  register uint32_t __regControl         __ASM("control");
  return(__regControl);
}


/**
  \brief   Set Control Register
  \details Writes the given value to the Control Register.
  \param [in]    control  Control Register value to set
 */
__STATIC_INLINE void __set_CONTROL(uint32_t control)
{
  register uint32_t __regControl         __ASM("control");
  __regControl = control;
  __ISB();
}


/**
  \brief   Get IPSR Register
  \details Returns the content of the IPSR Register.
  \return               IPSR Register value
 */
__STATIC_INLINE uint32_t __get_IPSR(void)
{
  register uint32_t __regIPSR          __ASM("ipsr");
  return(__regIPSR);
}


/**
  \brief   Get APSR Register
  \details Returns the content of the APSR Register.
  \return               APSR Register value
 */
__STATIC_INLINE uint32_t __get_APSR(void)
{
  register uint32_t __regAPSR          __ASM("apsr");
  return(__regAPSR);
}


/**
  \brief   Get xPSR Register
  \details Returns the content of the xPSR Register.
  \return               xPSR Register value
 */
__STATIC_INLINE uint32_t __get_xPSR(void)
{
  register uint32_t __regXPSR          __ASM("xpsr");
  return(__regXPSR);
}


/**
  \brief   Get Process Stack Pointer
  \details Returns the current value of the Process Stack Pointer (PSP).
  \return               PSP Register value
 */
__STATIC_INLINE uint32_t __get_PSP(void)
{
  register uint32_t __regProcessStackPointer  __ASM("psp");
  return(__regProcessStackPointer);
}


/**
  \brief   Set Process Stack Pointer
  \details Assigns the given value to the Process Stack Pointer (PSP).
  \param [in]    topOfProcStack  Process Stack Pointer value to set
 */
__STATIC_INLINE void __set_PSP(uint32_t topOfProcStack)
{
  register uint32_t __regProcessStackPointer  __ASM("psp");
  __regProcessStackPointer = topOfProcStack;
}


/**
  \brief   Get Main Stack Pointer
  \details Returns the current value of the Main Stack Pointer (MSP).
  \return               MSP Register value
 */
__STATIC_INLINE uint32_t __get_MSP(void)
{
  register uint32_t __regMainStackPointer     __ASM("msp");
  return(__regMainStackPointer);
}


/**
  \brief   Set Main Stack Pointer
  \details Assigns the given value to the Main Stack Pointer (MSP).
  \param [in]    topOfMainStack  Main Stack Pointer value to set
 */
__STATIC_INLINE void __set_MSP(uint32_t topOfMainStack)
{
  register uint32_t __regMainStackPointer     __ASM("msp");
  __regMainStackPointer = topOfMainStack;
}


/**
  \brief   Get Priority Mask
  \details Returns the current state of the priority mask bit from the Priority Mask Register.
  \return               Priority Mask value
 */
__STATIC_INLINE uint32_t __get_PRIMASK(void)
{
  register uint32_t __regPriMask         __ASM("primask");
  return(__regPriMask);
}


/**
  \brief   Set Priority Mask
  \details Assigns the given value to the Priority Mask Register.
  \param [in]    priMask  Priority Mask
 */
__STATIC_INLINE void __set_PRIMASK(uint32_t priMask)
{
  register uint32_t __regPriMask         __ASM("primask");
  __regPriMask = (priMask);
}


#if ( defined (__arm__         ) || \
     (defined (__ARM_ARCH_7M__ ) && (__ARM_ARCH_7M__  == 1)) || \
     (defined (__ARM_ARCH_7EM__) && (__ARM_ARCH_7EM__ == 1))     )
  /**
    \brief   Get Base Priority
    \details Returns the current value of the Base Priority register.
    \return               Base Priority register value
   */
  __STATIC_INLINE uint32_t  __get_BASEPRI(void)
  {
    register uint32_t __regBasePri         __ASM("basepri");
    return(__regBasePri);
  }
  
  
  /**
    \brief   Set Base Priority
    \details Assigns the given value to the Base Priority register.
    \param [in]    basePri  Base Priority value to set
   */
  __STATIC_INLINE void __set_BASEPRI(uint32_t basePri)
  {
    register uint32_t __regBasePri         __ASM("basepri");
    __regBasePri = (basePri & 0xFFU);
  }
  
  
  /**
    \brief   Set Base Priority with condition
    \details Assigns the given value to the Base Priority register only if BASEPRI masking is disabled,
             or the new value increases the BASEPRI priority level.
    \param [in]    basePri  Base Priority value to set
   */
  __STATIC_INLINE void __set_BASEPRI_MAX(uint32_t basePri)
  {
    register uint32_t __regBasePriMax      __ASM("basepri_max");
    __regBasePriMax = (basePri & 0xFFU);
  }
  
  
  /**
    \brief   Get Fault Mask
    \details Returns the current value of the Fault Mask register.
    \return               Fault Mask register value
   */
  __STATIC_INLINE uint32_t __get_FAULTMASK(void)
  {
    register uint32_t __regFaultMask       __ASM("faultmask");
    return(__regFaultMask);
  }
  
  
  /**
    \brief   Set Fault Mask
    \details Assigns the given value to the Fault Mask register.
    \param [in]    faultMask  Fault Mask value to set
   */
  __STATIC_INLINE void __set_FAULTMASK(uint32_t faultMask)
  {
    register uint32_t __regFaultMask       __ASM("faultmask");
    __regFaultMask = (faultMask & (uint32_t)1U);
  }
#endif /* ( defined (__arm__         ) || \
           (defined (__ARM_ARCH_7M__ ) && (__ARM_ARCH_7M__  == 1)) || \
           (defined (__ARM_ARCH_7EM__) && (__ARM_ARCH_7EM__ == 1))     ) */


/*@} end of CMSIS_Core_RegAccFunctions */


#endif /* __CMSIS_ARMCC_H */
