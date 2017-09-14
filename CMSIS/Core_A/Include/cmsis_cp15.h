/**************************************************************************//**
 * @file     cmsis_cp15.h
 * @brief    CMSIS compiler specific macros, functions, instructions
 * @version  V1.0.1
 * @date     07. Sep 2017
 ******************************************************************************/
/*
 * Copyright (c) 2009-2017 ARM Limited. All rights reserved.
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

#ifndef __CMSIS_CP15_H
#define __CMSIS_CP15_H

/** \brief  Get ACTLR
    \return               Auxiliary Control register value
 */
__STATIC_FORCEINLINE uint32_t __get_ACTLR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c1, c0, 1" : "=r" (result) : : "memory" );
  __get_CP(15, 0, result, 1, 0, 1);
  return(result);
}

/** \brief  Set ACTLR
    \param [in]    actlr  Auxiliary Control value to set
 */
__STATIC_FORCEINLINE void __set_ACTLR(uint32_t actlr)
{
  // __ASM volatile ("MCR p15, 0, %0, c1, c0, 1" : : "r" (actlr) : "memory");
  __set_CP(15, 0, actlr, 1, 0, 1);
}

/** \brief  Get CPACR
    \return               Coprocessor Access Control register value
 */
__STATIC_FORCEINLINE uint32_t __get_CPACR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c1, c0, 2" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 1, 0, 2);
  return result;
}

/** \brief  Set CPACR
    \param [in]    cpacr  Coprocessor Access Control value to set
 */
__STATIC_FORCEINLINE void __set_CPACR(uint32_t cpacr)
{
//   __ASM volatile("MCR p15, 0, %0, c1, c0, 2" : : "r"(cpacr) : "memory");
  __set_CP(15, 0, cpacr, 1, 0, 2);
}

/** \brief  Get DFSR
    \return               Data Fault Status Register value
 */
__STATIC_FORCEINLINE uint32_t __get_DFSR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c5, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 5, 0, 0);
  return result;
}

/** \brief  Set DFSR
    \param [in]    dfsr  Data Fault Status value to set
 */
__STATIC_FORCEINLINE void __set_DFSR(uint32_t dfsr)
{
//   __ASM volatile("MCR p15, 0, %0, c5, c0, 0" : : "r"(dfsr) : "memory");
  __set_CP(15, 0, dfsr, 5, 0, 0);
}

/** \brief  Get IFSR
    \return               Instruction Fault Status Register value
 */
__STATIC_FORCEINLINE uint32_t __get_IFSR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c5, c0, 1" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 5, 0, 1);
  return result;
}

/** \brief  Set IFSR
    \param [in]    ifsr  Instruction Fault Status value to set
 */
__STATIC_FORCEINLINE void __set_IFSR(uint32_t ifsr)
{
//   __ASM volatile("MCR p15, 0, %0, c5, c0, 1" : : "r"(ifsr) : "memory");
  __set_CP(15, 0, ifsr, 5, 0, 1);
}

/** \brief  Get ISR
    \return               Interrupt Status Register value
 */
__STATIC_FORCEINLINE uint32_t __get_ISR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c12, c1, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 12, 1, 0);
  return result;
}

/** \brief  Get CBAR
    \return               Configuration Base Address register value
 */
__STATIC_FORCEINLINE uint32_t __get_CBAR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 4, %0, c15, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 4, result, 15, 0, 0);
  return result;
}

/** \brief  Get TTBR0

    This function returns the value of the Translation Table Base Register 0.

    \return               Translation Table Base Register 0 value
 */
__STATIC_FORCEINLINE uint32_t __get_TTBR0(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c2, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 2, 0, 0);
  return result;
}

/** \brief  Set TTBR0

    This function assigns the given value to the Translation Table Base Register 0.

    \param [in]    ttbr0  Translation Table Base Register 0 value to set
 */
__STATIC_FORCEINLINE void __set_TTBR0(uint32_t ttbr0)
{
//   __ASM volatile("MCR p15, 0, %0, c2, c0, 0" : : "r"(ttbr0) : "memory");
  __set_CP(15, 0, ttbr0, 2, 0, 0);
}

/** \brief  Get DACR

    This function returns the value of the Domain Access Control Register.

    \return               Domain Access Control Register value
 */
__STATIC_FORCEINLINE uint32_t __get_DACR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c3, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 3, 0, 0);
  return result;
}

/** \brief  Set DACR

    This function assigns the given value to the Domain Access Control Register.

    \param [in]    dacr   Domain Access Control Register value to set
 */
__STATIC_FORCEINLINE void __set_DACR(uint32_t dacr)
{
//   __ASM volatile("MCR p15, 0, %0, c3, c0, 0" : : "r"(dacr) : "memory");
  __set_CP(15, 0, dacr, 3, 0, 0);
}

/** \brief  Set SCTLR

    This function assigns the given value to the System Control Register.

    \param [in]    sctlr  System Control Register value to set
 */
__STATIC_FORCEINLINE void __set_SCTLR(uint32_t sctlr)
{
//   __ASM volatile("MCR p15, 0, %0, c1, c0, 0" : : "r"(sctlr) : "memory");
  __set_CP(15, 0, sctlr, 1, 0, 0);
}

/** \brief  Get SCTLR
    \return               System Control Register value
 */
__STATIC_FORCEINLINE uint32_t __get_SCTLR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c1, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 1, 0, 0);
  return result;
}

/** \brief  Set ACTRL
    \param [in]    actrl  Auxiliary Control Register value to set
 */
__STATIC_FORCEINLINE void __set_ACTRL(uint32_t actrl)
{
//   __ASM volatile("MCR p15, 0, %0, c1, c0, 1" : : "r"(actrl) : "memory");
  __set_CP(15, 0, actrl, 1, 0, 1);
}

/** \brief  Get ACTRL
    \return               Auxiliary Control Register value
 */
__STATIC_FORCEINLINE uint32_t __get_ACTRL(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c1, c0, 1" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 1, 0, 1);
  return result;
}

/** \brief  Get MPIDR

    This function returns the value of the Multiprocessor Affinity Register.

    \return               Multiprocessor Affinity Register value
 */
__STATIC_FORCEINLINE uint32_t __get_MPIDR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c0, c0, 5" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 0, 0, 5);
  return result;
}

 /** \brief  Get VBAR

    This function returns the value of the Vector Base Address Register.

    \return               Vector Base Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_VBAR(void)
{
  uint32_t result;
//   __ASM volatile("MRC p15, 0, %0, c12, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 12, 0, 0);
  return result;
}

/** \brief  Set VBAR

    This function assigns the given value to the Vector Base Address Register.

    \param [in]    vbar  Vector Base Address Register value to set
 */
__STATIC_FORCEINLINE void __set_VBAR(uint32_t vbar)
{
//   __ASM volatile("MCR p15, 0, %0, c12, c0, 1" : : "r"(vbar) : "memory");
  __set_CP(15, 0, vbar, 12, 0, 1);
}

#if (defined(__CORTEX_A) && (__CORTEX_A == 7U) && \
    defined(__TIM_PRESENT) && (__TIM_PRESENT == 1U)) || \
    defined(DOXYGEN)

/** \brief  Set CNTFRQ

  This function assigns the given value to PL1 Physical Timer Counter Frequency Register (CNTFRQ).

  \param [in]    value  CNTFRQ Register value to set
*/
__STATIC_FORCEINLINE void __set_CNTFRQ(uint32_t value)
{
  // __ASM volatile("MCR p15, 0, %0, c14, c0, 0" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 14, 0, 0);
}

/** \brief  Get CNTFRQ

    This function returns the value of the PL1 Physical Timer Counter Frequency Register (CNTFRQ).

    \return               CNTFRQ Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CNTFRQ(void)
{
  uint32_t result;
  // __ASM volatile("MRC p15, 0, %0, c14, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 14, 0 , 0);
  return result;
}

/** \brief  Set CNTP_TVAL

  This function assigns the given value to PL1 Physical Timer Value Register (CNTP_TVAL).

  \param [in]    value  CNTP_TVAL Register value to set
*/
__STATIC_FORCEINLINE void __set_CNTP_TVAL(uint32_t value)
{
  // __ASM volatile("MCR p15, 0, %0, c14, c2, 0" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 14, 2, 0);
}

/** \brief  Get CNTP_TVAL

    This function returns the value of the PL1 Physical Timer Value Register (CNTP_TVAL).

    \return               CNTP_TVAL Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CNTP_TVAL(void)
{
  uint32_t result;
  // __ASM volatile("MRC p15, 0, %0, c14, c2, 0" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 14, 2, 0);
  return result;
}

/** \brief  Set CNTP_CTL

  This function assigns the given value to PL1 Physical Timer Control Register (CNTP_CTL).

  \param [in]    value  CNTP_CTL Register value to set
*/
__STATIC_FORCEINLINE void __set_CNTP_CTL(uint32_t value)
{
  // __ASM volatile("MCR p15, 0, %0, c14, c2, 1" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 14, 2, 1);
}

/** \brief  Get CNTP_CTL register
    \return               CNTP_CTL Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CNTP_CTL(void)
{
  uint32_t result;
  // __ASM volatile("MRC p15, 0, %0, c14, c2, 1" : "=r"(result) : : "memory");
  __get_CP(15, 0, result, 14, 2, 1);
  return result;
}

#endif

/** \brief  Set TLBIALL

  TLB Invalidate All
 */
__STATIC_FORCEINLINE void __set_TLBIALL(uint32_t value)
{
//   __ASM volatile("MCR p15, 0, %0, c8, c7, 0" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 8, 7, 0);
}

/** \brief  Set BPIALL.

  Branch Predictor Invalidate All
 */
__STATIC_FORCEINLINE void __set_BPIALL(uint32_t value)
{
//   __ASM volatile("MCR p15, 0, %0, c7, c5, 6" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 7, 5, 6);
}

/** \brief  Set ICIALLU

  Instruction Cache Invalidate All
 */
__STATIC_FORCEINLINE void __set_ICIALLU(uint32_t value)
{
//   __ASM volatile("MCR p15, 0, %0, c7, c5, 0" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 7, 5, 0);
}

/** \brief  Set DCCMVAC

  Data cache clean
 */
__STATIC_FORCEINLINE void __set_DCCMVAC(uint32_t value)
{
//   __ASM volatile("MCR p15, 0, %0, c7, c10, 1" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 7, 10, 1);
}

/** \brief  Set DCIMVAC

  Data cache invalidate
 */
__STATIC_FORCEINLINE void __set_DCIMVAC(uint32_t value)
{
//   __ASM volatile("MCR p15, 0, %0, c7, c6, 1" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 7, 6, 1);
}

/** \brief  Set DCCIMVAC

  Data cache clean and invalidate
 */
__STATIC_FORCEINLINE void __set_DCCIMVAC(uint32_t value)
{
//   __ASM volatile("MCR p15, 0, %0, c7, c14, 1" : : "r"(value) : "memory");
  __set_CP(15, 0, value, 7, 14, 1);
}


/** \brief  Set CCSIDR
 */
__STATIC_FORCEINLINE void __set_CCSIDR(uint32_t value)
{
//  __ASM volatile("MCR p15, 2, %0, c0, c0, 0" : : "r"(value) : "memory");
  __set_CP(15, 2, value, 0, 0, 0);
}

/** \brief  Get CCSIDR
    \return CCSIDR Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CCSIDR(void)
{
  uint32_t result;
//  __ASM volatile("MRC p15, 1, %0, c0, c0, 0" : "=r"(result) : : "memory");
  __get_CP(15, 1, result, 0, 0, 0);
  return result;
}

/** \brief  Get CLIDR
    \return CLIDR Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CLIDR(void)
{
  uint32_t result;
//  __ASM volatile("MRC p15, 1, %0, c0, c0, 1" : "=r"(result) : : "memory");
  __get_CP(15, 1, result, 0, 0, 1);
  return result;
}

#endif
