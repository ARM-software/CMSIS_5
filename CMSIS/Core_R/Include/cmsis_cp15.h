/**************************************************************************//**
 * @file     cmsis_cp15.h
 * @brief    CMSIS compiler specific macros, functions, instructions
 * @version  
 * @date     
 ******************************************************************************/
/*
 * Copyright (c) 2019 ARM Limited. All rights reserved.
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
#endif

#ifndef __CMSIS_CP15_H
#define __CMSIS_CP15_H


/** \brief  Get MPIDR

    This function returns the value of the Multiprocessor Affinity Register.

    \return               Multiprocessor Affinity Register value
 */
__STATIC_FORCEINLINE uint32_t __get_MPIDR(void)
{
  uint32_t result;
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
  __get_CP(15, 0, result, 12, 0, 0);
  return result;
}

/** \brief  Set VBAR

    This function assigns the given value to the Vector Base Address Register.

    \param [in]    vbar  Vector Base Address Register value to set
 */
__STATIC_FORCEINLINE void __set_VBAR(uint32_t vbar)
{
  __set_CP(15, 0, vbar, 12, 0, 0);
}

/** \brief  Get HVBAR

    This function returns the value of the Hypervisor Vector Base Address Register.

    \return               Hypervisor Vector Base Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_HVBAR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 12, 0, 1);
  return result;
}

/** \brief  Set HVBAR

    This function assigns the given value to the Hypervisor Vector Base Address Register.

    \param [in]    hvbar  Hypervisor Vector Base Address Register value to set
 */
__STATIC_FORCEINLINE void __set_HVBAR(uint32_t hvbar)
{
  __set_CP(15, 4, hvbar, 12, 0, 1);
}

/** \brief  Get HSCTRL

    This function returns the value of the Hypervisor System Control Register.

    \return            Hypervisor System Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_HSCTRL(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 1, 0, 0);
  return result;
}

/** \brief  Set HSCTRL

    This function assigns the given value to the Hypervisor System Control Register.

    \param [in]    hsctrl  Hypervisor System Control Register value to set
 */
__STATIC_FORCEINLINE void __set_HSCTRL(uint32_t hsctrl)
{
  __set_CP(15, 4, hsctrl, 1, 0, 0);
}

/** \brief  Get SCTRL

    This function returns the value of the System Control Register.

    \return            System Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_SCTRL(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 1, 0, 0);
  return result;
}

/** \brief  Set SCTRL

    This function assigns the given value to the System Control Register.

    \param [in]    sctrl  System Control Register value to set
 */
__STATIC_FORCEINLINE void __set_SCTRL(uint32_t sctrl)
{
  __set_CP(15, 0, sctrl, 1, 0, 0);
}

/** \brief  Set ATCMREGIONR register

    This function assigns the given value to the TCM Region Register A.

    \param [in]    config  ATCMREGIONR Register value to set
 */
__STATIC_FORCEINLINE void __set_ATCMREGIONR(uint32_t config)
{
  __set_CP(15, 0, config, 9, 1, 0);
}

/** \brief  Set BTCMREGIONR config register

    This function assigns the given value to the TCM Region Register B.

    \param [in]    config  BTCMREGIONR Register value to set
 */
__STATIC_FORCEINLINE void __set_BTCMREGIONR(uint32_t config)
{
  __set_CP(15, 0, config, 9, 1, 1);
}

/** \brief  Set CTCMREGIONR config register

    This function assigns the given value to the TCM Region Register C.

    \param [in]    config  CTCMREGIONR Register value to set
 */
__STATIC_FORCEINLINE void __set_CTCMREGIONR(uint32_t config)
{
  __set_CP(15, 0, config, 9, 1, 2);
}


#endif
