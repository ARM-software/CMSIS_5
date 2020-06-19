/**************************************************************************//**
 * @file     cmsis_cp15.h
 * @brief    CMSIS compiler specific macros, functions, instructions
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
#endif

#ifndef __CMSIS_CP15_H
#define __CMSIS_CP15_H

/** \brief  Get MIDR
    This function returns the value of the Main ID Register.
    \return               Main ID Register value
 */
__STATIC_FORCEINLINE uint32_t __get_MIDR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 0);
  return result;
}

/** \brief  Get CTR
    This function returns the value of the Cache Type Register.
    \return               Cache Type Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CTR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 1);
  return result;
}

/** \brief  Get TCMTR
    This function returns the value of the TCM Type Register.
    \return               TCM Type Register value
 */
__STATIC_FORCEINLINE uint32_t __get_TCMTR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 2);
  return result;
}

/** \brief  Get TLBTR
    This function returns the value of the TLB Type Register.
    \return               TLB Type Register value
 */
__STATIC_FORCEINLINE uint32_t __get_TLBTR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 3);
  return result;
}

/** \brief  Get MPUIR
    This function returns the value of the MPU Type Register.
    \return               MPU Type Register value
 */
__STATIC_FORCEINLINE uint32_t __get_MPUIR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 4);
  return result;
}

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

/** \brief  Get REVIDR
    This function returns the value of the Revision ID Register.
    \return               Revision ID Register value
 */
__STATIC_FORCEINLINE uint32_t __get_REVIDR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 6);
  return result;
}

/** \brief  Get MIDR alias
    This function returns the value of the Alias of the Main ID Register,.
    \return               Alias of the Main ID Register value
 */
__STATIC_FORCEINLINE uint32_t __get_MIDR_alias(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 0, 7);
  return result;
}

/** \brief  Get ID_PFR0
    This function returns the value of the Processor Feature Register 0.
    \return               Processor Feature Register 0 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_PFR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 0);
  return result;
}

/** \brief  Get ID_PFR1
    This function returns the value of the Processor Feature Register 1.
    \return               Processor Feature Register 1 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_PFR1(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 1);
  return result;
}

/** \brief  Get ID_DFR0
    This function returns the value of the Debug Feature Register 0.
    \return               Debug Feature Register 0 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_DFR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 2);
  return result;
}

/** \brief  Get ID_AFR0
    This function returns the value of the Auxiliary Feature Register 0.
    \return               Auxiliary Feature Register 0 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_AFR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 3);
  return result;
}

/** \brief  Get ID_MMFR0
    This function returns the value of the Memory Model Feature Register 0.
    \return               Memory Model Feature Register 0 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_MMFR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 4);
  return result;
}

/** \brief  Get ID_MMFR1
    This function returns the value of the Memory Model Feature Register 1.
    \return               Memory Model Feature Register 1 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_MMFR1(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 5);
  return result;
}

/** \brief  Get ID_MMFR2
    This function returns the value of the Memory Model Feature Register 2.
    \return               Memory Model Feature Register 2 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_MMFR2(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 6);
  return result;
}

/** \brief  Get ID_MMFR3
    This function returns the value of the Memory Model Feature Register 3.
    \return               Memory Model Feature Register 3 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_MMFR3(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 1, 7);
  return result;
}

/** \brief  Get ID_ISAR0
    This function returns the value of the Instruction Set Attribute Register 0.
    \return               Instruction Set Attribute Register 0 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_ISAR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 0);
  return result;
}

/** \brief  Get ID_ISAR1
    This function returns the value of the Instruction Set Attribute Register 1.
    \return               Instruction Set Attribute Register 1 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_ISAR1(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 1);
  return result;
}

/** \brief  Get ID_ISAR2
    This function returns the value of the Instruction Set Attribute Register 2.
    \return               Instruction Set Attribute Register 2 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_ISAR2(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 2);
  return result;
}

/** \brief  Get ID_ISAR3
    This function returns the value of the Instruction Set Attribute Register 3.
    \return               Instruction Set Attribute Register 3 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_ISAR3(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 3);
  return result;
}

/** \brief  Get ID_ISAR4
    This function returns the value of the Instruction Set Attribute Register 4.
    \return               Instruction Set Attribute Register 4 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_ISAR4(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 4);
  return result;
}

/** \brief  Get ID_ISAR5
    This function returns the value of the Instruction Set Attribute Register 5.
    \return               Instruction Set Attribute Register 5 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_ISAR5(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 5);
  return result;
}

/** \brief  Get ID_MMFR4
    This function returns the value of the Memory Model Feature Register 4.
    \return               Memory Model Feature Register 4 value
 */
__STATIC_FORCEINLINE uint32_t __get_ID_MMFR4(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 0, 2, 6);
  return result;
}

/** \brief  Get SCTLR
    This function returns the value of the System Control Register.
    \return            System Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_SCTLR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 1, 0, 0);
  return result;
}

/** \brief  Set SCTLR
    This function assigns the given value to the System Control Register.
    \param [in]    sctlr  System Control Register value to set
 */
__STATIC_FORCEINLINE void __set_SCTLR(uint32_t sctlr)
{
  __set_CP(15, 0, sctlr, 1, 0, 0);
}

/** \brief  Get NSACR
    This function returns the value of the Non-Secure Access Control Register.
    \return            Non-Secure Access Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_NSACR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 1, 1, 2);
  return result;
}

/** \brief  Get DFAR
    This function returns the value of the Data Fault Address Register.
    \return            Data Fault Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_DFAR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 6, 0, 0);
  return result;
}

/** \brief  Set DFAR
    This function assigns the given value to the Data Fault Address Register.
    \param [in]    dfar  Data Fault Address Register value to set
 */
__STATIC_FORCEINLINE void __set_DFAR(uint32_t dfar)
{
  __set_CP(15, 0, dfar, 6, 0, 0);
}

/** \brief  Get IFAR
    This function returns the value of the Instruction Fault Address Register.
    \return            Instruction Fault Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_IFAR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 6, 0, 2);
  return result;
}

/** \brief  Set IFAR
    This function assigns the given value to the Instruction Fault Address Register.
    \param [in]    ifar  Instruction Fault Address Register value to set
 */
__STATIC_FORCEINLINE void __set_IFAR(uint32_t ifar)
{
  __set_CP(15, 0, ifar, 6, 0, 2);
}

/** \brief  Get PRSELR
    This function returns the value of the Protection Region Selection Register.
    \return            Protection Region Selection Register
 */
__STATIC_FORCEINLINE uint32_t __get_PRSELR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 6, 2, 1);
  return result;
}

/** \brief  Set PRSELR
    This function assigns the given value to the Protection Region Selection Register.
    \param [in]    prselr  Protection Region Selection Register value to set
 */
__STATIC_FORCEINLINE void __set_PRSELR(uint32_t prselr)
{
  __set_CP(15, 0, prselr, 6, 2, 1);
}

/** \brief  Get PRBAR
    This function returns the value of the Protection Region Base Address Register.
    \return            Protection Region Base Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_PRBAR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 6, 3, 0);
  return result;
}

/** \brief  Set PRBAR
    This function assigns the given value to the Protection Region Base Address Register.
    \param [in]    prbar  Protection Region Base Address Register value to set
 */
__STATIC_FORCEINLINE void __set_PRBAR(uint32_t prbar)
{
  __set_CP(15, 0, prbar, 6, 3, 0);
}

/** \brief  Get PRLAR
    This function returns the value of the Protection Region Limit Address Register.
    \return            Protection Region Limit Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_PRLAR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 6, 3, 1);
  return result;
}

/** \brief  Set PRLAR
    This function assigns the given value to the Protection Region Limit Address Register.
    \param [in]    prlar  Protection Region Limit Address Register value to set
 */
__STATIC_FORCEINLINE void __set_PRLAR(uint32_t prlar)
{
  __set_CP(15, 0, prlar, 6, 3, 1);
}

/** \brief  Get PAR
    This function returns the value of the Physical Address Register.
    \return            Physical Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_PAR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 7, 4, 0);
  return result;
}

/** \brief  Set PAR
    This function assigns the given value to the Physical Address Register.
    \param [in]    par  PPhysical Address Register value to set
 */
__STATIC_FORCEINLINE void __set_PAR(uint32_t par)
{
  __set_CP(15, 0, par, 7, 4, 0);
}

/** \brief  Get IMP_ATCMREGIONR
    This function returns the value of the TCM Region Register A.
    \return            TCM Region Register A
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_ATCMREGIONR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 1, 0);
  return result;
}

/** \brief  Set IMP_ATCMREGIONR
    This function assigns the given value to the TCM Region Register A.
    \param [in]    config  TCM Region Register A value to set
 */
__STATIC_FORCEINLINE void __set_IMP_ATCMREGIONR(uint32_t config)
{
  __set_CP(15, 0, config, 9, 1, 0);
}

/** \brief  Get IMP_BTCMREGIONR
    This function returns the value of the TCM Region Register B.
    \return            TCM Region Register B
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_BTCMREGIONR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 1, 1);
  return result;
}

/** \brief  Set IMP_BTCMREGIONR
    This function assigns the given value to the TCM Region Register B.
    \param [in]    config  TCM Region Register B value to set
 */
__STATIC_FORCEINLINE void __set_IMP_BTCMREGIONR(uint32_t config)
{
  __set_CP(15, 0, config, 9, 1, 1);
}

/** \brief  Get IMP_CTCMREGIONR
    This function returns the value of the TCM Region Register C.
    \return            TCM Region Register C
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_CTCMREGIONR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 1, 2);
  return result;
}

/** \brief  Set IMP_CTCMREGIONR
    This function assigns the given value to the TCM Region Register C.
    \param [in]    config  TCM Region Register C value to set
 */
__STATIC_FORCEINLINE void __set_IMP_CTCMREGIONR(uint32_t config)
{
  __set_CP(15, 0, config, 9, 1, 2);
}

/** \brief  Get PMCR
    This function returns the value of the Performance Monitors Control Register.
    \return            Performance Monitors Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMCR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 0);
  return result;
}

/** \brief  Set PMCR
    This function assigns the given value to the Performance Monitors Control Register.
    \param [in]    pmcr  Performance Monitors Control Register value to set
 */
__STATIC_FORCEINLINE void __set_PMCR(uint32_t pmcr)
{
  __set_CP(15, 0, pmcr, 9, 12, 0);
}

/** \brief  Get PMCNTENSET
    This function returns the value of the Performance Monitors Count Enable Set Register.
    \return            Performance Monitors Count Enable Set Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMCNTENSET(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 1);
  return result;
}

/** \brief  Set PMCNTENSET
    This function assigns the given value to the Performance Monitors Count Enable Set Register.
    \param [in]    pmcntenset  Performance Monitors Count Enable Set Register value to set
 */
__STATIC_FORCEINLINE void __set_PMCNTENSET(uint32_t pmcntenset)
{
  __set_CP(15, 0, pmcntenset, 9, 12, 1);
}

/** \brief  Get PMCNTENCLR
    This function returns the value of the Performance Monitors Count Enable Clear Register.
    \return            Performance Monitors Count Enable Clear Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMCNTENCLR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 2);
  return result;
}

/** \brief  Set PMCNTENCLR
    This function assigns the given value to the Performance Monitors Count Enable Clear Register.
    \param [in]    pmcntenclr  Performance Monitors Count Enable Clear Register value to set
 */
__STATIC_FORCEINLINE void __set_PMCNTENCLR(uint32_t pmcntenclr)
{
  __set_CP(15, 0, pmcntenclr, 9, 12, 2);
}

/** \brief  Get PMOVSR
    This function returns the value of the Performance Monitor Overflow Flag Status Clear Register.
    \return            Performance Monitor Overflow Flag Status Clear Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMOVSR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 3);
  return result;
}

/** \brief  Set PMOVSR
    This function assigns the given value to the Performance Monitor Overflow Flag Status Clear Register.
    \param [in]    pmovsr  Performance Monitor Overflow Flag Status Clear Register value to set
 */
__STATIC_FORCEINLINE void __set_PMOVSR(uint32_t pmovsr)
{
  __set_CP(15, 0, pmovsr, 9, 12, 3);
}

/** \brief  Set PMSWINC
    This function assigns the given value to the Performance Monitors Software Increment Register.
    \param [in]    pmswinc  Performance Monitors Software Increment Register value to set
 */
__STATIC_FORCEINLINE void __set_PMSWINC(uint32_t pmswinc)
{
  __set_CP(15, 0, pmswinc, 9, 12, 4);
}

/** \brief  Get PMSELR
    This function returns the value of the Performance Monitors Event Counter Selection Register.
    \return            Performance Monitors Event Counter Selection Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMSELR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 5);
  return result;
}

/** \brief  Set PMSELR
    This function assigns the given value to the Performance Monitors Event Counter Selection Register.
    \param [in]    pmselr  Performance Monitors Event Counter Selection Register value to set
 */
__STATIC_FORCEINLINE void __set_PMSELR(uint32_t pmselr)
{
  __set_CP(15, 0, pmselr, 9, 12, 5);
}

/** \brief  Get PMCEID0
    This function returns the value of the Performance Monitors Common Event Identification Register 0.
    \return            Performance Monitors Common Event Identification Register 0
 */
__STATIC_FORCEINLINE uint32_t __get_PMCEID0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 6);
  return result;
}

/** \brief  Get PMCEID1
    This function returns the value of the Performance Monitors Common Event Identification Register 1.
    \return            Performance Monitors Common Event Identification Register 1
 */
__STATIC_FORCEINLINE uint32_t __get_PMCEID1(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 12, 7);
  return result;
}

/** \brief  Get PMCCNTR
    This function returns the value of the Performance Monitors Cycle Count Register.
    \return            Performance Monitors Cycle Count Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMCCNTR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 13, 0);
  return result;
}

/** \brief  Set PMCCNTR
    This function assigns the given value to the Performance Monitors Cycle Count Register.
    \param [in]    pmccntr  Performance Monitors Cycle Count Register value to set
 */
__STATIC_FORCEINLINE void __set_PMCCNTR(uint32_t pmccntr)
{
  __set_CP(15, 0, pmccntr, 9, 13, 0);
}

/** \brief  Get PMXEVTYPER
    This function returns the value of the Performance Monitors Selected Event Type Register.
    \return            Performance Monitors Selected Event Type Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMXEVTYPER(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 13, 1);
  return result;
}

/** \brief  Set PMXEVTYPER
    This function assigns the given value to the Performance Monitors Selected Event Type Register.
    \param [in]    pmxevtyper  Performance Monitors Selected Event Type Register value to set
 */
__STATIC_FORCEINLINE void __set_PMXEVTYPER(uint32_t pmxevtyper)
{
  __set_CP(15, 0, pmxevtyper, 9, 13, 1);
}

/** \brief  Get PMXEVCNTR
    This function returns the value of the Performance Monitors Selected Event Count Register.
    \return            Performance Monitors Selected Event Count Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMXEVCNTR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 13, 2);
  return result;
}

/** \brief  Set PMXEVCNTR
    This function assigns the given value to the Performance Monitors Selected Event Count Register.
    \param [in]    pmxevcntr  Performance Monitors Selected Event Count Register value to set
 */
__STATIC_FORCEINLINE void __set_PMXEVCNTR(uint32_t pmxevcntr)
{
  __set_CP(15, 0, pmxevcntr, 9, 13, 2);
}

/** \brief  Get PMUSERENR
    This function returns the value of the Performance Monitors User Enable Register.
    \return            Performance Monitors User Enable Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMUSERENR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 14, 0);
  return result;
}

/** \brief  Set PMUSERENR
    This function assigns the given value to the Performance Monitors User Enable Register.
    \param [in]    pmuserenr  Performance Monitors User Enable Register value to set
 */
__STATIC_FORCEINLINE void __set_PMUSERENR(uint32_t pmuserenr)
{
  __set_CP(15, 0, pmuserenr, 9, 14, 0);
}

/** \brief  Get PMINTENSET
    This function returns the value of the Performance Monitors Interrupt Enable Set Register.
    \return            Performance Monitors Interrupt Enable Set Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMINTENSET(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 14, 1);
  return result;
}

/** \brief  Set PMINTENSET
    This function assigns the given value to the Performance Monitors Interrupt Enable Set Register.
    \param [in]    pmintenset  Performance Monitors Interrupt Enable Set Register value to set
 */
__STATIC_FORCEINLINE void __set_PMINTENSET(uint32_t pmintenset)
{
  __set_CP(15, 0, pmintenset, 9, 14, 1);
}

/** \brief  Get PMINTENCLR
    This function returns the value of the Performance Monitors Interrupt Enable Clear Register.
    \return            Performance Monitors Interrupt Enable Clear Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMINTENCLR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 14, 2);
  return result;
}

/** \brief  Set PMINTENCLR
    This function assigns the given value to the Performance Monitors Interrupt Enable Clear Register.
    \param [in]    pmintenclr  Performance Monitors Interrupt Enable Clear Register value to set
 */
__STATIC_FORCEINLINE void __set_PMINTENCLR(uint32_t pmintenclr)
{
  __set_CP(15, 0, pmintenclr, 9, 14, 2);
}

/** \brief  Get PMOVSSET
    This function returns the value of the Performance Monitor Overflow Flag Status Set Register.
    \return            Performance Monitor Overflow Flag Status Set Register
 */
__STATIC_FORCEINLINE uint32_t __get_PMOVSSET(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 9, 14, 3);
  return result;
}

/** \brief  Set PMOVSSET
    This function assigns the given value to the Performance Monitor Overflow Flag Status Set Register.
    \param [in]    pmovsset  Performance Monitor Overflow Flag Status Set Register value to set
 */
__STATIC_FORCEINLINE void __set_PMOVSSET(uint32_t pmovsset)
{
  __set_CP(15, 0, pmovsset, 9, 14, 3);
}

/** \brief  Get MAIR0
    This function returns the value of the Memory Attribute Indirection Registers 0.
    \return            Memory Attribute Indirection Registers 0
 */
__STATIC_FORCEINLINE uint32_t __get_MAIR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 10, 2, 0);
  return result;
}

/** \brief  Set MAIR0
    This function assigns the given value to the Memory Attribute Indirection Registers 0.
    \param [in]    mair0  Memory Attribute Indirection Registers 0 value to set
 */
__STATIC_FORCEINLINE void __set_MAIR0(uint32_t mair0)
{
  __set_CP(15, 0, mair0, 10, 2, 0);
}

/** \brief  Get MAIR1
    This function returns the value of the Memory Attribute Indirection Registers 1.
    \return            Memory Attribute Indirection Registers 1
 */
__STATIC_FORCEINLINE uint32_t __get_MAIR1(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 10, 2, 1);
  return result;
}

/** \brief  Set MAIR0
    This function assigns the given value to the Memory Attribute Indirection Registers 1.
    \param [in]    mair1  Memory Attribute Indirection Registers 1 value to set
 */
__STATIC_FORCEINLINE void __set_MAIR1(uint32_t mair1)
{
  __set_CP(15, 0, mair1, 10, 2, 1);
}

/** \brief  Get AMAIR0
    This function returns the value of the Auxiliary Memory Attribute Indirection Register 0.
    \return            Auxiliary Memory Attribute Indirection Register 0
 */
__STATIC_FORCEINLINE uint32_t __get_AMAIR0(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 10, 3, 0);
  return result;
}

/** \brief  Set AMAIR0
    This function assigns the given value to the Auxiliary Memory Attribute Indirection Register 0.
    \param [in]    config  Auxiliary Memory Attribute Indirection Register 0 value to set
 */
__STATIC_FORCEINLINE void __set_AMAIR0(uint32_t config)
{
  __set_CP(15, 0, config, 10, 3, 0);
}

/** \brief  Get AMAIR1
    This function returns the value of the Auxiliary Memory Attribute Indirection Register 1.
    \return            Auxiliary Memory Attribute Indirection Register 1
 */
__STATIC_FORCEINLINE uint32_t __get_AMAIR1(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 10, 3, 1);
  return result;
}

/** \brief  Set AMAIR1
    This function assigns the given value to the Auxiliary Memory Attribute Indirection Register 1.
    \param [in]    config  Auxiliary Memory Attribute Indirection Register 1 value to set
 */
__STATIC_FORCEINLINE void __set_AMAIR1(uint32_t config)
{
  __set_CP(15, 0, config, 10, 3, 1);
}

/** \brief  Get IMP_SLAVEPCTLR
    This function returns the value of the Slave Port Control Register.
    \return            Slave Port Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_SLAVEPCTLR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 11, 0, 0);
  return result;
}

/** \brief  Set IMP_SLAVEPCTLR
    This function assigns the given value to the Slave Port Control Register.
    \param [in]    slavepctlr  Slave Port Control Register value to set
 */
__STATIC_FORCEINLINE void __set_IMP_SLAVEPCTLR(uint32_t slavepctlr)
{
  __set_CP(15, 0, slavepctlr, 11, 0, 0);
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

/** \brief  Get RVBAR
    This function returns the value of the Reset Vector Base Address Register.
    \return               Reset Vector Base Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_RVBAR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 12, 0, 1);
  return result;
}

/** \brief  Get ISR
    This function returns the value of the Interrupt Status Register.
    \return               Interrupt Status Register
 */
__STATIC_FORCEINLINE uint32_t __get_ISR(void)
{
  uint32_t result;
  __get_CP(15, 0, result, 12, 1, 0);
  return result;
}

/** \brief  Get CCSIDR
    This function returns the value of the Current Cache Size ID Register.
    \return               Current Cache Size ID Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CCSIDR(void)
{
  uint32_t result;
  __get_CP(15, 1, result, 0, 0, 0);
  return result;
}

/** \brief  Get CLIDR
    This function returns the value of the Cache Level ID Register.
    \return               Cache Level ID Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CLIDR(void)
{
  uint32_t result;
  __get_CP(15, 1, result, 0, 0, 1);
  return result;
}

/** \brief  Get AIDR
    This function returns the value of the Auxiliary ID Register.
    \return               Auxiliary ID Register value
 */
__STATIC_FORCEINLINE uint32_t __get_AIDR(void)
{
  uint32_t result;
  __get_CP(15, 1, result, 0, 0, 7);
  return result;
}

/** \brief  Get IMP_CSCTLR
    This function returns the value of the Cache Segregation Control Register.
    \return            Cache Segregation Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_CSCTLR(void)
{
  uint32_t result;
  __get_CP(15, 1, result, 9, 1, 0);
  return result;
}

/** \brief  Set IMP_CSCTLR
    This function assigns the given value to the Cache Segregation Control Register.
    \param [in]    csctlr  Cache Segregation Control Register value to set
 */
__STATIC_FORCEINLINE void __set_IMP_CSCTLR(uint32_t csctlr)
{
  __set_CP(15, 1, csctlr, 9, 1, 0);
}

/** \brief  Get IMP_BPCTLR
    This function returns the value of the Branch Predictor Control Register.
    \return            Branch Predictor Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_BPCTLR(void)
{
  uint32_t result;
  __get_CP(15, 1, result, 9, 1, 1);
  return result;
}

/** \brief  Set IMP_BPCTLR
    This function assigns the given value to the Branch Predictor Control Register.
    \param [in]    bpctlr  Branch Predictor Control Register value to set
 */
__STATIC_FORCEINLINE void __set_IMP_BPCTLR(uint32_t bpctlr)
{
  __set_CP(15, 1, bpctlr, 9, 1, 1);
}

/** \brief  Get IMP_MEMPROTCLR
    This function returns the value of the Memory Protection Control Register.
    \return            Memory Protection Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_IMP_MEMPROTCLR(void)
{
  uint32_t result;
  __get_CP(15, 1, result, 9, 1, 2);
  return result;
}

/** \brief  Set IMP_MEMPROTCLR
    This function assigns the given value to the Memory Protection Control Register.
    \param [in]    memprotclr  Memory Protection Control Register value to set
 */
__STATIC_FORCEINLINE void __set_IMP_MEMPROTCLR(uint32_t memprotclr)
{
  __set_CP(15, 1, memprotclr, 9, 1, 2);
}

/** \brief  Get CSSELR
    This function returns the value of the Cache Size Selection Register.
    \return               Cache Size Selection Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CSSELR(void)
{
  uint32_t result;
  __get_CP(15, 2, result, 0, 0, 7);
  return result;
}

/** \brief  Get HSCTLR
    This function returns the value of the Hyp System Control Register.
    \return            Hyp System Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_HSCTLR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 1, 0, 0);
  return result;
}

/** \brief  Set HSCTLR
    This function assigns the given value to the Hyp System Control Register.
    \param [in]    hsctlr  Hyp System Control Register value to set
 */
__STATIC_FORCEINLINE void __set_HSCTLR(uint32_t hsctlr)
{
  __set_CP(15, 4, hsctlr, 1, 0, 0);
}

/** \brief  Get HACTLR
    This function returns the value of the Hyp Auxiliary Control Register.
    \return            Hyp Auxiliary Control Register
 */
__STATIC_FORCEINLINE uint32_t __get_HACTLR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 1, 0, 1);
  return result;
}

/** \brief  Set HACTLR
    This function assigns the given value to the Hyp Auxiliary Control Register.
    \param [in]    hactlr  Hyp Auxiliary Control Register value to set
 */
__STATIC_FORCEINLINE void __set_HACTLR(uint32_t hactlr)
{
  __set_CP(15, 4, hactlr, 1, 0, 1);
}

/** \brief  Get HACTLR2
    This function returns the value of the Hyp Auxiliary Control Register 2.
    \return            Hyp Auxiliary Control Register 2
 */
__STATIC_FORCEINLINE uint32_t __get_HACTLR2(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 1, 0, 3);
  return result;
}

/** \brief  Set HACTLR2
    This function assigns the given value to the Hyp Auxiliary Control Register 2.
    \param [in]    hactlr  Hyp Auxiliary Control Register 2 value to set
 */
__STATIC_FORCEINLINE void __set_HACTLR2(uint32_t hactlr2)
{
  __set_CP(15, 4, hactlr2, 1, 0, 3);
}

/** \brief  Get HPRSELR
    This function returns the value of the Hyp Protection Region Selection Register.
    \return               Hyp Protection Region Selection Register
 */
__STATIC_FORCEINLINE uint32_t __get_HPRSELR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 6, 2, 1);
  return result;
}

/** \brief  Set HPRSELR
    This function assigns the given value to the Hyp Protection Region Selection Register.
    \param [in]    hprselr  Hyp Protection Region Selection Register value to set
 */
__STATIC_FORCEINLINE void __set_HPRSELR(uint32_t hprselr)
{
  __set_CP(15, 4, hprselr, 6, 2, 1);
}

/** \brief  Get HPRBAR
    This function returns the value of the Hyp Protection Region Base Address Register.
    \return               Hyp Protection Region Base Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_HPRBAR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 6, 3, 0);
  return result;
}

/** \brief  Set HPRBAR
    This function assigns the given value to the Hyp Protection Region Base Address Register.
    \param [in]    hprbar  Hyp Protection Region Base Address Register value to set
 */
__STATIC_FORCEINLINE void __set_HPRBAR(uint32_t hprbar)
{
  __set_CP(15, 4, hprbar, 6, 3, 0);
}

/** \brief  Get HPRLAR
    This function returns the value of the Hyp Protection Region Limit Address Register.
    \return               Hyp Protection Region Limit Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_HPRLAR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 6, 3, 1);
  return result;
}

/** \brief  Set HPRLAR
    This function assigns the given value to the Hyp Protection Region Limit Address Register.
    \param [in]    hprlar  Hyp Protection Region Limit Address Register value to set
 */
__STATIC_FORCEINLINE void __set_HPRLAR(uint32_t hprlar)
{
  __set_CP(15, 4, hprlar, 6, 3, 1);
}

/** \brief  Get HMAIR0
    This function returns the value of the Hyp Memory Attribute Indirection Register 0.
    \return               Hyp Memory Attribute Indirection Register 0
 */
__STATIC_FORCEINLINE uint32_t __get_HMAIR0(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 10, 2, 0);
  return result;
}

/** \brief  Set HMAIR0
    This function assigns the given value to the Hyp Memory Attribute Indirection Register 0.
    \param [in]    hmair0  Hyp Memory Attribute Indirection Register 0 value to set
 */
__STATIC_FORCEINLINE void __set_HMAIR0(uint32_t hmair0)
{
  __set_CP(15, 4, hmair0, 10, 2, 0);
}

/** \brief  Get HMAIR1
    This function returns the value of the Hyp Memory Attribute Indirection Register 1.
    \return               Hyp Memory Attribute Indirection Register 1
 */
__STATIC_FORCEINLINE uint32_t __get_HMAIR1(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 10, 2, 1);
  return result;
}

/** \brief  Set HMAIR1
    This function assigns the given value to the Hyp Memory Attribute Indirection Register 1.
    \param [in]    hmair1  Hyp Memory Attribute Indirection Register 1 value to set
 */
__STATIC_FORCEINLINE void __set_HMAIR1(uint32_t hmair1)
{
  __set_CP(15, 4, hmair1, 10, 2, 1);
}

/** \brief  Get HAMAIR0
    This function returns the value of the Hyp Auxiliary Memory Attribute Indirection Register 0.
    \return               Hyp Auxiliary Memory Attribute Indirection Register 0
 */
__STATIC_FORCEINLINE uint32_t __get_HAMAIR0(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 10, 3, 0);
  return result;
}

/** \brief  Set HAMAIR0
    This function assigns the given value to the Hyp Auxiliary Memory Attribute Indirection Register 0.
    \param [in]    hamair0  Hyp Auxiliary Memory Attribute Indirection Register 0 value to set
 */
__STATIC_FORCEINLINE void __set_HAMAIR0(uint32_t hamair0)
{
  __set_CP(15, 4, hamair0, 10, 3, 0);
}

/** \brief  Get HAMAIR1
    This function returns the value of the Hyp Auxiliary Memory Attribute Indirection Register 1.
    \return               Hyp Auxiliary Memory Attribute Indirection Register 1
 */
__STATIC_FORCEINLINE uint32_t __get_HAMAIR1(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 10, 3, 1);
  return result;
}

/** \brief  Set HAMAIR0
    This function assigns the given value to the Hyp Auxiliary Memory Attribute Indirection Register 1.
    \param [in]    hamair1  Hyp Auxiliary Memory Attribute Indirection Register 1 value to set
 */
__STATIC_FORCEINLINE void __set_HAMAIR1(uint32_t hamair1)
{
  __set_CP(15, 4, hamair1, 10, 3, 1);
}

/** \brief  Get HVBAR
    This function returns the value of the Hyp Vector Base Address Register.
    \return               Hyp Vector Base Address Register
 */
__STATIC_FORCEINLINE uint32_t __get_HVBAR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 12, 0, 0);
  return result;
}

/** \brief  Set HVBAR
    This function assigns the given value to the Hyp Vector Base Address Register.
    \param [in]    hvbar  Hyp Vector Base Address Register value to set
 */
__STATIC_FORCEINLINE void __set_HVBAR(uint32_t hvbar)
{
  __set_CP(15, 4, hvbar, 12, 0, 0);
}

/** \brief  Get HRMR
    This function returns the value of the Hypervisor Reset Management Register.
    \return               Hypervisor Reset Management Register
 */
__STATIC_FORCEINLINE uint32_t __get_HRMR(void)
{
  uint32_t result;
  __get_CP(15, 4, result, 12, 0, 2);
  return result;
}

/** \brief  Set HRMR
    This function assigns the given value to the Hypervisor Reset Management Register.
    \param [in]    hrmr  Hypervisor Reset Management Register value to set
 */
__STATIC_FORCEINLINE void __set_HRMR(uint32_t hrmr)
{
  __set_CP(15, 4, hrmr, 12, 0, 2);
}

#endif
