/*-----------------------------------------------------------------------------
 *      Name:         CV_Config.h 
 *      Purpose:      CV Config header
 *----------------------------------------------------------------------------
 *      Copyright (c) 2017 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/
#ifndef __CV_CONFIG_H
#define __CV_CONFIG_H

#include "RTE_Components.h"
#include CMSIS_device_header

#define RTE_CV_COREINSTR 1
#define RTE_CV_COREFUNC  1
#define RTE_CV_MPUFUNC   __MPU_PRESENT

#if ((defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) || \
    (defined (__ARM_ARCH_8M_BASE__ ) && (__ARM_ARCH_8M_BASE__ == 1))    )
#define ARM_ARCH_8M 1
#else
#define ARM_ARCH_8M 0
#endif

//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------

// <h> Common Test Settings
// <o> Print Output Format <0=> Plain Text <1=> XML
// <i> Set the test results output format to plain text or XML
#ifndef PRINT_XML_REPORT
#define PRINT_XML_REPORT            1
#endif
// <o> Buffer size for assertions results
// <i> Set the buffer size for assertions results buffer
#define BUFFER_ASSERTIONS           128U
// </h>

// <h> Disable Test Cases
// <i> Uncheck to disable an individual test case
// <q00> TC_CoreInstr_NOP
// <q01> TC_CoreInstr_REV
// <q02> TC_CoreInstr_REV16
// <q03> TC_CoreInstr_REVSH
// <q04> TC_CoreInstr_ROR
// <q05> TC_CoreInstr_RBIT
// <q06> TC_CoreInstr_CLZ
// <q07> TC_CoreInstr_Exclusives
// <q08> TC_CoreInstr_SSAT
// <q09> TC_CoreInstr_USAT
//
// <q10> TC_CoreFunc_EnDisIRQ
// <q11> TC_CoreFunc_IRQPrio
// <q12> TC_CoreFunc_EncDecIRQPrio
// <q13> TC_CoreFunc_IRQVect
// <q14> TC_CoreFunc_Control
// <q15> TC_CoreFunc_IPSR
// <q16> TC_CoreFunc_APSR
// <q17> TC_CoreFunc_PSP
// <q18> TC_CoreFunc_MSP
// <q19> TC_CoreFunc_PSPLIM
// <q20> TC_CoreFunc_PSPLIM_NS
// <q21> TC_CoreFunc_MSPLIM
// <q22> TC_CoreFunc_MSPLIM_NS
// <q23> TC_CoreFunc_PRIMASK
// <q24> TC_CoreFunc_FAULTMASK
// <q25> TC_CoreFunc_BASEPRI
// <q26> TC_CoreFunc_FPUType
// <q27> TC_CoreFunc_FPSCR
//
// <q28> TC_MPU_SetClear
// <q29> TC_MPU_Load
#define TC_COREINSTR_NOP_EN          1
#define TC_COREINSTR_REV_EN          1
#define TC_COREINSTR_REV16_EN        1
#define TC_COREINSTR_REVSH_EN        1
#define TC_COREINSTR_ROR_EN          1
#define TC_COREINSTR_RBIT_EN         1
#define TC_COREINSTR_CLZ_EN          1
#define TC_COREINSTR_EXCLUSIVES_EN   1
#define TC_COREINSTR_SSAT_EN         1
#define TC_COREINSTR_USAT_EN         1

#define TC_COREFUNC_ENDISIRQ_EN      1
#define TC_COREFUNC_IRQPRIO_EN       1
#define TC_COREFUNC_ENCDECIRQPRIO_EN 1
#define TC_COREFUNC_IRQVECT_EN       1
#define TC_COREFUNC_CONTROL_EN       1
#define TC_COREFUNC_IPSR_EN          1
#define TC_COREFUNC_APSR_EN          1
#define TC_COREFUNC_PSP_EN           1
#define TC_COREFUNC_MSP_EN           1

#define TC_COREFUNC_PSPLIM_EN        ARM_ARCH_8M
#define TC_COREFUNC_PSPLIM_NS_EN     ARM_ARCH_8M
#define TC_COREFUNC_MSPLIM_EN        ARM_ARCH_8M
#define TC_COREFUNC_MSPLIM_NS_EN     ARM_ARCH_8M
#define TC_COREFUNC_PRIMASK_EN       1
#define TC_COREFUNC_FAULTMASK_EN     1
#define TC_COREFUNC_BASEPRI_EN       1
#define TC_COREFUNC_FPUTYPE_EN       1
#define TC_COREFUNC_FPSCR_EN         1

#define TC_MPU_SETCLEAR_EN           1
#define TC_MPU_LOAD_EN               1
// </h>

#endif /* __CV_CONFIG_H */

