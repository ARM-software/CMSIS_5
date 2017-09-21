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
// <q07> TC_CoreInstr_SSAT
// <q08> TC_CoreInstr_USAT
//
// <q09> TC_CoreFunc_EnDisIRQ
// <q10> TC_CoreFunc_Control
// <q11> TC_CoreFunc_IPSR
// <q12> TC_CoreFunc_APSR
// <q13> TC_CoreFunc_PSP
// <q14> TC_CoreFunc_MSP
// <q15> TC_CoreFunc_PRIMASK
// <q16> TC_CoreFunc_FAULTMASK
// <q17> TC_CoreFunc_BASEPRI
// <q18> TC_CoreFunc_FPSCR
//
// <q19> TC_MPU_SetClear
// <q20> TC_MPU_Load
#define TC_COREINSTR_NOP_EN         1
#define TC_COREINSTR_REV_EN         1
#define TC_COREINSTR_REV16_EN       1
#define TC_COREINSTR_REVSH_EN       1
#define TC_COREINSTR_ROR_EN         1
#define TC_COREINSTR_RBIT_EN        1
#define TC_COREINSTR_CLZ_EN         1
#define TC_COREINSTR_SSAT_EN        1
#define TC_COREINSTR_USAT_EN        1

#define TC_COREFUNC_ENDISIRQ_EN     1
#define TC_COREFUNC_CONTROL_EN      1
#define TC_COREFUNC_IPSR_EN         1
#define TC_COREFUNC_APSR_EN         1
#define TC_COREFUNC_PSP_EN          1
#define TC_COREFUNC_MSP_EN          1
#define TC_COREFUNC_PRIMASK_EN      1
#define TC_COREFUNC_FAULTMASK_EN    1
#define TC_COREFUNC_BASEPRI_EN      1
#define TC_COREFUNC_FPSCR_EN        1

#define TC_MPU_SETCLEAR_EN          1
#define TC_MPU_LOAD_EN              1
// </h>

#endif /* __CV_CONFIG_H */

