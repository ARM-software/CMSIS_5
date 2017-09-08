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

//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------

// <h> Common Test Settings
// <o> Print Output Format <0=> Plain Text <1=> XML
// <i> Set the test results output format to plain text or XML
#ifndef PRINT_XML_REPORT
#define PRINT_XML_REPORT            0
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
// <q09> TC_CoreAFunc_FPSCR
// <q10> TC_CoreAFunc_CPSR
// <q11> TC_CoreAFunc_Mode
// <q12> TC_CoreAFunc_SP
// <q13> TC_CoreAFunc_SP_usr
// <q14> TC_CoreAFunc_FPEXC
// <q15> TC_COREAFUNC_ACTLR
// <q16> TC_COREAFUNC_CPACR
// <q17> TC_COREAFUNC_DFSR
// <q18> TC_COREAFUNC_IFSR
// <q19> TC_COREAFUNC_ISR
// <q20> TC_COREAFUNC_CBAR
// <q21> TC_COREAFUNC_TTBR0
// <q22> TC_COREAFUNC_DACR
// <q23> TC_COREAFUNC_SCTLR
// <q24> TC_COREAFUNC_ACTRL
// <q25> TC_COREAFUNC_MPIDR
// <q26> TC_COREAFUNC_VBAR
//
// <q27> TC_GENTIMER_CNTFRQ
// <q28> TC_GENTIMER_CNTP_TVAL
// <q29> TC_GENTIMER_CNTP_CTL
#define TC_COREINSTR_NOP_EN         1
#define TC_COREINSTR_REV_EN         1
#define TC_COREINSTR_REV16_EN       1
#define TC_COREINSTR_REVSH_EN       1
#define TC_COREINSTR_ROR_EN         1
#define TC_COREINSTR_RBIT_EN        1
#define TC_COREINSTR_CLZ_EN         1
#define TC_COREINSTR_SSAT_EN        1
#define TC_COREINSTR_USAT_EN        1

#define TC_COREAFUNC_IRQ            1
#define TC_COREAFUNC_FPSCR          1
#define TC_COREAFUNC_CPSR           1
#define TC_COREAFUNC_MODE           1
#define TC_COREAFUNC_SP             1
#define TC_COREAFUNC_SP_USR         1
#define TC_COREAFUNC_FPEXC          1
#define TC_COREAFUNC_ACTLR          1
#define TC_COREAFUNC_CPACR          1
#define TC_COREAFUNC_DFSR           1
#define TC_COREAFUNC_IFSR           1
#define TC_COREAFUNC_ISR            1
#define TC_COREAFUNC_CBAR           1
#define TC_COREAFUNC_TTBR0          1
#define TC_COREAFUNC_DACR           1
#define TC_COREAFUNC_SCTLR          1
#define TC_COREAFUNC_ACTRL          1
#define TC_COREAFUNC_MPIDR          1
#define TC_COREAFUNC_VBAR           1

#define TC_GENTIMER_CNTFRQ          1
#define TC_GENTIMER_CNTP_TVAL       1
#define TC_GENTIMER_CNTP_CTL        1
// </h>

#endif /* __CV_CONFIG_H */

