/*-----------------------------------------------------------------------------
 *      Name:         cmsis_cv.h 
 *      Purpose:      cmsis_cv header
 *----------------------------------------------------------------------------
 *      Copyright (c) 2017 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/
#ifndef __CMSIS_CV_H
#define __CMSIS_CV_H

#include <stdint.h>
#include "CV_Config.h"
    
/* Expansion macro used to create CMSIS Driver references */
#define EXPAND_SYMBOL(name, port) name##port
#define CREATE_SYMBOL(name, port) EXPAND_SYMBOL(name, port)
  
// Simulator counter
#ifndef HW_PRESENT
extern uint32_t SIM_CYCCNT;
#endif

// SVC interrupt callback
extern void (*TST_IRQHandler)(void);

// Test main function
extern void cmsis_cv (void);

// Test cases
#ifdef RTE_CV_COREINSTR
extern void TC_CoreInstr_NOP (void);
extern void TC_CoreInstr_REV (void);
extern void TC_CoreInstr_REV16 (void);
extern void TC_CoreInstr_REVSH (void);
extern void TC_CoreInstr_ROR (void);
extern void TC_CoreInstr_RBIT (void);
extern void TC_CoreInstr_CLZ (void);
extern void TC_CoreInstr_SSAT (void);
extern void TC_CoreInstr_USAT (void);
#endif

#ifdef RTE_CV_COREFUNC
  #if defined(__CORTEX_M)
    extern void TC_CoreFunc_EnDisIRQ (void);
    extern void TC_CoreFunc_Control (void);
    extern void TC_CoreFunc_IPSR (void);
    extern void TC_CoreFunc_APSR (void);
    extern void TC_CoreFunc_PSP (void);
    extern void TC_CoreFunc_MSP (void);
    extern void TC_CoreFunc_PRIMASK (void);

    #if ((defined (__ARM_ARCH_7M__      ) && (__ARM_ARCH_7M__      == 1)) || \
       (defined (__ARM_ARCH_7EM__     ) && (__ARM_ARCH_7EM__     == 1)) || \
       (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1))    )

    extern void TC_CoreFunc_FAULTMASK (void);
    extern void TC_CoreFunc_BASEPRI (void);

    #endif

    #if ((defined (__ARM_ARCH_7EM__     ) && (__ARM_ARCH_7EM__     == 1)) || \
       (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1))    )

    extern void TC_CoreFunc_FPSCR (void);

    #endif
  #elif defined(__CORTEX_A)
    extern void TC_CoreAFunc_IRQ (void);
    extern void TC_CoreAFunc_FPSCR (void);
    extern void TC_CoreAFunc_CPSR (void);
    extern void TC_CoreAFunc_Mode (void);
    extern void TC_CoreAFunc_SP (void);
    extern void TC_CoreAFunc_SP_usr (void);
    extern void TC_CoreAFunc_FPEXC (void);
    extern void TC_CoreAFunc_ACTLR (void);
    extern void TC_CoreAFunc_CPACR (void);
    extern void TC_CoreAFunc_DFSR (void);
    extern void TC_CoreAFunc_IFSR (void);
    extern void TC_CoreAFunc_ISR (void);
    extern void TC_CoreAFunc_CBAR (void);
    extern void TC_CoreAFunc_TTBR0 (void);
    extern void TC_CoreAFunc_DACR (void);
    extern void TC_CoreAFunc_SCTLR (void);
    extern void TC_CoreAFunc_ACTRL (void);
    extern void TC_CoreAFunc_MPIDR (void);
    extern void TC_CoreAFunc_VBAR (void);
  #endif
#endif

#ifdef RTE_CV_MPUFUNC
#if defined(__MPU_PRESENT) && __MPU_PRESENT
extern void TC_MPU_SetClear (void);
extern void TC_MPU_Load (void);
#endif
#endif

#ifdef RTE_CV_GENTIMER
extern void TC_GenTimer_CNTFRQ (void);
extern void TC_GenTimer_CNTP_TVAL (void);
extern void TC_GenTimer_CNTP_CTL (void);
#endif

#endif /* __CMSIS_CV_H */
