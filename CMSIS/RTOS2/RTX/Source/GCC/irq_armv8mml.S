/*
 * Copyright (c) 2016-2020 Arm Limited. All rights reserved.
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
 *
 * -----------------------------------------------------------------------------
 *
 * Project:     CMSIS-RTOS RTX
 * Title:       ARMv8M Mainline Exception handlers
 *
 * -----------------------------------------------------------------------------
 */


        .syntax  unified

        #ifdef   _RTE_
        #include "RTE_Components.h"
        #ifdef   RTE_CMSIS_RTOS2_RTX5_ARMV8M_NS
        #define  DOMAIN_NS    1
        #endif
        #endif

        #ifndef  DOMAIN_NS
        #define  DOMAIN_NS    0
        #endif

        #if (defined(__ARM_FP) && (__ARM_FP > 0))
        .equ     FPU_USED,    1
        #else
        .equ     FPU_USED,    0
        #endif

        #if (defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE > 0))
        .equ     MVE_USED,    1
        #else
        .equ     MVE_USED,    0
        #endif

        .equ     I_T_RUN_OFS, 20        // osRtxInfo.thread.run offset
        .equ     TCB_SM_OFS,  48        // TCB.stack_mem offset
        .equ     TCB_SP_OFS,  56        // TCB.SP offset
        .equ     TCB_SF_OFS,  34        // TCB.stack_frame offset
        .equ     TCB_TZM_OFS, 64        // TCB.tz_memory offset

        .section ".rodata"
        .global  irqRtxLib              // Non weak library reference
irqRtxLib:
        .byte    0


        .thumb
        .section ".text"
        .align   2


        .thumb_func
        .type    SVC_Handler, %function
        .global  SVC_Handler
        .fnstart
        .cantunwind
SVC_Handler:

        tst      lr,#0x04               // Determine return stack from EXC_RETURN bit 2
        ite      eq
        mrseq    r0,msp                 // Get MSP if return stack is MSP
        mrsne    r0,psp                 // Get PSP if return stack is PSP

        ldr      r1,[r0,#24]            // Load saved PC from stack
        ldrb     r1,[r1,#-2]            // Load SVC number
        cmp      r1,#0
        bne      SVC_User               // Branch if not SVC 0

        push     {r0,lr}                // Save SP and EXC_RETURN
        ldm      r0,{r0-r3,r12}         // Load function parameters and address from stack
        blx      r12                    // Call service function
        pop      {r12,lr}               // Restore SP and EXC_RETURN
        stm      r12,{r0-r1}            // Store function return values

SVC_Context:
        ldr      r3,=osRtxInfo+I_T_RUN_OFS // Load address of osRtxInfo.run
        ldm      r3,{r1,r2}             // Load osRtxInfo.thread.run: curr & next
        cmp      r1,r2                  // Check if thread switch is required
        it       eq
        bxeq     lr                     // Exit when threads are the same

        .if      (FPU_USED == 1) || (MVE_USED == 1)
        cbnz     r1,SVC_ContextSave     // Branch if running thread is not deleted
        tst      lr,#0x10               // Check if extended stack frame
        bne      SVC_ContextSwitch
        ldr      r1,=0xE000EF34         // FPCCR Address
        ldr      r0,[r1]                // Load FPCCR
        bic      r0,r0,#1               // Clear LSPACT (Lazy state)
        str      r0,[r1]                // Store FPCCR
        b        SVC_ContextSwitch
        .else
        cbz      r1,SVC_ContextSwitch   // Branch if running thread is deleted
        .endif

SVC_ContextSave:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r1,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      r0,SVC_ContextSave1    // Branch if there is no secure context
        push     {r1,r2,r3,lr}          // Save registers and EXC_RETURN
        BL       TZ_StoreContext_S      // Store secure context
        pop      {r1,r2,r3,lr}          // Restore registers and EXC_RETURN
        #endif

SVC_ContextSave1:
        mrs      r0,psp                 // Get PSP
        stmdb    r0!,{r4-r11}           // Save R4..R11
        .if      (FPU_USED == 1) || (MVE_USED == 1)
        tst      lr,#0x10               // Check if extended stack frame
        it       eq
        vstmdbeq r0!,{s16-s31}          //  Save VFP S16.S31
        .endif

SVC_ContextSave2:
        str      r0,[r1,#TCB_SP_OFS]    // Store SP
        strb     lr,[r1,#TCB_SF_OFS]    // Store stack frame information

SVC_ContextSwitch:
        str      r2,[r3]                // osRtxInfo.thread.run: curr = next

SVC_ContextRestore:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r2,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      r0,SVC_ContextRestore1 // Branch if there is no secure context
        push     {r2,r3}                // Save registers
        bl       TZ_LoadContext_S       // Load secure context
        pop      {r2,r3}                // Restore registers
        #endif

SVC_ContextRestore1:
        ldr      r0,[r2,#TCB_SM_OFS]    // Load stack memory base
        ldrb     r1,[r2,#TCB_SF_OFS]    // Load stack frame information
        msr      psplim,r0              // Set PSPLIM
        ldr      r0,[r2,#TCB_SP_OFS]    // Load SP
        orr      lr,r1,#0xFFFFFF00      // Set EXC_RETURN

        #if      (DOMAIN_NS == 1)
        tst      lr,#0x40               // Check domain of interrupted thread
        bne      SVC_ContextRestore2    // Branch if secure
        #endif

        .if      (FPU_USED == 1) || (MVE_USED == 1)
        tst      lr,#0x10               // Check if extended stack frame
        it       eq
        vldmiaeq r0!,{s16-s31}          //  Restore VFP S16..S31
        .endif
        ldmia    r0!,{r4-r11}           // Restore R4..R11

SVC_ContextRestore2:
        msr      psp,r0                 // Set PSP

SVC_Exit:
        bx       lr                     // Exit from handler

SVC_User:
        ldr      r2,=osRtxUserSVC       // Load address of SVC table
        ldr      r3,[r2]                // Load SVC maximum number
        cmp      r1,r3                  // Check SVC number range
        bhi      SVC_Exit               // Branch if out of range

        push     {r0,lr}                // Save SP and EXC_RETURN
        ldr      r12,[r2,r1,lsl #2]     // Load address of SVC function
        ldm      r0,{r0-r3}             // Load function parameters from stack
        blx      r12                    // Call service function
        pop      {r12,lr}               // Restore SP and EXC_RETURN
        str      r0,[r12]               // Store function return value

        bx       lr                     // Return from handler

        .fnend
        .size    SVC_Handler, .-SVC_Handler


        .thumb_func
        .type    PendSV_Handler, %function
        .global  PendSV_Handler
        .fnstart
        .cantunwind
PendSV_Handler:

        push     {r0,lr}                // Save EXC_RETURN
        bl       osRtxPendSV_Handler    // Call osRtxPendSV_Handler
        pop      {r0,lr}                // Restore EXC_RETURN
        b        Sys_Context

        .fnend
        .size    PendSV_Handler, .-PendSV_Handler


        .thumb_func
        .type    SysTick_Handler, %function
        .global  SysTick_Handler
        .fnstart
        .cantunwind
SysTick_Handler:

        push     {r0,lr}                // Save EXC_RETURN
        bl       osRtxTick_Handler      // Call osRtxTick_Handler
        pop      {r0,lr}                // Restore EXC_RETURN
        b        Sys_Context

        .fnend
        .size   SysTick_Handler, .-SysTick_Handler


        .thumb_func
        .type    Sys_Context, %function
        .global  Sys_Context
        .fnstart
        .cantunwind
Sys_Context:

        ldr      r3,=osRtxInfo+I_T_RUN_OFS // Load address of osRtxInfo.run
        ldm      r3,{r1,r2}             // Load osRtxInfo.thread.run: curr & next
        cmp      r1,r2                  // Check if thread switch is required
        it       eq
        bxeq     lr                     // Exit when threads are the same

Sys_ContextSave:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r1,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      R0,Sys_ContextSave1    // Branch if there is no secure context
        push     {r1,r2,r3,lr}          // Save registers and EXC_RETURN
        bl       TZ_StoreContext_S      // Store secure context
        pop      {r1,r2,r3,lr}          // Restore registers and EXC_RETURN

Sys_ContextSave1:
        tst      lr,#0x40               // Check domain of interrupted thread
        it       ne
        mrsne    r0,psp                 // Get PSP
        bne      Sys_ContextSave3       // Branch if secure
        #endif

Sys_ContextSave2:
        mrs      r0,psp                 // Get PSP
        stmdb    r0!,{r4-r11}           // Save R4..R11
        .if      (FPU_USED == 1) || (MVE_USED == 1)
        tst      lr,#0x10               // Check if extended stack frame
        it       eq
        vstmdbeq r0!,{s16-s31}          //  Save VFP S16.S31
        .endif

Sys_ContextSave3:
        str      r0,[r1,#TCB_SP_OFS]    // Store SP
        strb     lr,[r1,#TCB_SF_OFS]    // Store stack frame information

Sys_ContextSwitch:
        str      r2,[r3]                // osRtxInfo.run: curr = next

Sys_ContextRestore:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r2,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      r0,Sys_ContextRestore1 // Branch if there is no secure context
        push     {r2,r3}                // Save registers
        bl       TZ_LoadContext_S       // Load secure context
        pop      {r2,r3}                // Restore registers
        #endif

Sys_ContextRestore1:
        ldr      r0,[r2,#TCB_SM_OFS]    // Load stack memory base
        ldrb     r1,[r2,#TCB_SF_OFS]    // Load stack frame information
        msr      psplim,r0              // Set PSPLIM
        ldr      r0,[r2,#TCB_SP_OFS]    // Load SP
        orr      lr,r1,#0xFFFFFF00      // Set EXC_RETURN

        #if      (DOMAIN_NS == 1)
        tst      lr,#0x40               // Check domain of interrupted thread
        bne      Sys_ContextRestore2    // Branch if secure
        #endif

        .if      (FPU_USED == 1) || (MVE_USED == 1)
        tst      lr,#0x10               // Check if extended stack frame
        it       eq
        vldmiaeq r0!,{s16-s31}          //  Restore VFP S16..S31
        .endif
        ldmia    r0!,{r4-r11}           // Restore R4..R11

Sys_ContextRestore2:
        msr      psp,r0                 // Set PSP

Sys_ContextExit:
        bx       lr                     // Exit from handler

        .fnend
        .size    Sys_Context, .-Sys_Context


        .end
