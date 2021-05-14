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
 * Title:       ARMv8M Baseline Exception handlers
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

        mov      r0,lr
        lsrs     r0,r0,#3               // Determine return stack from EXC_RETURN bit 2
        bcc      SVC_MSP                // Branch if return stack is MSP
        mrs      r0,psp                 // Get PSP

SVC_Number:
        ldr      r1,[r0,#24]            // Load saved PC from stack
        subs     r1,r1,#2               // Point to SVC instruction
        ldrb     r1,[r1]                // Load SVC number
        cmp      r1,#0
        bne      SVC_User               // Branch if not SVC 0

        push     {r0,lr}                // Save SP and EXC_RETURN
        ldm      r0,{r0-r3}             // Load function parameters from stack
        blx      r7                     // Call service function
        pop      {r2,r3}                // Restore SP and EXC_RETURN
        stmia    r2!,{r0-r1}            // Store function return values
        mov      lr,r3                  // Set EXC_RETURN

SVC_Context:
        ldr      r3,=osRtxInfo+I_T_RUN_OFS // Load address of osRtxInfo.run
        ldmia    r3!,{r1,r2}            // Load osRtxInfo.thread.run: curr & next
        cmp      r1,r2                  // Check if thread switch is required
        beq      SVC_Exit               // Branch when threads are the same

        cbz      r1,SVC_ContextSwitch   // Branch if running thread is deleted

SVC_ContextSave:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r1,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      r0,SVC_ContextSave1    // Branch if there is no secure context
        push     {r1,r2,r3,r7}          // Save registers
        mov      r7,lr                  // Get EXC_RETURN
        bl       TZ_StoreContext_S      // Store secure context
        mov      lr,r7                  // Set EXC_RETURN
        pop      {r1,r2,r3,r7}          // Restore registers
        #endif

SVC_ContextSave1:
        mrs      r0,psp                 // Get PSP
        subs     r0,r0,#32              // Calculate SP
        str      r0,[r1,#TCB_SP_OFS]    // Store SP
        stmia    r0!,{r4-r7}            // Save R4..R7
        mov      r4,r8
        mov      r5,r9
        mov      r6,r10
        mov      r7,r11
        stmia    r0!,{r4-r7}            // Save R8..R11

SVC_ContextSave2:
        mov      r0,lr                  // Get EXC_RETURN
        adds     r1,r1,#TCB_SF_OFS      // Adjust address
        strb     r0,[r1]                // Store stack frame information

SVC_ContextSwitch:
        subs     r3,r3,#8               // Adjust address
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
        mov      r1,r2
        adds     r1,r1,#TCB_SF_OFS      // Adjust address
        ldrb     r0,[r1]                // Load stack frame information
        movs     r1,#0xFF
        mvns     r1,r1                  // R1=0xFFFFFF00
        orrs     r0,r1
        mov      lr,r0                  // Set EXC_RETURN

        #if      (DOMAIN_NS == 1)
        lsls     r0,r0,#25              // Check domain of interrupted thread
        bpl      SVC_ContextRestore2    // Branch if non-secure
        ldr      r0,[r2,#TCB_SP_OFS]    // Load SP
        msr      psp,r0                 // Set PSP
        bx       lr                     // Exit from handler
        #else
        ldr      r0,[r2,#TCB_SM_OFS]    // Load stack memory base
        msr      psplim,r0              // Set PSPLIM
        #endif

SVC_ContextRestore2:
        ldr      r0,[r2,#TCB_SP_OFS]    // Load SP
        adds     r0,r0,#16              // Adjust address
        ldmia    r0!,{r4-r7}            // Restore R8..R11
        mov      r8,r4
        mov      r9,r5
        mov      r10,r6
        mov      r11,r7
        msr      psp,r0                 // Set PSP
        subs     r0,r0,#32              // Adjust address
        ldmia    r0!,{r4-r7}            // Restore R4..R7

SVC_Exit:
        bx       lr                     // Exit from handler

SVC_MSP:
        mrs      r0,msp                 // Get MSP
        b        SVC_Number

SVC_User:
        ldr      r2,=osRtxUserSVC       // Load address of SVC table
        ldr      r3,[r2]                // Load SVC maximum number
        cmp      r1,r3                  // Check SVC number range
        bhi      SVC_Exit               // Branch if out of range

        push     {r0,lr}                // Save SP and EXC_RETURN
        lsls     r1,r1,#2
        ldr      r3,[r2,r1]             // Load address of SVC function
        mov      r12,r3
        ldmia    r0,{r0-r3}             // Load function parameters from stack
        blx      r12                    // Call service function
        pop      {r2,r3}                // Restore SP and EXC_RETURN
        str      r0,[r2]                // Store function return value
        mov      lr,r3                  // Set EXC_RETURN

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
        pop      {r0,r1}                // Restore EXC_RETURN
        mov      lr,r1                  // Set EXC_RETURN
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
        pop      {r0,r1}                // Restore EXC_RETURN
        mov      lr,r1                  // Set EXC_RETURN
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
        ldm      r3!,{r1,r2}            // Load osRtxInfo.thread.run: curr & next
        cmp      r1,r2                  // Check if thread switch is required
        beq      Sys_ContextExit        // Branch when threads are the same

Sys_ContextSave:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r1,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      r0,Sys_ContextSave1    // Branch if there is no secure context
        push     {r1,r2,r3,r7}          // Save registers
        mov      r7,lr                  // Get EXC_RETURN
        bl       TZ_StoreContext_S      // Store secure context
        mov      lr,r7                  // Set EXC_RETURN
        pop      {r1,r2,r3,r7}          // Restore registers

Sys_ContextSave1:
        mov      r0,lr                  // Get EXC_RETURN
        lsls     r0,r0,#25              // Check domain of interrupted thread
        bpl      Sys_ContextSave2       // Branch if non-secure
        mrs      r0,psp                 // Get PSP
        str      r0,[r1,#TCB_SP_OFS]    // Store SP
        b        Sys_ContextSave3
        #endif

Sys_ContextSave2:
        mrs      r0,psp                 // Get PSP
        subs     r0,r0,#32              // Adjust address
        str      r0,[r1,#TCB_SP_OFS]    // Store SP
        stmia    r0!,{r4-r7}            // Save R4..R7
        mov      r4,r8
        mov      r5,r9
        mov      r6,r10
        mov      r7,r11
        stmia    r0!,{r4-r7}            // Save R8..R11

Sys_ContextSave3:
        mov      r0,lr                  // Get EXC_RETURN
        adds     r1,r1,#TCB_SF_OFS      // Adjust address
        strb     r0,[r1]                // Store stack frame information

Sys_ContextSwitch:
        subs     r3,r3,#8               // Adjust address
        str      r2,[r3]                // osRtxInfo.run: curr = next

Sys_ContextRestore:
        #if      (DOMAIN_NS == 1)
        ldr      r0,[r2,#TCB_TZM_OFS]   // Load TrustZone memory identifier
        cbz      r0,Sys_ContextRestore1 // Branch if there is no secure context
        push     {r2,r3}                // Save registers
        BL       TZ_LoadContext_S       // Load secure context
        pop      {r2,r3}                // Restore registers
        #endif

Sys_ContextRestore1:
        mov      r1,r2
        adds     r1,r1,#TCB_SF_OFS      // Adjust offset
        ldrb     r0,[r1]                // Load stack frame information
        movs     r1,#0xFF
        mvns     r1,r1                  // R1=0xFFFFFF00
        orrs     r0,r1
        mov      lr,r0                  // Set EXC_RETURN

        #if      (DOMAIN_NS == 1)
        lsls     r0,r0,#25              // Check domain of interrupted thread
        bpl      Sys_ContextRestore2    // Branch if non-secure
        ldr      r0,[r2,#TCB_SP_OFS]    // Load SP
        msr      psp,r0                 // Set PSP
        bx       lr                     // Exit from handler
        #else
        ldr      r0,[r2,#TCB_SM_OFS]    // Load stack memory base
        msr      psplim,r0              // Set PSPLIM
        #endif

Sys_ContextRestore2:
        ldr      r0,[r2,#TCB_SP_OFS]    // Load SP
        adds     r0,r0,#16              // Adjust address
        ldmia    r0!,{r4-r7}            // Restore R8..R11
        mov      r8,r4
        mov      r9,r5
        mov      r10,r6
        mov      r11,r7
        msr      psp,r0                 // Set PSP
        subs     r0,r0,#32              // Adjust address
        ldmia    r0!,{r4-r7}            // Restore R4..R7

Sys_ContextExit:
        bx       lr                     // Exit from handler

        .fnend
        .size    Sys_Context, .-Sys_Context


        .end
