;/*
; * Copyright (c) 2013-2017 ARM Limited. All rights reserved.
; *
; * SPDX-License-Identifier: Apache-2.0
; *
; * Licensed under the Apache License, Version 2.0 (the License); you may
; * not use this file except in compliance with the License.
; * You may obtain a copy of the License at
; *
; * www.apache.org/licenses/LICENSE-2.0
; *
; * Unless required by applicable law or agreed to in writing, software
; * distributed under the License is distributed on an AS IS BASIS, WITHOUT
; * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; * See the License for the specific language governing permissions and
; * limitations under the License.
; *
; * -----------------------------------------------------------------------------
; *
; * Project:     CMSIS-RTOS RTX
; * Title:       Cortex-M0 Exception handlers
; *
; * -----------------------------------------------------------------------------
; */


                NAME    irq_cm0.s


I_T_RUN_OFS     EQU      20                     ; osRtxInfo.thread.run offset
TCB_SP_OFS      EQU      56                     ; TCB.SP offset


                PRESERVE8
                SECTION .rodata:DATA:NOROOT(2)


                EXPORT   irqRtxLib
irqRtxLib       DCB      0                      ; Non weak library reference


                THUMB
                SECTION .text:CODE:NOROOT(2)


SVC_Handler     
                EXPORT   SVC_Handler
                IMPORT   osRtxUserSVC
                IMPORT   osRtxInfo

                MRS      R0,PSP                 ; Get PSP
                LDR      R1,[R0,#24]            ; Load saved PC from stack
                SUBS     R1,R1,#2               ; Point to SVC instruction
                LDRB     R1,[R1]                ; Load SVC number
                CMP      R1,#0
                BNE      SVC_User               ; Branch if not SVC 0

                PUSH     {R0,LR}                ; Save PSP and EXC_RETURN
                LDMIA    R0,{R0-R3}             ; Load function parameters from stack
                BLX      R7                     ; Call service function
                POP      {R2,R3}                ; Restore PSP and EXC_RETURN
                STMIA    R2!,{R0-R1}            ; Store function return values
                MOV      LR,R3                  ; Set EXC_RETURN

SVC_Context
                LDR      R3,=osRtxInfo+I_T_RUN_OFS; Load address of osRtxInfo.run
                LDMIA    R3!,{R1,R2}            ; Load osRtxInfo.thread.run: curr & next
                CMP      R1,R2                  ; Check if thread switch is required
                BEQ      SVC_Exit               ; Branch when threads are the same

                CMP      R1,#0
                BEQ      SVC_ContextSwitch      ; Branch if running thread is deleted

SVC_ContextSave
                MRS      R0,PSP                 ; Get PSP
                SUBS     R0,R0,#32              ; Adjust address
                STR      R0,[R1,#TCB_SP_OFS]    ; Store SP
                STMIA    R0!,{R4-R7}            ; Save R4..R7
                MOV      R4,R8
                MOV      R5,R9
                MOV      R6,R10
                MOV      R7,R11
                STMIA    R0!,{R4-R7}            ; Save R8..R11

SVC_ContextSwitch
                SUBS     R3,R3,#8
                STR      R2,[R3]                ; osRtxInfo.thread.run: curr = next

SVC_ContextRestore
                LDR      R0,[R2,#TCB_SP_OFS]    ; Load SP
                ADDS     R0,R0,#16              ; Adjust address
                LDMIA    R0!,{R4-R7}            ; Restore R8..R11
                MOV      R8,R4
                MOV      R9,R5
                MOV      R10,R6
                MOV      R11,R7
                MSR      PSP,R0                 ; Set PSP
                SUBS     R0,R0,#32              ; Adjust address
                LDMIA    R0!,{R4-R7}            ; Restore R4..R7

                MOVS     R0,#~0xFFFFFFFD
                MVNS     R0,R0                  ; Set EXC_RETURN value
                BX       R0                     ; Exit from handler

SVC_Exit
                BX       LR                     ; Exit from handler

SVC_User
                PUSH     {R4,LR}                ; Save registers
                LDR      R2,=osRtxUserSVC       ; Load address of SVC table
                LDR      R3,[R2]                ; Load SVC maximum number
                CMP      R1,R3                  ; Check SVC number range
                BHI      SVC_Done               ; Branch if out of range

                LSLS     R1,R1,#2
                LDR      R4,[R2,R1]             ; Load address of SVC function

                LDMIA    R0,{R0-R3}             ; Load function parameters from stack
                BLX      R4                     ; Call service function
                MRS      R4,PSP                 ; Get PSP
                STMIA    R4!,{R0-R3}            ; Store function return values

SVC_Done
                POP      {R4,PC}                ; Return from handler


PendSV_Handler  
                EXPORT   PendSV_Handler
                IMPORT   osRtxPendSV_Handler

                PUSH     {R0,LR}                ; Save EXC_RETURN
                BL       osRtxPendSV_Handler    ; Call osRtxPendSV_Handler
                POP      {R0,R1}                ; Restore EXC_RETURN
                MOV      LR,R1                  ; Set EXC_RETURN
                B        SVC_Context


SysTick_Handler 
                EXPORT   SysTick_Handler
                IMPORT   osRtxTick_Handler

                PUSH     {R0,LR}                ; Save EXC_RETURN
                BL       osRtxTick_Handler      ; Call osRtxTick_Handler
                POP      {R0,R1}                ; Restore EXC_RETURN
                MOV      LR,R1                  ; Set EXC_RETURN
                B        SVC_Context


                END
