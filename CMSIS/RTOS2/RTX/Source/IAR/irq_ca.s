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
; * Title:       Cortex-A Exception handlers (using GIC)
; *
; * -----------------------------------------------------------------------------
; */

                NAME     irq_ca.s

ICDABR0_OFFSET  EQU      0x00000300                 ; GICD: Active Bit Register 0 offset
ICDIPR0_OFFSET  EQU      0x00000400                 ; GICD: Interrupt Priority Register 0 offset
ICCIAR_OFFSET   EQU      0x0000000C                 ; GICI: Interrupt Acknowledge Register offset
ICCEOIR_OFFSET  EQU      0x00000010                 ; GICI: End of Interrupt Register offset
ICCHPIR_OFFSET  EQU      0x00000018                 ; GICI: Highest Pending Interrupt Register offset

MODE_FIQ        EQU      0x11
MODE_IRQ        EQU      0x12
MODE_SVC        EQU      0x13
MODE_ABT        EQU      0x17
MODE_UND        EQU      0x1B

CPSR_BIT_T      EQU      0x20

I_T_RUN_OFS     EQU      28                         ; osRtxInfo.thread.run offset
TCB_SP_FRAME    EQU      34                         ; osRtxThread_t.stack_frame offset
TCB_SP_OFS      EQU      56                         ; osRtxThread_t.sp offset


                PRESERVE8
                ARM


                SECTION .rodata:DATA:NOROOT(2)
                EXPORT   irqRtxLib
irqRtxLib       DCB      0                          ; Non weak library reference


                SECTION .data:DATA:NOROOT(2)
ID0_Active      DCB      4                          ; Flag used to workaround GIC 390 errata 733075


                SECTION .text:CODE:NOROOT(2)


Undef_Handler
                EXPORT  Undef_Handler
                IMPORT  CUndefHandler

                SRSFD   SP!, #MODE_UND
                PUSH    {R0-R4, R12}                ; Save APCS corruptible registers to UND mode stack

                MRS     R0, SPSR
                TST     R0, #CPSR_BIT_T             ; Check mode
                MOVEQ   R1, #4                      ; R1 = 4 ARM mode
                MOVNE   R1, #2                      ; R1 = 2 Thumb mode
                SUB     R0, LR, R1
                LDREQ   R0, [R0]                    ; ARM mode - R0 points to offending instruction
                BEQ     Undef_Cont

                ; Thumb instruction
                ; Determine if it is a 32-bit Thumb instruction
                LDRH    R0, [R0]
                MOV     R2, #0x1C
                CMP     R2, R0, LSR #11
                BHS     Undef_Cont                  ; 16-bit Thumb instruction

                ; 32-bit Thumb instruction. Unaligned - reconstruct the offending instruction
                LDRH    R2, [LR]
                ORR     R0, R2, R0, LSL #16
Undef_Cont
                MOV     R2, LR                      ; Set LR to third argument

                AND     R12, SP, #4                 ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R12                 ; Adjust stack
                PUSH    {R12, LR}                   ; Store stack adjustment and dummy LR

                ; R0 =Offending instruction, R1 =2(Thumb) or =4(ARM)
                BL      CUndefHandler

                POP     {R12, LR}                   ; Get stack adjustment & discard dummy LR
                ADD     SP, SP, R12                 ; Unadjust stack

                LDR     LR, [SP, #24]               ; Restore stacked LR and possibly adjust for retry
                SUB     LR, LR, R0
                LDR     R0, [SP, #28]               ; Restore stacked SPSR
                MSR     SPSR_CXSF, R0
                POP     {R0-R4, R12}                ; Restore stacked APCS registers
                ADD     SP, SP, #8                  ; Adjust SP for already-restored banked registers
                MOVS    PC, LR


PAbt_Handler
                EXPORT  PAbt_Handler
                IMPORT  CPAbtHandler

                SUB     LR, LR, #4                  ; Pre-adjust LR
                SRSFD   SP!, #MODE_ABT              ; Save LR and SPRS to ABT mode stack
                PUSH    {R0-R4, R12}                ; Save APCS corruptible registers to ABT mode stack
                MRC     p15, 0, R0, c5, c0, 1       ; IFSR
                MRC     p15, 0, R1, c6, c0, 2       ; IFAR

                MOV     R2, LR                      ; Set LR to third argument

                AND     R12, SP, #4                 ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R12                 ; Adjust stack
                PUSH    {R12, LR}                   ; Store stack adjustment and dummy LR

                BL      CPAbtHandler

                POP     {R12, LR}                   ; Get stack adjustment & discard dummy LR
                ADD     SP, SP, R12                 ; Unadjust stack

                POP     {R0-R4, R12}                ; Restore stack APCS registers
                RFEFD   SP!                         ; Return from exception


DAbt_Handler
                EXPORT  DAbt_Handler
                IMPORT  CDAbtHandler

                SUB     LR, LR, #8                  ; Pre-adjust LR
                SRSFD   SP!, #MODE_ABT              ; Save LR and SPRS to ABT mode stack
                PUSH    {R0-R4, R12}                ; Save APCS corruptible registers to ABT mode stack
                CLREX                               ; State of exclusive monitors unknown after taken data abort
                MRC     p15, 0, R0, c5, c0, 0       ; DFSR
                MRC     p15, 0, R1, c6, c0, 0       ; DFAR

                MOV     R2, LR                      ; Set LR to third argument

                AND     R12, SP, #4                 ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R12                 ; Adjust stack
                PUSH    {R12, LR}                   ; Store stack adjustment and dummy LR

                BL      CDAbtHandler

                POP     {R12, LR}                   ; Get stack adjustment & discard dummy LR
                ADD     SP, SP, R12                 ; Unadjust stack

                POP     {R0-R4, R12}                ; Restore stacked APCS registers
                RFEFD   SP!                         ; Return from exception


IRQ_Handler
                EXPORT  IRQ_Handler
                IMPORT  IRQTable
                IMPORT  IRQCount
                IMPORT  osRtxIrqHandler
                IMPORT  irqRtxGicBase

                SUB     LR, LR, #4                  ; Pre-adjust LR
                SRSFD   SP!, #MODE_IRQ              ; Save LR_irq and SPRS_irq
                PUSH    {R0-R3, R12, LR}            ; Save APCS corruptible registers

                ; Identify and acknowledge interrupt
                LDR     R1, =irqRtxGicBase;
                LDR     R1, [R1, #4]
                LDR     R0, [R1, #ICCHPIR_OFFSET]   ; Dummy Read GICI ICCHPIR to avoid GIC 390 errata 801120
                LDR     R0, [R1, #ICCIAR_OFFSET]    ; Read GICI ICCIAR
                DSB                                 ; Ensure that interrupt acknowledge completes before re-enabling interrupts

                ; Workaround GIC 390 errata 733075 - see GIC-390_Errata_Notice_v6.pdf dated 09-Jul-2014
                ; The following workaround code is for a single-core system.  It would be different in a multi-core system.
                ; If the ID is 0 or 0x3FE or 0x3FF, then the GIC CPU interface may be locked-up so unlock it, otherwise service the interrupt as normal
                ; Special IDs 1020=0x3FC and 1021=0x3FD are reserved values in GICv1 and GICv2 so will not occur here
                CMP     R0, #0
                BEQ     IRQ_Unlock
                MOV     R2, #0x3FE
                CMP     R0, R2
                BLT     IRQ_Normal
IRQ_Unlock
                ; Unlock the CPU interface with a dummy write to ICDIPR0
                LDR     R2, =irqRtxGicBase
                LDR     R2, [R2]
                LDR     R3, [R2, #ICDIPR0_OFFSET]
                STR     R3, [R2, #ICDIPR0_OFFSET]
                DSB                                 ; Ensure the write completes before continuing

                ; If the ID is 0 and it is active and has not been seen before, then service it as normal,
                ; otherwise the interrupt should be treated as spurious and not serviced.
                CMP     R0, #0
                BNE     IRQ_Exit                    ; Not 0, so spurious
                LDR     R3, [R2, #ICDABR0_OFFSET]   ; Get the interrupt state
                TST     R3, #1
                BEQ     IRQ_Exit                    ; Not active, so spurious
                LDR     R2, =ID0_Active
                LDRB    R3, [R2]
                CMP     R3, #1
                BEQ     IRQ_Exit                    ; Seen it before, so spurious

                ; Record that ID0 has now been seen, then service it as normal
                MOV     R3, #1
                STRB    R3, [R2]
                ; End of Workaround GIC 390 errata 733075

IRQ_Normal
                LDR     R2, =IRQCount               ; Read number of entries in IRQ handler table
                LDR     R2, [R2]
                CMP     R0, R2                      ; Check if IRQ ID is within range
                MOV     R2, #0
                BHS     IRQ_End                     ; Out of range, return as normal
                LDR     R2, =IRQTable               ; Read IRQ handler address from IRQ table
                LDR     R2, [R2, R0, LSL #2]
                CMP     R2, #0                      ; Check if handler address is 0
                BEQ     IRQ_End                     ; If 0, end interrupt and return
                PUSH    {R0, R1}                    ; Store IRQ ID and GIC CPU Interface base address

                CPS     #MODE_SVC                   ; Change to SVC mode

                MOV     R3, SP                      ; Move SP into R3
                AND     R3, R3, #4                  ; Get stack adjustment to ensure 8-byte alignment
                SUB     SP, SP, R3                  ; Adjust stack
                PUSH    {R2, R3, R12, LR}           ; Store handler address(R2), stack adjustment(R3) and user R12, LR

                CPSIE   i                           ; Re-enable interrupts
                BLX     R2                          ; Call IRQ handler
                CPSID   i                           ; Disable interrupts

                POP     {R2, R3, R12, LR}           ; Restore handler address(R2), stack adjustment(R3) and user R12, LR
                ADD     SP, SP, R3                  ; Unadjust stack

                CPS     #MODE_IRQ                   ; Change to IRQ mode
                POP     {R0, R1}                    ; Restore IRQ ID and GIC CPU Interface base address
                DSB                                 ; Ensure that interrupt source is cleared before signalling End Of Interrupt
IRQ_End
                ; R0 =IRQ ID, R1 =GICI_BASE
                ; EOI does not need to be written for IDs 1020 to 1023 (0x3FC to 0x3FF)
                STR     R0, [R1, #ICCEOIR_OFFSET]   ; Normal end-of-interrupt write to EOIR (GIC CPU Interface register) to clear the active bit

                ; If it was ID0, clear the seen flag, otherwise return as normal
                CMP     R0, #0
                LDREQ   R1, =ID0_Active
                STRBEQ  R0, [R1]                    ; Clear the seen flag, using R0 (which is 0), to save loading another register

                LDR     R3, =osRtxIrqHandler        ; Load osRtxIrqHandler function address
                CMP     R2, R3                      ; If is the same ass current IRQ handler
                BEQ     osRtxContextSwitch          ; Call context switcher

IRQ_Exit
                POP     {R0-R3, R12, LR}            ; Restore stacked APCS registers
                RFEFD   SP!                         ; Return from IRQ handler
                

SVC_Handler
                EXPORT  SVC_Handler
                IMPORT  osRtxIrqLock
                IMPORT  osRtxIrqUnlock
                IMPORT  osRtxUserSVC
                IMPORT  osRtxInfo

                SRSFD   SP!, #MODE_SVC              ; Store SPSR_svc and LR_svc onto SVC stack
                PUSH    {R12, LR}

                MRS     R12, SPSR                   ; Load SPSR
                TST     R12, #CPSR_BIT_T            ; Thumb bit set?
                LDRHNE  R12, [LR,#-2]               ; Thumb: load halfword
                BICNE   R12, R12, #0xFF00           ;        extract SVC number
                LDREQ   R12, [LR,#-4]               ; ARM:   load word
                BICEQ   R12, R12, #0xFF000000       ;        extract SVC number
                CMP     R12, #0                     ; Compare SVC number
                BNE     SVC_User                    ; Branch if User SVC

                PUSH    {R0-R3}
                BLX     osRtxIrqLock                ; Disable RTX interrupt (timer, PendSV)
                POP     {R0-R3}

                LDR     R12, [SP]                   ; Reload R12 from stack

                CPSIE   i                           ; Re-enable interrupts
                BLX     R12                         ; Branch to SVC function
                CPSID   i                           ; Disable interrupts

                SUB     SP, SP, #4                  ; Adjust SP
                STM     SP, {SP}^                   ; Store SP_usr onto stack
                POP     {R12}                       ; Pop SP_usr into R12
                SUB     R12, R12, #16               ; Adjust pointer to SP_usr
                LDMDB   R12, {R2,R3}                ; Load return values from SVC function
                PUSH    {R0-R3}                     ; Push return values to stack

                BLX     osRtxIrqUnlock              ; Enable RTX interrupt (timer, PendSV)
                B       osRtxContextSwitch          ; Continue in context switcher

SVC_User
                PUSH    {R4, R5}
                LDR     R5,=osRtxUserSVC            ; Load address of SVC table
                LDR     R4,[R5]                     ; Load SVC maximum number
                CMP     R12,R4                      ; Check SVC number range
                BHI     SVC_Done                    ; Branch if out of range

                LDR     R12,[R5,R12,LSL #2]         ; Load SVC Function Address
                BLX     R12                         ; Call SVC Function

SVC_Done
                POP     {R4, R5, R12, LR}
                RFEFD   SP!                         ; Return from exception


osRtxContextSwitch
                EXPORT  osRtxContextSwitch

                LDR     R12, =osRtxInfo+I_T_RUN_OFS ; Load address of osRtxInfo.run
                LDM     R12, {R0, R1}               ; Load osRtxInfo.thread.run: curr & next
                CMP     R0, R1                      ; Check if context switch is required
                BEQ     osRtxContextExit            ; Exit if curr and next are equal

                CMP     R0, #0                      ; Is osRtxInfo.thread.run.curr == 0
                ADDEQ   SP, SP, #32                 ; Equal, curr deleted, adjust current SP
                BEQ     osRtxContextRestore         ; Restore context, run.curr = run.next;

osRtxContextSave
                SUB     SP, SP, #4
                STM     SP, {SP}^                   ; Save SP_usr to current stack
                POP     {R3}                        ; Pop SP_usr into R3

                SUB     R3, R3, #64                 ; Adjust user sp to end of basic frame (R4)
                STMIA   R3!, {R4-R11}               ; Save R4-R11 to user
                POP     {R4-R8}                     ; Pop current R0-R12 into R4-R8
                STMIA   R3!, {R4-R8}                ; Store them to user stack
                STM     R3, {LR}^                   ; Store LR_usr directly
                ADD     R3, R3, #4                  ; Adjust user sp to PC
                POP     {R4-R6}                     ; Pop current LR, PC, CPSR
                STMIA   R3!, {R5-R6}                ; Restore user PC and CPSR

                SUB     R3, R3, #64                 ; Adjust user sp to R4

                ; Check if VFP state need to be saved
                MRC     p15, 0, R2, c1, c0, 2       ; VFP/NEON access enabled? (CPACR)
                AND     R2, R2, #0x00F00000
                CMP     R2, #0x00F00000
                BNE     osRtxContextSave1           ; Continue, no VFP

                VMRS    R2, FPSCR
                STMDB   R3!, {R2,R12}               ; Push FPSCR, maintain 8-byte alignment
                IF {TARGET_FEATURE_EXTENSION_REGISTER_COUNT} == 16
                VSTMDB  R3!, {D0-D15}
                LDRB    R2, [R0, #TCB_SP_FRAME]     ; Record in TCB that VFP/D16 state is stacked
                ORR     R2, R2, #2
                STRB    R2, [R0, #TCB_SP_FRAME]
                ENDIF
                IF {TARGET_FEATURE_EXTENSION_REGISTER_COUNT} == 32
                VSTMDB  R3!, {D0-D15}
                VSTMDB  R3!, {D16-D31}
                LDRB    R2, [R0, #TCB_SP_FRAME]     ; Record in TCB that NEON/D32 state is stacked
                ORR     R2, R2, #4
                STRB    R2, [R0, #TCB_SP_FRAME]
                ENDIF

osRtxContextSave1
                STR     R3, [R0, #TCB_SP_OFS]       ; Store user sp to osRtxInfo.thread.run.curr

osRtxContextRestore
                STR     R1, [R12]                   ; Store run.next to run.curr
                LDR     R3, [R1, #TCB_SP_OFS]       ; Load next osRtxThread_t.sp
                LDRB    R2, [R1, #TCB_SP_FRAME]     ; Load next osRtxThread_t.stack_frame

                ANDS    R2, R2, #0x6                ; Check stack frame for VFP context
                MRC     p15, 0, R2, c1, c0, 2       ; Read CPACR
                ANDEQ   R2, R2, #0xFF0FFFFF         ; Disable VFP/NEON access if incoming task does not have stacked VFP/NEON state
                ORRNE   R2, R2, #0x00F00000         ; Enable VFP/NEON access if incoming task does have stacked VFP/NEON state
                MCR     p15, 0, R2, c1, c0, 2       ; Write CPACR
                BEQ     osRtxContextRestore1        ; No VFP
                ISB                                 ; Only sync if we enabled VFP, otherwise we will context switch before next VFP instruction anyway
                IF {TARGET_FEATURE_EXTENSION_REGISTER_COUNT} == 32
                VLDMIA  R3!, {D16-D31}
                ENDIF
                VLDMIA  R3!, {D0-D15}
                LDR     R2, [R3]
                VMSR    FPSCR, R2
                ADD     R3, R3, #8

osRtxContextRestore1
                LDMIA   R3!, {R4-R11}               ; Restore R4-R11
                MOV     R12, R3                     ; Move sp pointer to R12
                ADD     R3, R3, #32                 ; Adjust sp
                PUSH    {R3}                        ; Push sp onto stack
                LDMIA   SP, {SP}^                   ; Restore SP_usr
                LDMIA   R12!, {R0-R3}               ; Restore User R0-R3
                LDR     LR, [R12, #12]              ; Load SPSR into LR
                MSR     SPSR_CXSF, LR               ; Restore SPSR
                ADD     R12, R12, #4                ; Adjust pointer to LR
                LDM     R12, {LR}^                  ; Restore LR_usr directly into LR
                LDR     LR, [R12, #4]               ; Restore LR
                LDR     R12, [R12, #-4]             ; Restore R12

                MOVS    PC, LR                      ; Return from exception

osRtxContextExit
                POP     {R0-R3, R12, LR}            ; Restore stacked APCS registers
                RFEFD   SP!                         ; Return from exception

                END
