;/**************************************************************************//**
; * @file     startup_ARMCA7.s
; * @brief    CMSIS Core Device Startup File for ARM Cortex-A7 Device Series
; * @version  V1.00
; * @date     22 Feb 2017
; *
; * @note
; *
; ******************************************************************************/
;/*
; * Copyright (c) 2009-2017 ARM Limited. All rights reserved.
; *
; * SPDX-License-Identifier: Apache-2.0
; *
; * Licensed under the Apache License, Version 2.0 (the License); you may
; * not use this file except in compliance with the License.
; * You may obtain a copy of the License at
;   *
; * www.apache.org/licenses/LICENSE-2.0
; *
; * Unless required by applicable law or agreed to in writing, software
; * distributed under the License is distributed on an AS IS BASIS, WITHOUT
; * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; * See the License for the specific language governing permissions and
; * limitations under the License.
; */
;/*
;//-------- <<< Use Configuration Wizard in Context Menu >>> ------------------
;*/

GICC_OFFSET     EQU     0x2000
ICCIAR_OFFSET   EQU     0x000C
ICCEOIR_OFFSET  EQU     0x0010

Mode_USR        EQU     0x10
Mode_FIQ        EQU     0x11
Mode_IRQ        EQU     0x12
Mode_SVC        EQU     0x13
Mode_ABT        EQU     0x17
Mode_UND        EQU     0x1B
Mode_SYS        EQU     0x1F

I_Bit           EQU     0x80            ; when I bit is set, IRQ is disabled
F_Bit           EQU     0x40            ; when F bit is set, FIQ is disabled
T_Bit           EQU     0x20            ; when T bit is set, core is in Thumb state

Sect_Normal     EQU     0x00005c06 ;outer & inner wb/wa, non-shareable, executable, rw, domain 0, base addr 0
Sect_Normal_Cod EQU     0x0000dc06 ;outer & inner wb/wa, non-shareable, executable, ro, domain 0, base addr 0
Sect_Normal_RO  EQU     0x0000dc16 ;as Sect_Normal_Cod, but not executable
Sect_Normal_RW  EQU     0x00005c16 ;as Sect_Normal_Cod, but writeable and not executable
Sect_SO         EQU     0x00000c12 ;strongly-ordered (therefore shareable), not executable, rw, domain 0, base addr 0
Sect_Device_RO  EQU     0x00008c12 ;device, non-shareable, non-executable, ro, domain 0, base addr 0
Sect_Device_RW  EQU     0x00000c12 ;as Sect_Device_RO, but writeable
Sect_Fault      EQU     0x00000000 ;this translation will fault (the bottom 2 bits are important, the rest are ignored)

RAM_BASE        EQU     0x80000000
VRAM_BASE       EQU     0x18000000
SRAM_BASE       EQU     0x2e000000
ETHERNET        EQU     0x1a000000
CS3_PERIPHERAL_BASE EQU 0x1c000000

; <h> Stack Configuration
;   <o> Stack Size (in Bytes, per mode) <0x0-0xFFFFFFFF:8>
; </h>

UND_Stack_Size  EQU     0x00000100
SVC_Stack_Size  EQU     0x00000100
ABT_Stack_Size  EQU     0x00000100
FIQ_Stack_Size  EQU     0x00000000
IRQ_Stack_Size  EQU     0x00000100
USR_Stack_Size  EQU     0x00000100

ISR_Stack_Size  EQU     (UND_Stack_Size + SVC_Stack_Size + ABT_Stack_Size + \
                         FIQ_Stack_Size + IRQ_Stack_Size)

                AREA    STACK, NOINIT, READWRITE, ALIGN=3
Stack_Mem       SPACE   USR_Stack_Size
__initial_sp    SPACE   ISR_Stack_Size

Stack_Top


; <h> Heap Configuration
;   <o>  Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
; </h>

Heap_Size       EQU     0x00000000

                AREA    HEAP, NOINIT, READWRITE, ALIGN=3
__heap_base
Heap_Mem        SPACE   Heap_Size
__heap_limit


                PRESERVE8
                ARM


; Vector Table Mapped to Address 0 at Reset

                AREA    RESET, CODE, READONLY
                EXPORT  __Vectors
                EXPORT  __Vectors_End
                EXPORT  __Vectors_Size

__Vectors       LDR     PC, Reset_Addr            ; Address of Reset Handler
                LDR     PC, Undef_Addr            ; Address of Undef Handler
                LDR     PC, SVC_Addr              ; Address of SVC Handler
                LDR     PC, PAbt_Addr             ; Address of Prefetch Abort Handler
                LDR     PC, DAbt_Addr             ; Address of Data Abort Handler
                NOP                               ; Reserved Vector
                LDR     PC, IRQ_Addr              ; Address of IRQ Handler
                LDR     PC, FIQ_Addr              ; Address of FIQ Handler
__Vectors_End

__Vectors_Size  EQU     __Vectors_End - __Vectors

Reset_Addr      DCD     Reset_Handler
Undef_Addr      DCD     Undef_Handler
SVC_Addr        DCD     SVC_Handler
PAbt_Addr       DCD     PAbt_Handler
DAbt_Addr       DCD     DAbt_Handler
IRQ_Addr        DCD     IRQ_Handler
FIQ_Addr        DCD     FIQ_Handler

                AREA    |.text|, CODE, READONLY

Reset_Handler   PROC
                EXPORT  Reset_Handler             [WEAK]
                IMPORT  SystemInit
                IMPORT  __main

                ; Put any cores other than 0 to sleep
                MRC     p15, 0, R0, c0, c0, 5     ; Read MPIDR
                ANDS    R0, R0, #3
goToSleep
                WFINE
                BNE     goToSleep

                MRC     p15, 0, R0, c1, c0, 0       ; Read CP15 System Control register
                BIC     R0, R0, #(0x1 << 12)        ; Clear I bit 12 to disable I Cache
                BIC     R0, R0, #(0x1 <<  2)        ; Clear C bit  2 to disable D Cache
                BIC     R0, R0, #0x1                ; Clear M bit  0 to disable MMU
                BIC     R0, R0, #(0x1 << 11)        ; Clear Z bit 11 to disable branch prediction
                BIC     R0, R0, #(0x1 << 13)        ; Clear V bit 13 to disable hivecs
                MCR     p15, 0, R0, c1, c0, 0       ; Write value back to CP15 System Control register
                ISB

; ACTLR.SMP bit must be set before the caches and MMU are enabled,
; or any cache and TLB maintenance operations are performed, even for "AMP" CPUs
                MRC     p15, 0, r0, c1, c0, 1       ; Read CP15 Auxiliary Control Register
                ORR     r0, r0, #(1 <<  6)          ; Set ACTLR.SMP bit
                ORR     r0, r0, #(1 << 13)          ; Set L1PCTL L1 Data prefetch control to 0b11
                ORR     r0, r0, #(1 << 14)          ;  3 outstanding pre-fetches permitted, this is the reset value
                MCR     p15, 0, r0, c1, c0, 1       ; Write CP15 Auxiliary Control Register

; Set Vector Base Address Register (VBAR) to point to this application's vector table
                LDR     R0, =__Vectors
                MCR     p15, 0, R0, c12, c0, 0

;  Setup Stack for each exceptional mode
                LDR     R0, =Stack_Top

;  Enter Undefined Instruction Mode and set its Stack Pointer
                MSR     CPSR_C, #Mode_UND:OR:I_Bit:OR:F_Bit
                MOV     SP, R0
                SUB     R0, R0, #UND_Stack_Size

;  Enter Abort Mode and set its Stack Pointer
                MSR     CPSR_C, #Mode_ABT:OR:I_Bit:OR:F_Bit
                MOV     SP, R0
                SUB     R0, R0, #ABT_Stack_Size

;  Enter FIQ Mode and set its Stack Pointer
                MSR     CPSR_C, #Mode_FIQ:OR:I_Bit:OR:F_Bit
                MOV     SP, R0
                SUB     R0, R0, #FIQ_Stack_Size

;  Enter IRQ Mode and set its Stack Pointer
                MSR     CPSR_C, #Mode_IRQ:OR:I_Bit:OR:F_Bit
                MOV     SP, R0
                SUB     R0, R0, #IRQ_Stack_Size

;  Enter Supervisor Mode and set its Stack Pointer
                MSR     CPSR_C, #Mode_SVC:OR:I_Bit:OR:F_Bit
                MOV     SP, R0

;  Enter System Mode to complete initialization and enter kernel
                MSR     CPSR_C, #Mode_SYS:OR:I_Bit:OR:F_Bit
                MOV     SP, R0

                IMPORT  MMU_CreateTranslationTable
                BL      MMU_CreateTranslationTable

                MOV     r0, #0x0
                MCR     p15, 0, r0, c8, c7, 0     ; TLBIALL - Invalidate entire Unified TLB
                MCR     p15, 0, r0, c7, c5, 6     ; BPIALL  - Invalidate entire branch predictor array
                DSB
                ISB
                MCR     p15, 0, r0, c7, c5, 0     ; ICIALLU - Invalidate instruction cache and flush branch target cache
                DSB
                ISB

;  Invalidate data cache
                MOV     r0, #0x0                  ; 0 = invalidate data cache, 1 = clean data cache.

                MRC     p15, 1, R6, c0, c0, 1     ; Read CLIDR
                ANDS    R3, R6, #0x07000000       ; Extract coherency level
                MOV     R3, R3, LSR #23           ; Total cache levels << 1
                BEQ     Finished                  ; If 0, no need to clean

                MOV     R10, #0                   ; R10 holds current cache level << 1
Loop1           ADD     R2, R10, R10, LSR #1      ; R2 holds cache "Set" position
                MOV     R1, R6, LSR R2            ; Bottom 3 bits are the Cache-type for this level
                AND     R1, R1, #7                ; Isolate those lower 3 bits
                CMP     R1, #2
                BLT     Skip                      ; No cache or only instruction cache at this level

                MCR     p15, 2, R10, c0, c0, 0    ; Write the Cache Size selection register
                ISB                               ; ISB to sync the change to the CacheSizeID reg
                MRC     p15, 1, R1, c0, c0, 0     ; Reads current Cache Size ID register
                AND     R2, R1, #7                ; Extract the line length field
                ADD     R2, R2, #4                ; Add 4 for the line length offset (log2 16 bytes)
                LDR     R4, =0x3FF
                ANDS    R4, R4, R1, LSR #3        ; R4 is the max number on the way size (right aligned)
                CLZ     R5, R4                    ; R5 is the bit position of the way size increment
                LDR     R7, =0x7FFF
                ANDS    R7, R7, R1, LSR #13       ; R7 is the max number of the index size (right aligned)

Loop2           MOV     R9, R4                    ; R9 working copy of the max way size (right aligned)

Loop3           ORR     R11, R10, R9, LSL R5      ; Factor in the Way number and cache number into R11
                ORR     R11, R11, R7, LSL R2      ; Factor in the Set number
                CMP     R0, #0
                BNE     Dccsw
                MCR     p15, 0, R11, c7, c6, 2    ; DCISW. Invalidate by Set/Way
                B       cont
Dccsw           CMP     R0, #1
                BNE     Dccisw
                MCR     p15, 0, R11, c7, c10, 2   ; DCCSW. Clean by Set/Way
                B       cont
Dccisw          MCR     p15, 0, R11, c7, c14, 2   ; DCCISW. Clean and Invalidate by Set/Way
cont            SUBS    R9, R9, #1                ; Decrement the Way number
                BGE     Loop3
                SUBS    R7, R7, #1                ; Decrement the Set number
                BGE     Loop2
Skip            ADD     R10, R10, #2              ; Increment the cache number
                CMP     R3, R10
                BGT     Loop1
Finished
                DSB

;  Enable MMU, but leave caches disabled (they will be enabled later)
                MRC     p15, 0, r0, c1, c0, 0     ; Read CP15 System Control register
                ORR     r0, r0, #(0x1 << 29)      ; Set AFE bit 29 to enable simplified access permissions model
                BIC     r0, r0, #(0x1 << 28)      ; Clear TRE bit 28 to disable TEX remap
                BIC     r0, r0, #(0x1 << 12)      ; Clear I bit 12 to disable I Cache
                BIC     r0, r0, #(0x1 <<  2)      ; Clear C bit  2 to disable D Cache
                BIC     r0, r0, #(0x1 <<  1)      ; Clear A bit  1 to disable strict alignment fault checking
                ORR     r0, r0, #0x1              ; Set M bit 0 to enable MMU
                MCR     p15, 0, r0, c1, c0, 0     ; Write CP15 System Control register

;  USR/SYS stack pointer will be set during kernel init

                LDR     R0, =SystemInit
                BLX     R0
                LDR     R0, =__main
                BLX     R0

                ENDP

Undef_Handler\
                PROC
                EXPORT  Undef_Handler             [WEAK]
                IMPORT  CUndefHandler
                SRSFD   SP!, #Mode_UND
                PUSH    {R0-R4, R12}              ; Save APCS corruptible registers to UND mode stack

                MRS     R0, SPSR
                TST     R0, #T_Bit                ; Check mode
                MOVEQ   R1, #4                    ; R1 = 4 ARM mode
                MOVNE   R1, #2                    ; R1 = 2 Thumb mode
                SUB     R0, LR, R1
                LDREQ   R0, [R0]                  ; ARM mode - R0 points to offending instruction
                BEQ     undef_cont

                ;Thumb instruction
                ;Determine if it is a 32-bit Thumb instruction
                LDRH    R0, [R0]
                MOV     R2, #0x1c
                CMP     R2, R0, LSR #11
                BHS     undef_cont                ;16-bit Thumb instruction

                ;32-bit Thumb instruction. Unaligned - we need to reconstruct the offending instruction
                LDRH    R2, [LR]
                ORR     R0, R2, R0, LSL #16
undef_cont
                MOV     R2, LR                    ; Set LR to third argument

                AND     R12, SP, #4               ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R12               ; Adjust stack
                PUSH    {R12, LR}                 ; Store stack adjustment and dummy LR

                ;R0 Offending instruction
                ;R1 =2 (Thumb) or =4 (ARM)
                BL      CUndefHandler

                POP     {R12, LR}                 ; Get stack adjustment & discard dummy LR
                ADD     SP, SP, R12               ; Unadjust stack

                LDR     LR, [SP, #24]             ; Restore stacked LR and possibly adjust for retry
                SUB     LR, LR, R0
                LDR     R0, [SP, #28]             ; Restore stacked SPSR
                MSR     SPSR_CXSF, R0
                POP     {R0-R4, R12}              ; Restore stacked APCS registers
                ADD     SP, SP, #8                ; Adjust SP for already-restored banked registers
                MOVS    PC, LR
                ENDP

PAbt_Handler\
                PROC
                EXPORT  PAbt_Handler              [WEAK]
                IMPORT  CPAbtHandler
                SUB     LR, LR, #4                ; Pre-adjust LR
                SRSFD   SP!, #Mode_ABT            ; Save LR and SPRS to ABT mode stack
                PUSH    {R0-R4, R12}              ; Save APCS corruptible registers to ABT mode stack
                MRC     p15, 0, R0, c5, c0, 1     ; IFSR
                MRC     p15, 0, R1, c6, c0, 2     ; IFAR

                MOV     R2, LR                    ; Set LR to third argument

                AND     R12, SP, #4               ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R12               ; Adjust stack
                PUSH    {R12, LR}                 ; Store stack adjustment and dummy LR

                BL      CPAbtHandler

                POP     {R12, LR}                 ; Get stack adjustment & discard dummy LR
                ADD     SP, SP, R12               ; Unadjust stack

                POP     {R0-R4, R12}              ; Restore stack APCS registers
                RFEFD   SP!                       ; Return from exception
                ENDP


DAbt_Handler\
                PROC
                EXPORT  DAbt_Handler              [WEAK]
                IMPORT  CDAbtHandler
                SUB     LR, LR, #8                ; Pre-adjust LR
                SRSFD   SP!, #Mode_ABT            ; Save LR and SPRS to ABT mode stack
                PUSH    {R0-R4, R12}              ; Save APCS corruptible registers to ABT mode stack
                CLREX                             ; State of exclusive monitors unknown after taken data abort
                MRC     p15, 0, R0, c5, c0, 0     ; DFSR
                MRC     p15, 0, R1, c6, c0, 0     ; DFAR

                MOV     R2, LR                    ; Set LR to third argument

                AND     R12, SP, #4               ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R12               ; Adjust stack
                PUSH    {R12, LR}                 ; Store stack adjustment and dummy LR

                BL      CDAbtHandler

                POP     {R12, LR}                 ; Get stack adjustment & discard dummy LR
                ADD     SP, SP, R12               ; Unadjust stack

                POP     {R0-R4, R12}              ; Restore stacked APCS registers
                RFEFD   SP!                       ; Return from exception
                ENDP

FIQ_Handler\
                PROC
                EXPORT  FIQ_Handler               [WEAK]
                B       .
                ENDP

SVC_Handler\
                PROC
                EXPORT  SVC_Handler               [WEAK]
                B       .
                ENDP

IRQ_Handler\
                PROC
                EXPORT  IRQ_Handler                [WEAK]
                IMPORT  IRQCount
                IMPORT  IRQTable
                IMPORT  IRQNestLevel                ; Flag indicates whether inside an ISR, and the depth of nesting.  0 = not in ISR.


                ;prologue
                SUB     LR, LR, #4                  ; Pre-adjust LR
                SRSFD   SP!, #Mode_SVC              ; Save LR_IRQ and SPRS_IRQ to SVC mode stack
                CPS     #Mode_SVC                   ; Switch to SVC mode, to avoid a nested interrupt corrupting LR on a BL
                PUSH    {R0-R3, R12}                ; Save remaining APCS corruptible registers to SVC stack

                AND     R1, SP, #4                  ; Ensure stack is 8-byte aligned
                SUB     SP, SP, R1                  ; Adjust stack
                PUSH    {R1, LR}                    ; Store stack adjustment and LR_SVC to SVC stack

                LDR     R0, =IRQNestLevel           ; Get address of nesting counter
                LDR     R1, [R0]
                ADD     R1, R1, #1                  ; Increment nesting counter
                STR     R1, [R0]

                ;identify and acknowledge interrupt
                MRC     p15, 4, R1, c15, c0, 0      ; Read CBAR
                ADD     R1, R1, #GICC_OFFSET        ; Add GICC offset
                LDR     R0, [R1, #ICCIAR_OFFSET]    ; Read ICCIAR (GIC CPU Interface register)
                DSB                                 ; Ensure that interrupt acknowledge completes before re-enabling interrupts

                LDR     R2, =IRQCount               ; Read number of entries in IRQ handler table
                LDR     R2, [R2]
                CMP     R0, R2                      ; Is there a handler for this IRQ?
                BHS     end_int                     ; No handler, so return as normal
                LDR     R2, =IRQTable               ; Get address of handler
                LDR     R2, [R2, R0, LSL #2]
                CMP     R2, #0                      ; Clean up and return if handler address is 0
                BEQ     end_int
                PUSH    {R0,R1}

                CPSIE   i                           ; Now safe to re-enable interrupts
                BLX     R2                          ; Call handler. R0 = IRQ number. Beware calls to PendSV_Handler and OS_Tick_Handler do not return this way
                CPSID   i                           ; Disable interrupts again

                POP     {R0,R1}
                DSB                                 ; Ensure that interrupt source is cleared before signalling End Of Interrupt
end_int
                ; R0 still contains the interrupt ID
                ; R1 still contains GICI_BASE
                STR     R0, [R1, #ICCEOIR_OFFSET]   ; Normal end-of-interrupt write to EOIR (GIC CPU Interface register) to clear the active bit
ret_irq
                ;epilogue
                LDR     R0, =IRQNestLevel           ; Get address of nesting counter
                LDR     R1, [R0]
                SUB     R1, R1, #1                  ; Decrement nesting counter
                STR     R1, [R0]

                POP     {R1, LR}                    ; Get stack adjustment and restore LR_SVC
                ADD     SP, SP, R1                  ; Unadjust stack

                POP     {R0-R3,R12}                 ; Restore stacked APCS registers
                RFEFD   SP!                         ; Return from exception
                ENDP

                END
