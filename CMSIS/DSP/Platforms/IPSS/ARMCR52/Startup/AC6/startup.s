/******************************************************************************
 * @file     startup_ARMCR8.c
 * @brief    Unvalidated Startup File for a Cortex-R8 Device
 ******************************************************************************/
/*
 * Copyright (c) 2009-2020 Arm Limited. All rights reserved.
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

// MPU region defines

// Protection Base Address Register
#define Execute_Never 0b1         // Bit 0
#define RW_Access 0b01            // AP[2:1]
#define RO_Access 0b11
#define Non_Shareable 0b00        // SH[1:0]
#define Outer_Shareable 0x10
#define Inner_Shareable 0b11

// Protection Limit Address Register
#define ENable 0b1                // Bit 0
#define AttrIndx0 0b000           // AttrIndx[2:0]
#define AttrIndx1 0b001
#define AttrIndx2 0b010
#define AttrIndx3 0b011
#define AttrIndx4 0b100
#define AttrIndx5 0b101
#define AttrIndx6 0b110
#define AttrIndx7 0b111

//----------------------------------------------------------------

// Define some values
#define Mode_USR 0x10
#define Mode_FIQ 0x11
#define Mode_IRQ 0x12
#define Mode_SVC 0x13
#define Mode_MON 0x16
#define Mode_ABT 0x17
#define Mode_UND 0x1B
#define Mode_SYS 0x1F
#define Mode_HYP 0x1A
#define I_Bit 0x80 // when I bit is set, IRQ is disabled
#define F_Bit 0x40 // when F bit is set, FIQ is disabled


// Initial Setup & Entry point
//----------------------------------------------------------------

    .eabi_attribute Tag_ABI_align8_preserved,1
    .section  VECTORS,"ax"
    .align 3

    .global Reset_Handler
Reset_Handler:


// Reset Handlers (EL1 and EL2)
//----------------------------------------------------------------

EL2_Reset_Handler:

    .global  Image$$ARM_LIB_STACKHEAP$$ZI$$Limit
    LDR SP, =Image$$ARM_LIB_STACKHEAP$$ZI$$Limit


    //----------------------------------------------------------------
    // Disable MPU and caches
    //----------------------------------------------------------------

    // Disable MPU and cache in case it was left enabled from an earlier run
    // This does not need to be done from a cold reset

        MRC p15, 0, r0, c1, c0, 0       // Read System Control Register
        BIC r0, r0, #0x05               // Disable MPU (M bit) and data cache (C bit)
        BIC r0, r0, #0x1000             // Disable instruction cache (I bit)
        DSB                             // Ensure all previous loads/stores have completed
        MCR p15, 0, r0, c1, c0, 0       // Write System Control Register
        ISB                             // Ensure subsequent insts execute wrt new MPU settings

//----------------------------------------------------------------
// Cache invalidation. However Cortex-R52 provides CFG signals to
// invalidate cache automatically out of reset (CFGL1CACHEINVDISx)
//----------------------------------------------------------------

        DSB             // Complete all outstanding explicit memory operations

        MOV r0, #0

        MCR p15, 0, r0, c7, c5, 0       // Invalidate entire instruction cache

        // Invalidate Data/Unified Caches

        MRC     p15, 1, r0, c0, c0, 1      // Read CLIDR
        ANDS    r3, r0, #0x07000000        // Extract coherency level
        MOV     r3, r3, LSR #23            // Total cache levels << 1
        BEQ     Finished                   // If 0, no need to clean

        MOV     r10, #0                    // R10 holds current cache level << 1
Loop1:  ADD     r2, r10, r10, LSR #1       // R2 holds cache "Set" position
        MOV     r1, r0, LSR r2             // Bottom 3 bits are the Cache-type for this level
        AND     r1, r1, #7                 // Isolate those lower 3 bits
        CMP     r1, #2
        BLT     Skip                       // No cache or only instruction cache at this level

        MCR     p15, 2, r10, c0, c0, 0     // Write the Cache Size selection register
        ISB                                // ISB to sync the change to the CacheSizeID reg
        MRC     p15, 1, r1, c0, c0, 0      // Reads current Cache Size ID register
        AND     r2, r1, #7                 // Extract the line length field
        ADD     r2, r2, #4                 // Add 4 for the line length offset (log2 16 bytes)
        LDR     r4, =0x3FF
        ANDS    r4, r4, r1, LSR #3         // R4 is the max number on the way size (right aligned)
        CLZ     r5, r4                     // R5 is the bit position of the way size increment
        LDR     r7, =0x7FFF
        ANDS    r7, r7, r1, LSR #13        // R7 is the max number of the index size (right aligned)

Loop2:  MOV     r9, r4                     // R9 working copy of the max way size (right aligned)

#ifdef __THUMB__
Loop3:  LSL     r12, r9, r5
        ORR     r11, r10, r12              // Factor in the Way number and cache number into R11
        LSL     r12, r7, r2
        ORR     r11, r11, r12              // Factor in the Set number
#else
Loop3:  ORR     r11, r10, r9, LSL r5       // Factor in the Way number and cache number into R11
        ORR     r11, r11, r7, LSL r2       // Factor in the Set number
#endif
        MCR     p15, 0, r11, c7, c6, 2     // Invalidate by Set/Way
        SUBS    r9, r9, #1                 // Decrement the Way number
        BGE     Loop3
        SUBS    r7, r7, #1                 // Decrement the Set number
        BGE     Loop2
Skip:   ADD     r10, r10, #2               // Increment the cache number
        CMP     r3, r10
        BGT     Loop1

Finished:

        

//----------------------------------------------------------------
// TCM Configuration
//----------------------------------------------------------------

// Cortex-R52 optionally provides three Tightly-Coupled Memory (TCM) blocks (ATCM, BTCM and CTCM)
//    for fast access to code or data.

// The following illustrates basic TCM configuration, as the basis for exploration by the user

#ifdef TCM

        MRC p15, 0, r0, c0, c0, 2       // Read TCM Type Register
        // r0 now contains TCM availability

        MRC p15, 0, r0, c9, c1, 0       // Read ATCM Region Register
        // r0 now contains ATCM size in bits [5:2]
        LDR r0, =Image$$CODE$$Base      // Set ATCM base address
        ORR r0, r0, #3                  // Enable it
        MCR p15, 0, r0, c9, c1, 0       // Write ATCM Region Register

        MRC p15, 0, r0, c9, c1, 1       // Read BTCM Region Register
        // r0 now contains BTCM size in bits [5:2]
        LDR r0, =Image$$DATA$$Base      // Set BTCM base address
        ORR r0, r0, #3                  // Enable it
        MCR p15, 0, r0, c9, c1, 1       // Write BTCM Region Register

        MRC p15, 0, r0, c9, c1, 2       // Read CTCM Region Register
        // r0 now contains CTCM size in bits [5:2]
        LDR r0, =Image$$CTCM$$Base      // Set CTCM base address
        ORR r0, r0, #1                  // Enable it
        MCR p15, 0, r0, c9, c1, 2       // Write CTCM Region Register

#endif

//----------------------------------------------------------------
// MPU Configuration
//----------------------------------------------------------------

// Notes:
// * Regions apply to both instruction and data accesses.
// * Each region base address must be a multiple of its size
// * Any address range not covered by an enabled region will abort
// * The region at 0x0 over the Vector table is needed to support semihosting

// Region 0: Code          Base = See scatter file  Limit = Based on usage   Normal  Non-shared  Read-only    Executable
// Region 1: Data          Base = See scatter file  Limit = Based on usage   Normal  Non-shared  Full access  Not Executable
// Region 2: Stack/Heap    Base = See scatter file  Limit = Based on usage   Normal  Non-shared  Full access  Not Executable
// Region 3: Peripherals   Base = 0xB0000000        Limit = 0xBFFFFFC0       Device              Full access  Not Executable
// Region 4: ATCM          Base = Configurable      Limit = Based on usage   Normal  Non-shared  Full access  Executable
// Region 5: BTCM          Base = Configurable      Limit = Based on usage   Normal  Non-shared  Full access  Executable
// Region 6: CTCM          Base = Configurable      Limit = Based on usage   Normal  Non-shared  Full access  Executable

        LDR     r0, =64

        // Region 0 - Code
REG0:
        LDR     r1, =Image$$CODE$$Base
        LDR     r2, =((Non_Shareable<<3) | (RO_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c8, 0                   // write PRBAR0
        LDR     r1, =Image$$CODE$$Limit
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx0<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c8, 1                   // write PRLAR0

        // Region 1 - Data
REG1:
        LDR     r1, =Image$$DATA$$Base
        LDR     r2, =((Non_Shareable<<3) | (RW_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c8, 4                   // write PRBAR1
        LDR     r1, =Image$$DATA$$ZI$$Limit
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx0<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c8, 5                   // write PRLAR1

        // Region 2 - Stack-Heap
REG2:
        LDR     r1, =Image$$ARM_LIB_STACKHEAP$$Base
        LDR     r2, =((Non_Shareable<<3) | (RW_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c9, 0                   // write PRBAR2
        LDR     r1, =Image$$ARM_LIB_STACKHEAP$$ZI$$Limit
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx0<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c9, 1                   // write PRLAR2

        // Region 3 - Peripherals
REG3:
        LDR     r1, =0xAA000000
        LDR     r2, =((Non_Shareable<<3) | (RW_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c9, 4                   // write PRBAR3
        LDR     r1, =0xBFFFFFC0
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx0<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c9, 5                   // write PRLAR3

#ifdef TCM
        // Region 4 - ATCM
        LDR     r1, =Image$$ATCM$$Base
        LDR     r2, =((Non_Shareable<<3) | (RW_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c10, 0                  // write PRBAR4
        LDR     r1, =Image$$ATCM$$Limit
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx1<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c10, 1                  // write PRLAR4

        // Region 5 - BTCM
        LDR     r1, =Image$$BTCM$$Base
        LDR     r2, =((Non_Shareable<<3) | (RW_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c10, 4                  // write PRBAR5
        LDR     r1, =Image$$BTCM$$Limit
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx0<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c10, 5                  // write PRLAR5

        // Region 6 - CTCM
        LDR     r1, =Image$$CTCM$$Base
        LDR     r2, =((Non_Shareable<<3) | (RW_Access<<1))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c11, 0                  // write PRBAR6
        LDR     r1, =Image$$CTCM$$Limit
        ADD     r1, r1, #63
        BFC     r1, #0, #6                              // align Limit to 64bytes
        LDR     r2, =((AttrIndx0<<1) | (ENable))
        ORR     r1, r1, r2
        MCR     p15, 0, r1, c6, c11, 1                  // write PRLAR6
#endif


other_mems_en:
        // Enable PERIPHREGIONR (LLPP)
        mrc     p15, 0, r1, c15, c0, 0   // PERIPHREGIONR
        orr     r1, r1, #(0x1 << 1)      // Enable PERIPHREGIONR EL2
        orr     r1, r1, #(0x1)           // Enable PERIPHREGIONR EL10
        mcr     p15, 0, r1, c15, c0, 0   // PERIPHREGIONR

//#ifdef __ARM_FP
//----------------------------------------------------------------
// Enable access to VFP by enabling access to Coprocessors 10 and 11.
// Enables Full Access i.e. in both privileged and non privileged modes
//----------------------------------------------------------------

        MRC     p15, 0, r0, c1, c0, 2      // Read Coprocessor Access Control Register (CPACR)
        ORR     r0, r0, #(0xF << 20)       // Enable access to CP 10 & 11
        MCR     p15, 0, r0, c1, c0, 2      // Write Coprocessor Access Control Register (CPACR)
        ISB

//----------------------------------------------------------------
// Switch on the VFP hardware
//----------------------------------------------------------------

        MOV     r0, #0x40000000
        VMSR    FPEXC, r0                   // Write FPEXC register, EN bit set
//#endif


//----------------------------------------------------------------
// Enable MPU and branch to C library init
// Leaving the caches disabled until after scatter loading.
//----------------------------------------------------------------

        MRC     p15, 0, r0, c1, c0, 0       // Read System Control Register
        ORR     r0, r0, #0x01               // Set M bit to enable MPU
        DSB                                 // Ensure all previous loads/stores have completed
        MCR     p15, 0, r0, c1, c0, 0       // Write System Control Register
        ISB                                 // Ensure subsequent insts execute wrt new MPU settings

//Check which CPU I am
        MRC p15, 0, r0, c0, c0, 5       // Read MPIDR
        ANDS r0, r0, 0xF
        BEQ cpu0                        // If CPU0 then initialise C runtime
        CMP r0, #1
        BEQ loop_wfi                    // If CPU1 then jump to loop_wfi
        CMP r0, #2
        BEQ loop_wfi                    // If CPU2 then jump to loop_wfi
        CMP r0, #3
        BEQ loop_wfi                    // If CPU3 then jump to loop_wfi
error:
        B error                         // else.. something is wrong

loop_wfi:
        DSB SY      // Clear all pending data accesses
        WFI         // Go to sleep
        B loop_wfi


cpu0:

    // Branch to __main
    //------------------------
    .global     __main
        B      __main


//----------------------------------------------------------------
// Global Enable for Instruction and Data Caching
//----------------------------------------------------------------
    .global enable_caches
    .type enable_caches, "function"
    .cfi_startproc
enable_caches:

        MRC     p15, 4, r0, c1, c0, 0       // read System Control Register
        ORR     r0, r0, #(0x1 << 12)        // Set I bit 12 to enable I Cache
        ORR     r0, r0, #(0x1 << 2)         // Set C bit  2 to enable D Cache
        MCR     p15, 4, r0, c1, c0, 0       // write System Control Register
        ISB

        BX    lr
    .cfi_endproc

    .size enable_caches, . - enable_caches


// Exception Vector Table & Handlers
//----------------------------------------------------------------

EL2_Vectors:

    LDR PC, EL2_Reset_Addr
    LDR PC, EL2_Undefined_Addr
    LDR PC, EL2_HVC_Addr
    LDR PC, EL2_Prefetch_Addr
    LDR PC, EL2_Abort_Addr
    LDR PC, EL2_HypModeEntry_Addr
    LDR PC, EL2_IRQ_Addr
    LDR PC, EL2_FIQ_Addr

    EL2_Reset_Addr:         .word    EL2_Reset_Handler
    EL2_Undefined_Addr:     .word    EL2_Undefined_Handler
    EL2_HVC_Addr:           .word    EL2_HVC_Handler
    EL2_Prefetch_Addr:      .word    EL2_Prefetch_Handler
    EL2_Abort_Addr:         .word    EL2_Abort_Handler
    EL2_HypModeEntry_Addr:  .word    EL2_HypModeEntry_Handler
    EL2_IRQ_Addr:           .word    EL2_IRQ_Handler
    EL2_FIQ_Addr:           .word    EL2_FIQ_Handler

    EL2_Undefined_Handler:          B   EL2_Undefined_Handler
    EL2_HVC_Handler:                B   EL2_HVC_Handler
    EL2_Prefetch_Handler:           B   EL2_Prefetch_Handler
    EL2_Abort_Handler:              B   EL2_Abort_Handler
    EL2_HypModeEntry_Handler:       B   EL2_HypModeEntry_Handler
    EL2_IRQ_Handler:                B   EL2_IRQ_Handler
    EL2_FIQ_Handler:                B   EL2_FIQ_Handler


EL1_Vectors:

    LDR PC, EL1_Reset_Addr
    LDR PC, EL1_Undefined_Addr
    LDR PC, EL1_SVC_Addr
    LDR PC, EL1_Prefetch_Addr
    LDR PC, EL1_Abort_Addr
    LDR PC, EL1_Reserved
    LDR PC, EL1_IRQ_Addr
    LDR PC, EL1_FIQ_Addr

    EL1_Reset_Addr:     .word    EL1_Reset_Handler
    EL1_Undefined_Addr: .word    EL1_Undefined_Handler
    EL1_SVC_Addr:       .word    EL1_SVC_Handler
    EL1_Prefetch_Addr:  .word    EL1_Prefetch_Handler
    EL1_Abort_Addr:     .word    EL1_Abort_Handler
    EL1_Reserved_Addr:  .word    EL1_Reserved
    EL1_IRQ_Addr:       .word    EL1_IRQ_Handler
    EL1_FIQ_Addr:       .word    EL1_FIQ_Handler

    EL1_Reset_Handler:              B   EL1_Reset_Handler
    EL1_Undefined_Handler:          B   EL1_Undefined_Handler
    EL1_SVC_Handler:                B   EL1_SVC_Handler
    EL1_Prefetch_Handler:           B   EL1_Prefetch_Handler
    EL1_Abort_Handler:              B   EL1_Abort_Handler
    EL1_Reserved:                   B   EL1_Reserved
    EL1_IRQ_Handler:                B   EL1_IRQ_Handler
    EL1_FIQ_Handler:                B   EL1_FIQ_Handler