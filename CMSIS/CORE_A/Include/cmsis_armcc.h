/**************************************************************************//**
 * @file     cmsis_armcc.h
 * @brief    CMSIS compiler specific macros, functions, instructions
 * @version  V1.00
 * @date     22. Feb 2017
 ******************************************************************************/
/*
 * Copyright (c) 2009-2017 ARM Limited. All rights reserved.
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

#ifndef __CMSIS_ARMCC_H
#define __CMSIS_ARMCC_H

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 400677)
  #error "Please use ARM Compiler Toolchain V4.0.677 or later!"
#endif

/* CMSIS compiler control architecture macros */
#if (defined (__TARGET_ARCH_7_A ) && (__TARGET_ARCH_7_A  == 1))
  #define __ARM_ARCH_7A__           1
#endif

/* CMSIS compiler specific defines */
#ifndef   __ASM
  #define __ASM                     __asm
#endif
#ifndef   __INLINE
  #define __INLINE                  __inline
#endif
#ifndef   __STATIC_INLINE
  #define __STATIC_INLINE           static __inline
#endif
#ifndef   __STATIC_ASM
  #define __STATIC_ASM              static __asm
#endif
#ifndef   __NO_RETURN
  #define __NO_RETURN               __declspec(noreturn)
#endif
#ifndef   __USED
  #define __USED                    __attribute__((used))
#endif
#ifndef   __WEAK
  #define __WEAK                    __attribute__((weak))
#endif
#ifndef   __UNALIGNED_UINT32
  #define __UNALIGNED_UINT32(x)     (*((__packed uint32_t *)(x)))
#endif
#ifndef   __ALIGNED
  #define __ALIGNED(x)              __attribute__((aligned(x)))
#endif
#ifndef   __PACKED
  #define __PACKED                  __attribute__((packed))
#endif


/* ###########################  Core Function Access  ########################### */
/** \ingroup  CMSIS_Core_FunctionInterface
    \defgroup CMSIS_Core_RegAccFunctions CMSIS Core Register Access Functions
  @{
 */

/**
  \brief   Get FPSCR
  \details Returns the current value of the Floating Point Status/Control register.
  \return               Floating Point Status/Control register value
 */
__STATIC_INLINE uint32_t __get_FPSCR(void)
{
#if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
     (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
  register uint32_t __regfpscr         __ASM("fpscr");
  return(__regfpscr);
#else
   return(0U);
#endif
}

/**
  \brief   Set FPSCR
  \details Assigns the given value to the Floating Point Status/Control register.
  \param [in]    fpscr  Floating Point Status/Control value to set
 */
__STATIC_INLINE void __set_FPSCR(uint32_t fpscr)
{
#if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
     (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
  register uint32_t __regfpscr         __ASM("fpscr");
  __regfpscr = (fpscr);
#else
  (void)fpscr;
#endif
}

/*@} end of CMSIS_Core_RegAccFunctions */


/* ##########################  Core Instruction Access  ######################### */
/** \defgroup CMSIS_Core_InstructionInterface CMSIS Core Instruction Interface
  Access to dedicated instructions
  @{
*/

/**
  \brief   No Operation
  \details No Operation does nothing. This instruction can be used for code alignment purposes.
 */
#define __NOP                             __nop

/**
  \brief   Wait For Interrupt
  \details Wait For Interrupt is a hint instruction that suspends execution until one of a number of events occurs.
 */
#define __WFI                             __wfi

/**
  \brief   Wait For Event
  \details Wait For Event is a hint instruction that permits the processor to enter
           a low-power state until one of a number of events occurs.
 */
#define __WFE                             __wfe

/**
  \brief   Send Event
  \details Send Event is a hint instruction. It causes an event to be signaled to the CPU.
 */
#define __SEV                             __sev

/**
  \brief   Instruction Synchronization Barrier
  \details Instruction Synchronization Barrier flushes the pipeline in the processor,
           so that all instructions following the ISB are fetched from cache or memory,
           after the instruction has been completed.
 */
#define __ISB() do {\
                   __schedule_barrier();\
                   __isb(0xF);\
                   __schedule_barrier();\
                } while (0U)

/**
  \brief   Data Synchronization Barrier
  \details Acts as a special kind of Data Memory Barrier.
           It completes when all explicit memory accesses before this instruction complete.
 */
#define __DSB() do {\
                   __schedule_barrier();\
                   __dsb(0xF);\
                   __schedule_barrier();\
                } while (0U)

/**
  \brief   Data Memory Barrier
  \details Ensures the apparent order of the explicit memory operations before
           and after the instruction, without ensuring their completion.
 */
#define __DMB() do {\
                   __schedule_barrier();\
                   __dmb(0xF);\
                   __schedule_barrier();\
                } while (0U)

/**
  \brief   Reverse byte order (32 bit)
  \details Reverses the byte order in integer value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REV                             __rev

/**
  \brief   Reverse byte order (16 bit)
  \details Reverses the byte order in two unsigned short values.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#ifndef __NO_EMBEDDED_ASM
__attribute__((section(".rev16_text"))) __STATIC_INLINE __ASM uint32_t __REV16(uint32_t value)
{
  rev16 r0, r0
  bx lr
}
#endif

/**
  \brief   Reverse byte order in signed short value
  \details Reverses the byte order in a signed short value with sign extension to integer.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#ifndef __NO_EMBEDDED_ASM
__attribute__((section(".revsh_text"))) __STATIC_INLINE __ASM int32_t __REVSH(int32_t value)
{
  revsh r0, r0
  bx lr
}
#endif

/**
  \brief   Rotate Right in unsigned value (32 bit)
  \details Rotate Right (immediate) provides the value of the contents of a register rotated by a variable number of bits.
  \param [in]    op1  Value to rotate
  \param [in]    op2  Number of Bits to rotate
  \return               Rotated value
 */
#define __ROR                             __ror

/**
  \brief   Breakpoint
  \details Causes the processor to enter Debug state.
           Debug tools can use this to investigate system state when the instruction at a particular address is reached.
  \param [in]    value  is ignored by the processor.
                 If required, a debugger can use it to store additional information about the breakpoint.
 */
#define __BKPT(value)                     __breakpoint(value)

/**
  \brief   Reverse bit order of value
  \details Reverses the bit order of the given value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __RBIT                            __rbit

/**
  \brief   Count leading zeros
  \details Counts the number of leading zeros of a data value.
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
#define __CLZ                             __clz

/** \brief  Get CPSR Register

    This function returns the content of the CPSR Register.

    \return               CPSR Register value
 */
__STATIC_INLINE uint32_t __get_CPSR(void)
{
  register uint32_t __regCPSR          __ASM("cpsr");
  return(__regCPSR);
}

/** \brief  Set Stack Pointer

    This function assigns the given value to the current stack pointer.

    \param [in]    stack  Stack Pointer value to set
 */
__STATIC_INLINE __ASM void __set_SP(uint32_t stack)
{
  MOV  sp, r0
  BX   lr
}

/** \brief  Set Process Stack Pointer

    This function assigns the given value to the USR/SYS Stack Pointer (PSP).

    \param [in]    topOfProcStack  USR/SYS Stack Pointer value to set
 */
__STATIC_INLINE __ASM void __set_PSP(uint32_t topOfProcStack)
{
  ARM
  PRESERVE8

  BIC     R0, R0, #7  ;ensure stack is 8-byte aligned
  MRS     R1, CPSR
  CPS     #0x1F       ;no effect in USR mode
  MOV     SP, R0
  MSR     CPSR_c, R1  ;no effect in USR mode
  ISB
  BX      LR
}

/** \brief  Set User Mode

    This function changes the processor state to User Mode
 */
__STATIC_INLINE __ASM void __set_CPS_USR(void)
{
  ARM

  CPS  #0x10
  BX   LR
}

/** \brief  Set Mode

    This function changes the processor mode

    \param [in]    mode  Mode value to set
 */
__STATIC_INLINE __ASM void __set_mode(uint32_t mode) {
  MOV  r1, lr
  MSR  CPSR_C, r0
  BX   r1
}
/** \brief  Get FPEXC

    This function returns the current value of the Floating Point Exception Control register.

    \return               Floating Point Exception Control register value
 */
__STATIC_INLINE uint32_t __get_FPEXC(void)
{
#if (__FPU_PRESENT == 1)
  register uint32_t __regfpexc         __ASM("fpexc");
  return(__regfpexc);
#else
  return(0);
#endif
}

/** \brief  Set FPEXC

    This function assigns the given value to the Floating Point Exception Control register.

    \param [in]    fpexc  Floating Point Exception Control value to set
 */
__STATIC_INLINE void __set_FPEXC(uint32_t fpexc)
{
#if (__FPU_PRESENT == 1)
  register uint32_t __regfpexc         __ASM("fpexc");
  __regfpexc = (fpexc);
#endif
}

/** \brief  Get CPACR

    This function returns the current value of the Coprocessor Access Control register.

    \return               Coprocessor Access Control register value
 */
__STATIC_INLINE uint32_t __get_CPACR(void)
{
  register uint32_t __regCPACR         __ASM("cp15:0:c1:c0:2");
  return __regCPACR;
}

/** \brief  Set CPACR

    This function assigns the given value to the Coprocessor Access Control register.

    \param [in]    cpacr  Coprocessor Acccess Control value to set
 */
__STATIC_INLINE void __set_CPACR(uint32_t cpacr)
{
  register uint32_t __regCPACR         __ASM("cp15:0:c1:c0:2");
  __regCPACR = cpacr;
}

/** \brief  Get CBAR

    This function returns the value of the Configuration Base Address register.

    \return               Configuration Base Address register value
 */
__STATIC_INLINE uint32_t __get_CBAR() {
  register uint32_t __regCBAR         __ASM("cp15:4:c15:c0:0");
  return(__regCBAR);
}

/** \brief  Get TTBR0

    This function returns the value of the Translation Table Base Register 0.

    \return               Translation Table Base Register 0 value
 */
__STATIC_INLINE uint32_t __get_TTBR0() {
  register uint32_t __regTTBR0        __ASM("cp15:0:c2:c0:0");
  return(__regTTBR0);
}

/** \brief  Set TTBR0

    This function assigns the given value to the Translation Table Base Register 0.

    \param [in]    ttbr0  Translation Table Base Register 0 value to set
 */
__STATIC_INLINE void __set_TTBR0(uint32_t ttbr0) {
  register uint32_t __regTTBR0        __ASM("cp15:0:c2:c0:0");
  __regTTBR0 = ttbr0;
}

/** \brief  Get DACR

    This function returns the value of the Domain Access Control Register.

    \return               Domain Access Control Register value
 */
__STATIC_INLINE uint32_t __get_DACR() {
  register uint32_t __regDACR         __ASM("cp15:0:c3:c0:0");
  return(__regDACR);
}

/** \brief  Set DACR

    This function assigns the given value to the Domain Access Control Register.

    \param [in]    dacr   Domain Access Control Register value to set
 */
__STATIC_INLINE void __set_DACR(uint32_t dacr) {
  register uint32_t __regDACR         __ASM("cp15:0:c3:c0:0");
  __regDACR = dacr;
}

/** \brief  Set SCTLR

    This function assigns the given value to the System Control Register.

    \param [in]    sctlr  System Control Register value to set
 */
__STATIC_INLINE void __set_SCTLR(uint32_t sctlr)
{
  register uint32_t __regSCTLR         __ASM("cp15:0:c1:c0:0");
  __regSCTLR = sctlr;
}

/** \brief  Get SCTLR

    This function returns the value of the System Control Register.

    \return               System Control Register value
 */
__STATIC_INLINE uint32_t __get_SCTLR() {
  register uint32_t __regSCTLR         __ASM("cp15:0:c1:c0:0");
  return(__regSCTLR);
}

/** \brief  Set ACTRL

    This function assigns the given value to the Auxiliary Control Register.

    \param [in]    actlr  Auxiliary Control Register value to set
 */
__STATIC_INLINE void __set_ACTRL(uint32_t actrl)
{
  register uint32_t __regACTRL         __ASM("cp15:0:c1:c0:1");
  __regACTRL = actrl;
}

/** \brief  Get ACTRL

    This function returns the value of the Auxiliary Control Register.

    \return               Auxiliary Control Register value
 */
__STATIC_INLINE uint32_t __get_ACTRL(void)
{
  register uint32_t __regACTRL         __ASM("cp15:0:c1:c0:1");
  return(__regACTRL);
}

/** \brief  Get MPIDR

    This function returns the value of the Multiprocessor Affinity Register.

    \return               Multiprocessor Affinity Register value
 */
__STATIC_INLINE uint32_t __get_MPIDR(void)
{
  register uint32_t __regMPIDR         __ASM("cp15:0:c0:c0:5");
  return(__regMPIDR);
}

/** \brief  Set CNTP_TVAL

  This function assigns the given value to PL1 Physical Timer Value Register (CNTP_TVAL).

  \param [in]    value  CNTP_TVAL Register value to set
*/
__STATIC_INLINE void __set_CNTP_TVAL(uint32_t value) {
  register uint32_t __regCNTP_TVAL         __ASM("cp15:0:c14:c2:0");
  __regCNTP_TVAL = value;
}

 /** \brief  Get VBAR

    This function returns the value of the Vector Base Address Register.

    \return               Vector Base Address Register
 */
__STATIC_INLINE uint32_t __get_VBAR(void)
{
  register uint32_t __regVBAR         __ASM("cp15:0:c12:c0:0");
  return(__regVBAR);
}

/** \brief  Set VBAR

    This function assigns the given value to the Vector Base Address Register.

    \param [in]    vbar  Vector Base Address Register value to set
 */
__STATIC_INLINE void __set_VBAR(uint32_t vbar)
{
  register uint32_t __regVBAR          __ASM("cp15:0:c12:c0:0");
  __regVBAR = vbar;
}

/** \brief  Get CNTP_TVAL

    This function returns the value of the PL1 Physical Timer Value Register (CNTP_TVAL).

    \return               CNTP_TVAL Register value
 */
__STATIC_INLINE uint32_t __get_CNTP_TVAL() {
  register uint32_t __regCNTP_TVAL         __ASM("cp15:0:c14:c2:0");
  return(__regCNTP_TVAL);
}

/** \brief  Set CNTP_CTL

  This function assigns the given value to PL1 Physical Timer Control Register (CNTP_CTL).

  \param [in]    value  CNTP_CTL Register value to set
*/
__STATIC_INLINE void __set_CNTP_CTL(uint32_t value) {
  register uint32_t __regCNTP_CTL          __ASM("cp15:0:c14:c2:1");
  __regCNTP_CTL = value;
}

/** \brief  Set TLBIALL

  TLB Invalidate All
 */
__STATIC_INLINE void __set_TLBIALL(uint32_t value) {
  register uint32_t __TLBIALL              __ASM("cp15:0:c8:c7:0");
  __TLBIALL = value;
}

/** \brief  Set BPIALL.

  Branch Predictor Invalidate All
 */
__STATIC_INLINE void __set_BPIALL(uint32_t value) {
  register uint32_t __BPIALL            __ASM("cp15:0:c7:c5:6");
  __BPIALL = value;
}

/** \brief  Set ICIALLU

  Instruction Cache Invalidate All
 */
__STATIC_INLINE void __set_ICIALLU(uint32_t value) {
  register uint32_t __ICIALLU         __ASM("cp15:0:c7:c5:0");
  __ICIALLU = value;
}

/** \brief  Set DCCMVAC

  Data cache clean
 */
__STATIC_INLINE void __set_DCCMVAC(uint32_t value) {
  register uint32_t __DCCMVAC         __ASM("cp15:0:c7:c10:1");
  __DCCMVAC = value;
}

/** \brief  Set DCIMVAC

  Data cache invalidate
 */
__STATIC_INLINE void __set_DCIMVAC(uint32_t value) {
  register uint32_t __DCIMVAC         __ASM("cp15:0:c7:c6:1");
  __DCIMVAC = value;
}

/** \brief  Set DCCIMVAC

  Data cache clean and invalidate
 */
__STATIC_INLINE void __set_DCCIMVAC(uint32_t value) {
  register uint32_t __DCCIMVAC        __ASM("cp15:0:c7:c14:1");
  __DCCIMVAC = value;
}

/** \brief  Clean and Invalidate the entire data or unified cache

  Generic mechanism for cleaning/invalidating the entire data or unified cache to the point of coherency
 */
#pragma push
#pragma arm
__STATIC_INLINE __ASM void __L1C_CleanInvalidateCache(uint32_t op) {
        ARM

        PUSH    {R4-R11}

        MRC     p15, 1, R6, c0, c0, 1      // Read CLIDR
        ANDS    R3, R6, #0x07000000        // Extract coherency level
        MOV     R3, R3, LSR #23            // Total cache levels << 1
        BEQ     Finished                   // If 0, no need to clean

        MOV     R10, #0                    // R10 holds current cache level << 1
Loop1   ADD     R2, R10, R10, LSR #1       // R2 holds cache "Set" position
        MOV     R1, R6, LSR R2             // Bottom 3 bits are the Cache-type for this level
        AND     R1, R1, #7                 // Isolate those lower 3 bits
        CMP     R1, #2
        BLT     Skip                       // No cache or only instruction cache at this level

        MCR     p15, 2, R10, c0, c0, 0     // Write the Cache Size selection register
        ISB                                // ISB to sync the change to the CacheSizeID reg
        MRC     p15, 1, R1, c0, c0, 0      // Reads current Cache Size ID register
        AND     R2, R1, #7                 // Extract the line length field
        ADD     R2, R2, #4                 // Add 4 for the line length offset (log2 16 bytes)
        LDR     R4, =0x3FF
        ANDS    R4, R4, R1, LSR #3         // R4 is the max number on the way size (right aligned)
        CLZ     R5, R4                     // R5 is the bit position of the way size increment
        LDR     R7, =0x7FFF
        ANDS    R7, R7, R1, LSR #13        // R7 is the max number of the index size (right aligned)

Loop2   MOV     R9, R4                     // R9 working copy of the max way size (right aligned)

Loop3   ORR     R11, R10, R9, LSL R5       // Factor in the Way number and cache number into R11
        ORR     R11, R11, R7, LSL R2       // Factor in the Set number
        CMP     R0, #0
        BNE     Dccsw
        MCR     p15, 0, R11, c7, c6, 2     // DCISW. Invalidate by Set/Way
        B       cont
Dccsw   CMP     R0, #1
        BNE     Dccisw
        MCR     p15, 0, R11, c7, c10, 2    // DCCSW. Clean by Set/Way
        B       cont
Dccisw  MCR     p15, 0, R11, c7, c14, 2    // DCCISW. Clean and Invalidate by Set/Way
cont    SUBS    R9, R9, #1                 // Decrement the Way number
        BGE     Loop3
        SUBS    R7, R7, #1                 // Decrement the Set number
        BGE     Loop2
Skip    ADD     R10, R10, #2               // Increment the cache number
        CMP     R3, R10
        BGT     Loop1

Finished
        DSB
        POP    {R4-R11}
        BX     lr
}
#pragma pop

/** \brief  Enable Floating Point Unit

  Critical section, called from undef handler, so systick is disabled
 */
#pragma push
#pragma arm
__STATIC_INLINE __ASM void __FPU_Enable(void) {
        ARM

        //Permit access to VFP/NEON, registers by modifying CPACR
        MRC     p15,0,R1,c1,c0,2
        ORR     R1,R1,#0x00F00000
        MCR     p15,0,R1,c1,c0,2

        //Ensure that subsequent instructions occur in the context of VFP/NEON access permitted
        ISB

        //Enable VFP/NEON
        VMRS    R1,FPEXC
        ORR     R1,R1,#0x40000000
        VMSR    FPEXC,R1

        //Initialise VFP/NEON registers to 0
        MOV     R2,#0
  IF {TARGET_FEATURE_EXTENSION_REGISTER_COUNT} >= 16
        //Initialise D16 registers to 0
        VMOV    D0, R2,R2
        VMOV    D1, R2,R2
        VMOV    D2, R2,R2
        VMOV    D3, R2,R2
        VMOV    D4, R2,R2
        VMOV    D5, R2,R2
        VMOV    D6, R2,R2
        VMOV    D7, R2,R2
        VMOV    D8, R2,R2
        VMOV    D9, R2,R2
        VMOV    D10,R2,R2
        VMOV    D11,R2,R2
        VMOV    D12,R2,R2
        VMOV    D13,R2,R2
        VMOV    D14,R2,R2
        VMOV    D15,R2,R2
  ENDIF
  IF {TARGET_FEATURE_EXTENSION_REGISTER_COUNT} == 32
        //Initialise D32 registers to 0
        VMOV    D16,R2,R2
        VMOV    D17,R2,R2
        VMOV    D18,R2,R2
        VMOV    D19,R2,R2
        VMOV    D20,R2,R2
        VMOV    D21,R2,R2
        VMOV    D22,R2,R2
        VMOV    D23,R2,R2
        VMOV    D24,R2,R2
        VMOV    D25,R2,R2
        VMOV    D26,R2,R2
        VMOV    D27,R2,R2
        VMOV    D28,R2,R2
        VMOV    D29,R2,R2
        VMOV    D30,R2,R2
        VMOV    D31,R2,R2
  ENDIF

        //Initialise FPSCR to a known state
        VMRS    R2,FPSCR
        LDR     R3,=0x00086060 //Mask off all bits that do not have to be preserved. Non-preserved bits can/should be zero.
        AND     R2,R2,R3
        VMSR    FPSCR,R2

        BX      LR
}
#pragma pop

/*@}*/ /* end of group CMSIS_Core_InstructionInterface */


#endif /* __CMSIS_ARMCC_H */
