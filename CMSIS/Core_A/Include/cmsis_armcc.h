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
  #define __ASM                                  __asm
#endif                                          
#ifndef   __INLINE                              
  #define __INLINE                               __inline
#endif                                          
#ifndef   __STATIC_INLINE                       
  #define __STATIC_INLINE                        static __inline
#endif                                                                                   
#ifndef   __NO_RETURN                           
  #define __NO_RETURN                            __declspec(noreturn)
#endif                                          
#ifndef   __USED                                
  #define __USED                                 __attribute__((used))
#endif                                          
#ifndef   __WEAK                                
  #define __WEAK                                 __attribute__((weak))
#endif
#ifndef   __PACKED
  #define __PACKED                               __attribute__((packed))
#endif
#ifndef   __PACKED_STRUCT
  #define __PACKED_STRUCT                        __packed struct
#endif
#ifndef   __UNALIGNED_UINT16_WRITE
  #define __UNALIGNED_UINT16_WRITE(addr, val)    ((*((__packed uint16_t *)(addr))) = (val))
#endif
#ifndef   __UNALIGNED_UINT16_READ
  #define __UNALIGNED_UINT16_READ(addr)          (*((const __packed uint16_t *)(addr)))
#endif
#ifndef   __UNALIGNED_UINT32_WRITE
  #define __UNALIGNED_UINT32_WRITE(addr, val)    ((*((__packed uint32_t *)(addr))) = (val))
#endif
#ifndef   __UNALIGNED_UINT32_READ
  #define __UNALIGNED_UINT32_READ(addr)          (*((const __packed uint32_t *)(addr)))
#endif
#ifndef   __ALIGNED
  #define __ALIGNED(x)                           __attribute__((aligned(x)))
#endif                                          
#ifndef   __PACKED                              
  #define __PACKED                               __attribute__((packed))
#endif


/* ###########################  Core Function Access  ########################### */

/**
  \brief   Get FPSCR (Floating Point Status/Control)
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
  \brief   Set FPSCR (Floating Point Status/Control)
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

/* ##########################  Core Instruction Access  ######################### */
/**
  \brief   No Operation
 */
#define __NOP                             __nop

/**
  \brief   Wait For Interrupt
 */
#define __WFI                             __wfi

/**
  \brief   Wait For Event
 */
#define __WFE                             __wfe

/**
  \brief   Send Event
 */
#define __SEV                             __sev

/**
  \brief   Instruction Synchronization Barrier
 */
#define __ISB() do {\
                   __schedule_barrier();\
                   __isb(0xF);\
                   __schedule_barrier();\
                } while (0U)

/**
  \brief   Data Synchronization Barrier
 */
#define __DSB() do {\
                   __schedule_barrier();\
                   __dsb(0xF);\
                   __schedule_barrier();\
                } while (0U)

/**
  \brief   Data Memory Barrier
 */
#define __DMB() do {\
                   __schedule_barrier();\
                   __dmb(0xF);\
                   __schedule_barrier();\
                } while (0U)

/**
  \brief   Reverse byte order (32 bit)
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REV                             __rev

/**
  \brief   Reverse byte order (16 bit)
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
  \param [in]    op1  Value to rotate
  \param [in]    op2  Number of Bits to rotate
  \return               Rotated value
 */
#define __ROR                             __ror

/**
  \brief   Breakpoint
  \param [in]    value  is ignored by the processor.
                 If required, a debugger can use it to store additional information about the breakpoint.
 */
#define __BKPT(value)                     __breakpoint(value)

/**
  \brief   Reverse bit order of value
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __RBIT                            __rbit

/**
  \brief   Count leading zeros
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
#define __CLZ                             __clz

/**
  \brief   LDR Exclusive (8 bit)
  \details Executes a exclusive LDR instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 5060020)
  #define __LDREXB(ptr)                                                        ((uint8_t ) __ldrex(ptr))
#else
  #define __LDREXB(ptr)          _Pragma("push") _Pragma("diag_suppress 3731") ((uint8_t ) __ldrex(ptr))  _Pragma("pop")
#endif

/**
  \brief   LDR Exclusive (16 bit)
  \details Executes a exclusive LDR instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 5060020)
  #define __LDREXH(ptr)                                                        ((uint16_t) __ldrex(ptr))
#else
  #define __LDREXH(ptr)          _Pragma("push") _Pragma("diag_suppress 3731") ((uint16_t) __ldrex(ptr))  _Pragma("pop")
#endif

/**
  \brief   LDR Exclusive (32 bit)
  \details Executes a exclusive LDR instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 5060020)
  #define __LDREXW(ptr)                                                        ((uint32_t ) __ldrex(ptr))
#else
  #define __LDREXW(ptr)          _Pragma("push") _Pragma("diag_suppress 3731") ((uint32_t ) __ldrex(ptr))  _Pragma("pop")
#endif

/**
  \brief   STR Exclusive (8 bit)
  \details Executes a exclusive STR instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 5060020)
  #define __STREXB(value, ptr)                                                 __strex(value, ptr)
#else
  #define __STREXB(value, ptr)   _Pragma("push") _Pragma("diag_suppress 3731") __strex(value, ptr)        _Pragma("pop")
#endif

/**
  \brief   STR Exclusive (16 bit)
  \details Executes a exclusive STR instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 5060020)
  #define __STREXH(value, ptr)                                                 __strex(value, ptr)
#else
  #define __STREXH(value, ptr)   _Pragma("push") _Pragma("diag_suppress 3731") __strex(value, ptr)        _Pragma("pop")
#endif

/**
  \brief   STR Exclusive (32 bit)
  \details Executes a exclusive STR instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 5060020)
  #define __STREXW(value, ptr)                                                 __strex(value, ptr)
#else
  #define __STREXW(value, ptr)   _Pragma("push") _Pragma("diag_suppress 3731") __strex(value, ptr)        _Pragma("pop")
#endif

/**
  \brief   Remove the exclusive lock
  \details Removes the exclusive lock which is created by LDREX.
 */
#define __CLREX                           __clrex

/** \brief  Get CPSR (Current Program Status Register)
    \return               CPSR Register value
 */
__STATIC_INLINE uint32_t __get_CPSR(void)
{
  register uint32_t __regCPSR          __ASM("cpsr");
  return(__regCPSR);
}


/** \brief  Set CPSR (Current Program Status Register)
    \param [in]    cpsr  CPSR value to set
 */
__STATIC_INLINE void __set_CPSR(uint32_t cpsr)
{
  register uint32_t __regCPSR          __ASM("cpsr");
  __regCPSR = cpsr;
}

/** \brief  Get Mode
    \return                Processor Mode
 */
__STATIC_INLINE uint32_t __get_mode(void) {
  return (__get_CPSR() & 0x1FU);
}

/** \brief  Set Mode
    \param [in]    mode  Mode value to set
 */
__STATIC_INLINE __ASM void __set_mode(uint32_t mode) {
  MOV  r1, lr
  MSR  CPSR_C, r0
  BX   r1
}

/** \brief  Set Stack Pointer 
    \param [in]    stack  Stack Pointer value to set
 */
__STATIC_INLINE __ASM void __set_SP(uint32_t stack)
{
  MOV  sp, r0
  BX   lr
}

/** \brief  Set USR/SYS Stack Pointer
    \param [in]    topOfProcStack  USR/SYS Stack Pointer value to set
 */
__STATIC_INLINE __ASM void __set_SP_usr(uint32_t topOfProcStack)
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

/** \brief  Get FPEXC (Floating Point Exception Control Register)
    \return               Floating Point Exception Control Register value
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

/** \brief  Set FPEXC (Floating Point Exception Control Register)
    \param [in]    fpexc  Floating Point Exception Control value to set
 */
__STATIC_INLINE void __set_FPEXC(uint32_t fpexc)
{
#if (__FPU_PRESENT == 1)
  register uint32_t __regfpexc         __ASM("fpexc");
  __regfpexc = (fpexc);
#endif
}

/** \brief  Get ACTLR (Auxiliary Control Register)
    \return               Auxiliary Control Register value
 */
__STATIC_INLINE uint32_t __get_ACTLR(void)
{
  register uint32_t __regACTLR         __ASM("cp15:0:c1:c0:1");
  return __regACTLR;
}

/** \brief  Set ACTLR (Auxiliary Control Register)
    \param [in]    actlr  Auxiliary Control value to set
 */
__STATIC_INLINE void __set_ACTLR(uint32_t actlr)
{
  register uint32_t __regACTLR         __ASM("cp15:0:c1:c0:1");
  __regACTLR = actlr;
}

/** \brief  Get CPACR (Coprocessor Access Control Register)
    \return               Coprocessor Access Control Register value
 */
__STATIC_INLINE uint32_t __get_CPACR(void)
{
  register uint32_t __regCPACR         __ASM("cp15:0:c1:c0:2");
  return __regCPACR;
}

/** \brief  Set CPACR (Coprocessor Access Control Register)
    \param [in]    cpacr  Coprocessor Access Control value to set
 */
__STATIC_INLINE void __set_CPACR(uint32_t cpacr)
{
  register uint32_t __regCPACR         __ASM("cp15:0:c1:c0:2");
  __regCPACR = cpacr;
}

/** \brief  Get DFSR (Data Fault Status Register)
    \return               Data Fault Status Register value
 */
__STATIC_INLINE uint32_t __get_DFSR(void)
{
  register uint32_t __regDFSR         __ASM("cp15:0:c5:c0:0");
  return __regDFSR;
}

/** \brief  Set DFSR (Data Fault Status Register)
    \param [in]    dfsr  Data Fault Status value to set
 */
__STATIC_INLINE void __set_DFSR(uint32_t dfsr)
{
  register uint32_t __regDFSR         __ASM("cp15:0:c5:c0:0");
  __regDFSR = dfsr;
}

/** \brief  Get IFSR (Instruction Fault Status Register)
    \return               Instruction Fault Status Register value
 */
__STATIC_INLINE uint32_t __get_IFSR(void)
{
  register uint32_t __regIFSR         __ASM("cp15:0:c5:c0:1");
  return __regIFSR;
}

/** \brief  Set IFSR (Instruction Fault Status Register)
    \param [in]    ifsr  Instruction Fault Status value to set
 */
__STATIC_INLINE void __set_IFSR(uint32_t ifsr)
{
  register uint32_t __regIFSR         __ASM("cp15:0:c5:c0:1");
  __regIFSR = ifsr;
}

/** \brief  Get ISR (Interrupt Status Register)
    \return               Interrupt Status Register value
 */
__STATIC_INLINE uint32_t __get_ISR(void)
{
  register uint32_t __regISR         __ASM("cp15:0:c5:c0:1");
  return __regISR;
}

/** \brief  Get CBAR (Configuration Base Address Register)
    \return               Configuration Base Address Register value
 */
__STATIC_INLINE uint32_t __get_CBAR() {
  register uint32_t __regCBAR         __ASM("cp15:4:c15:c0:0");
  return(__regCBAR);
}

/** \brief  Get TTBR0 (Translation Table Base Register 0)
    \return               Translation Table Base Register 0 value
 */
__STATIC_INLINE uint32_t __get_TTBR0() {
  register uint32_t __regTTBR0        __ASM("cp15:0:c2:c0:0");
  return(__regTTBR0);
}

/** \brief  Set TTBR0 Translation Table Base Register 0
    \param [in]    ttbr0  Translation Table Base Register 0 value to set
 */
__STATIC_INLINE void __set_TTBR0(uint32_t ttbr0) {
  register uint32_t __regTTBR0        __ASM("cp15:0:c2:c0:0");
  __regTTBR0 = ttbr0;
}

/** \brief  Get DACR (Domain Access Control Register)
    \return               Domain Access Control Register value
 */
__STATIC_INLINE uint32_t __get_DACR() {
  register uint32_t __regDACR         __ASM("cp15:0:c3:c0:0");
  return(__regDACR);
}

/** \brief  Set DACR (Domain Access Control Register)
    \param [in]    dacr   Domain Access Control Register value to set
 */
__STATIC_INLINE void __set_DACR(uint32_t dacr) {
  register uint32_t __regDACR         __ASM("cp15:0:c3:c0:0");
  __regDACR = dacr;
}

/** \brief  Set SCTLR (System Control Register).
    \param [in]    sctlr  System Control Register value to set
 */
__STATIC_INLINE void __set_SCTLR(uint32_t sctlr)
{
  register uint32_t __regSCTLR         __ASM("cp15:0:c1:c0:0");
  __regSCTLR = sctlr;
}

/** \brief  Get SCTLR (System Control Register).
    \return               System Control Register value
 */
__STATIC_INLINE uint32_t __get_SCTLR() {
  register uint32_t __regSCTLR         __ASM("cp15:0:c1:c0:0");
  return(__regSCTLR);
}

/** \brief  Set ACTRL (Auxiliary Control Register)
    \param [in]    actrl  Auxiliary Control Register value to set
 */
__STATIC_INLINE void __set_ACTRL(uint32_t actrl)
{
  register uint32_t __regACTRL         __ASM("cp15:0:c1:c0:1");
  __regACTRL = actrl;
}

/** \brief  Get ACTRL (Auxiliary Control Register)
    \return               Auxiliary Control Register value
 */
__STATIC_INLINE uint32_t __get_ACTRL(void)
{
  register uint32_t __regACTRL         __ASM("cp15:0:c1:c0:1");
  return(__regACTRL);
}

/** \brief  Get MPIDR (Multiprocessor Affinity Register)
    \return               Multiprocessor Affinity Register value
 */
__STATIC_INLINE uint32_t __get_MPIDR(void)
{
  register uint32_t __regMPIDR         __ASM("cp15:0:c0:c0:5");
  return(__regMPIDR);
}

 /** \brief  Get VBAR (Vector Base Address Register)
    \return               Vector Base Address Register
 */
__STATIC_INLINE uint32_t __get_VBAR(void)
{
  register uint32_t __regVBAR         __ASM("cp15:0:c12:c0:0");
  return(__regVBAR);
}

/** \brief  Set VBAR (Vector Base Address Register)
    \param [in]    vbar  Vector Base Address Register value to set
 */
__STATIC_INLINE void __set_VBAR(uint32_t vbar)
{
  register uint32_t __regVBAR          __ASM("cp15:0:c12:c0:0");
  __regVBAR = vbar;
}

/** \brief  Set CNTFRQ (Counter Frequency Register)
  \param [in]    value  CNTFRQ Register value to set
*/
__STATIC_INLINE void __set_CNTFRQ(uint32_t value) {
  register uint32_t __regCNTFRQ         __ASM("cp15:0:c14:c0:0");
  __regCNTFRQ = value;
}

/** \brief  Set CNTP_TVAL (PL1 Physical TimerValue Register)
  \param [in]    value  CNTP_TVAL Register value to set
*/
__STATIC_INLINE void __set_CNTP_TVAL(uint32_t value) {
  register uint32_t __regCNTP_TVAL         __ASM("cp15:0:c14:c2:0");
  __regCNTP_TVAL = value;
}

/** \brief  Get CNTP_TVAL (PL1 Physical TimerValue Register)
    \return               CNTP_TVAL Register value
 */
__STATIC_INLINE uint32_t __get_CNTP_TVAL() {
  register uint32_t __regCNTP_TVAL         __ASM("cp15:0:c14:c2:0");
  return(__regCNTP_TVAL);
}

/** \brief  Set CNTP_CTL (PL1 Physical Timer Control Register)
  \param [in]    value  CNTP_CTL Register value to set
*/
__STATIC_INLINE void __set_CNTP_CTL(uint32_t value) {
  register uint32_t __regCNTP_CTL          __ASM("cp15:0:c14:c2:1");
  __regCNTP_CTL = value;
}

/** \brief  Get CNTP_CTL register
    \return               CNTP_CTL Register value
 */
__STATIC_INLINE uint32_t __get_CNTP_CTL() {
  register uint32_t __regCNTP_CTL          __ASM("cp15:0:c14:c2:1");
  return(__regCNTP_CTL);
}

/** \brief  Set TLBIALL (Invalidate Entire Unified TLB)
 */
__STATIC_INLINE void __set_TLBIALL(uint32_t value) {
  register uint32_t __TLBIALL              __ASM("cp15:0:c8:c7:0");
  __TLBIALL = value;
}

/** \brief  Set BPIALL (Branch Predictor Invalidate All)
* \param [in] value    BPIALL value to set
*/
__STATIC_INLINE void __set_BPIALL(uint32_t value) {
  register uint32_t __BPIALL            __ASM("cp15:0:c7:c5:6");
  __BPIALL = value;
}

/** \brief  Set ICIALLU (Instruction Cache Invalidate All)
 * \param [in] value    ICIALLU value to set
 */
__STATIC_INLINE void __set_ICIALLU(uint32_t value) {
  register uint32_t __ICIALLU         __ASM("cp15:0:c7:c5:0");
  __ICIALLU = value;
}

/** \brief  Set DCCMVAC (Clean data or unified cache line by MVA to PoC)
 * \param [in] value    DCCMVAC value to set
 */
__STATIC_INLINE void __set_DCCMVAC(uint32_t value) {
  register uint32_t __DCCMVAC         __ASM("cp15:0:c7:c10:1");
  __DCCMVAC = value;
}

/** \brief  Set DCIMVAC (Invalidate data or unified cache line by MVA to PoC)
 * \param [in] value    DCIMVAC value to set
 */
__STATIC_INLINE void __set_DCIMVAC(uint32_t value) {
  register uint32_t __DCIMVAC         __ASM("cp15:0:c7:c6:1");
  __DCIMVAC = value;
}

/** \brief  Set DCCIMVAC (Clean and Invalidate data or unified cache line by MVA to PoC)
 * \param [in] value    DCCIMVAC value to set
 */
__STATIC_INLINE void __set_DCCIMVAC(uint32_t value) {
  register uint32_t __DCCIMVAC        __ASM("cp15:0:c7:c14:1");
  __DCCIMVAC = value;
}

/** \brief  Clean and Invalidate the entire data or unified cache
 * \param [in] op 0 - invalidate, 1 - clean, otherwise - invalidate and clean
 */
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

/** \brief  Enable Floating Point Unit

  Critical section, called from undef handler, so systick is disabled
 */
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

#endif /* __CMSIS_ARMCC_H */
