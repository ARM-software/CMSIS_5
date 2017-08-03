/**************************************************************************//**
 * @file     cmsis_gcc.h
 * @brief    CMSIS compiler specific macros, functions, instructions
 * @version  V1.00
 * @date     06. Jul 2017
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

#ifndef __CMSIS_GCC_H
#define __CMSIS_GCC_H

/* ignore some GCC warnings */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"

/* Fallback for __has_builtin */
#ifndef __has_builtin
  #define __has_builtin(x) (0)
#endif

/* CMSIS compiler specific defines */
#ifndef   __ASM
  #define __ASM                                  asm
#endif
#ifndef   __INLINE
  #define __INLINE                               inline
#endif
#ifndef   __STATIC_INLINE
  #define __STATIC_INLINE                        static inline
#endif
#ifndef   __NO_RETURN
  #define __NO_RETURN                            __attribute__((noreturn))
#endif
#ifndef   __USED
  #define __USED                                 __attribute__((used))
#endif
#ifndef   __WEAK
  #define __WEAK                                 __attribute__((weak))
#endif
#ifndef   __PACKED
  #define __PACKED                               __attribute__((packed, aligned(1)))
#endif
#ifndef   __PACKED_STRUCT
  #define __PACKED_STRUCT                        struct __attribute__((packed, aligned(1)))
#endif
#ifndef   __UNALIGNED_UINT16_WRITE
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT16_WRITE)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT16_WRITE */
  __PACKED_STRUCT T_UINT16_WRITE { uint16_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT16_WRITE(addr, val)    (void)((((struct T_UINT16_WRITE *)(void *)(addr))->v) = (val))
#endif
#ifndef   __UNALIGNED_UINT16_READ
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT16_READ)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT16_READ */
  __PACKED_STRUCT T_UINT16_READ { uint16_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT16_READ(addr)          (((const struct T_UINT16_READ *)(const void *)(addr))->v)
#endif
#ifndef   __UNALIGNED_UINT32_WRITE
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT32_WRITE)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT32_WRITE */
  __PACKED_STRUCT T_UINT32_WRITE { uint32_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT32_WRITE(addr, val)    (void)((((struct T_UINT32_WRITE *)(void *)(addr))->v) = (val))
#endif
#ifndef   __UNALIGNED_UINT32_READ
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
  __PACKED_STRUCT T_UINT32_READ { uint32_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT32_READ(addr)          (((const struct T_UINT32_READ *)(const void *)(addr))->v)
#endif
#ifndef   __ALIGNED
  #define __ALIGNED(x)                           __attribute__((aligned(x)))
#endif


/* ###########################  Core Function Access  ########################### */

/**
  \brief   Enable IRQ Interrupts
  \details Enables IRQ interrupts by clearing the I-bit in the CPSR.
           Can only be executed in Privileged modes.
 */
__attribute__( ( always_inline ) ) __STATIC_INLINE void __enable_irq(void)
{
  __ASM volatile ("cpsie i" : : : "memory");
}

/**
  \brief   Disable IRQ Interrupts
  \details Disables IRQ interrupts by setting the I-bit in the CPSR.
  Can only be executed in Privileged modes.
 */
__attribute__( ( always_inline ) ) __STATIC_INLINE void __disable_irq(void)
{
  __ASM volatile ("cpsid i" : : : "memory");
}

/**
  \brief   Get FPSCR
  \details Returns the current value of the Floating Point Status/Control register.
  \return Floating Point Status/Control register value
*/
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_FPSCR(void)
{
  #if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
       (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
  #if __has_builtin(__builtin_arm_get_fpscr) || (__GNUC__ > 7) || (__GNUC__ == 7 && __GNUC_MINOR__ >= 2)
    /* see https://gcc.gnu.org/ml/gcc-patches/2017-04/msg00443.html */
    return __builtin_arm_get_fpscr();
  #else
    uint32_t result;

    __ASM volatile ("VMRS %0, fpscr" : "=r" (result) );
    return(result);
  #endif
  #else
    return(0U);
  #endif
}


/**
  \brief   Set FPSCR
  \details Assigns the given value to the Floating Point Status/Control register.
  \param [in] fpscr  Floating Point Status/Control value to set
*/
__attribute__((always_inline)) __STATIC_INLINE void __set_FPSCR(uint32_t fpscr)
{
  #if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
       (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
  #if __has_builtin(__builtin_arm_set_fpscr) || (__GNUC__ > 7) || (__GNUC__ == 7 && __GNUC_MINOR__ >= 2)
    /* see https://gcc.gnu.org/ml/gcc-patches/2017-04/msg00443.html */
    __builtin_arm_set_fpscr(fpscr);
  #else
    __ASM volatile ("VMSR fpscr, %0" : : "r" (fpscr) : "vfpcc", "memory");
  #endif
  #else
    (void)fpscr;
  #endif
}

/* ##########################  Core Instruction Access  ######################### */
/**
  \brief   No Operation
 */
#define __NOP()                             __ASM volatile ("nop")

/**
  \brief   Wait For Interrupt
 */
#define __WFI()                             __ASM volatile ("wfi")

/**
  \brief   Wait For Event
 */
#define __WFE()                             __ASM volatile ("wfe")

/**
  \brief   Send Event
 */
#define __SEV()                             __ASM volatile ("sev")

/**
  \brief   Instruction Synchronization Barrier
  \details Instruction Synchronization Barrier flushes the pipeline in the processor,
           so that all instructions following the ISB are fetched from cache or memory,
           after the instruction has been completed.
 */
__attribute__((always_inline)) __STATIC_INLINE void __ISB(void)
{
  __ASM volatile ("isb 0xF":::"memory");
}


/**
  \brief   Data Synchronization Barrier
  \details Acts as a special kind of Data Memory Barrier.
           It completes when all explicit memory accesses before this instruction complete.
 */
__attribute__((always_inline)) __STATIC_INLINE void __DSB(void)
{
  __ASM volatile ("dsb 0xF":::"memory");
}

/**
  \brief   Data Memory Barrier
  \details Ensures the apparent order of the explicit memory operations before
           and after the instruction, without ensuring their completion.
 */
__attribute__((always_inline)) __STATIC_INLINE void __DMB(void)
{
  __ASM volatile ("dmb 0xF":::"memory");
}

/**
  \brief   Reverse byte order (32 bit)
  \details Reverses the byte order in integer value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __REV(uint32_t value)
{
#if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)
  return __builtin_bswap32(value);
#else
  uint32_t result;

  __ASM volatile ("rev %0, %1" : __CMSIS_GCC_OUT_REG (result) : __CMSIS_GCC_USE_REG (value) );
  return(result);
#endif
}

/**
  \brief   Reverse byte order (16 bit)
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#ifndef __NO_EMBEDDED_ASM
__attribute__((section(".rev16_text"))) __STATIC_INLINE uint32_t __REV16(uint32_t value)
{
  uint32_t result;
  __ASM volatile("rev16 %0, %1" : "=r" (result) : "r" (value));
  return result;
}
#endif

/**
  \brief   Reverse byte order in signed short value
  \details Reverses the byte order in a signed short value with sign extension to integer.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
__attribute__((always_inline)) __STATIC_INLINE int32_t __REVSH(int32_t value)
{
#if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
  return (short)__builtin_bswap16(value);
#else
  int32_t result;

  __ASM volatile ("revsh %0, %1" : __CMSIS_GCC_OUT_REG (result) : __CMSIS_GCC_USE_REG (value) );
  return(result);
#endif
}

/**
  \brief   Rotate Right in unsigned value (32 bit)
  \details Rotate Right (immediate) provides the value of the contents of a register rotated by a variable number of bits.
  \param [in]    op1  Value to rotate
  \param [in]    op2  Number of Bits to rotate
  \return               Rotated value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __ROR(uint32_t op1, uint32_t op2)
{
  return (op1 >> op2) | (op1 << (32U - op2));
}

/**
  \brief   Breakpoint
  \param [in]    value  is ignored by the processor.
                 If required, a debugger can use it to store additional information about the breakpoint.
 */
#define __BKPT(value)                       __ASM volatile ("bkpt "#value)

/**
  \brief   Reverse bit order of value
  \details Reverses the bit order of the given value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __RBIT(uint32_t value)
{
  uint32_t result;

#if ((defined (__ARM_ARCH_7M__      ) && (__ARM_ARCH_7M__      == 1)) || \
     (defined (__ARM_ARCH_7EM__     ) && (__ARM_ARCH_7EM__     == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1))    )
   __ASM volatile ("rbit %0, %1" : "=r" (result) : "r" (value) );
#else
  int32_t s = (4 /*sizeof(v)*/ * 8) - 1; /* extra shift needed at end */

  result = value;                      /* r will be reversed bits of v; first get LSB of v */
  for (value >>= 1U; value; value >>= 1U)
  {
    result <<= 1U;
    result |= value & 1U;
    s--;
  }
  result <<= s;                        /* shift when v's highest bits are zero */
#endif
  return(result);
}

/**
  \brief   Count leading zeros
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
#define __CLZ                             __builtin_clz

/**
  \brief   LDR Exclusive (8 bit)
  \details Executes a exclusive LDR instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint8_t __LDREXB(volatile uint8_t *addr)
{
    uint32_t result;

#if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
   __ASM volatile ("ldrexb %0, %1" : "=r" (result) : "Q" (*addr) );
#else
    /* Prior to GCC 4.8, "Q" will be expanded to [rx, #0] which is not
       accepted by assembler. So has to use following less efficient pattern.
    */
   __ASM volatile ("ldrexb %0, [%1]" : "=r" (result) : "r" (addr) : "memory" );
#endif
   return ((uint8_t) result);    /* Add explicit type cast here */
}


/**
  \brief   LDR Exclusive (16 bit)
  \details Executes a exclusive LDR instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint16_t __LDREXH(volatile uint16_t *addr)
{
    uint32_t result;

#if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
   __ASM volatile ("ldrexh %0, %1" : "=r" (result) : "Q" (*addr) );
#else
    /* Prior to GCC 4.8, "Q" will be expanded to [rx, #0] which is not
       accepted by assembler. So has to use following less efficient pattern.
    */
   __ASM volatile ("ldrexh %0, [%1]" : "=r" (result) : "r" (addr) : "memory" );
#endif
   return ((uint16_t) result);    /* Add explicit type cast here */
}


/**
  \brief   LDR Exclusive (32 bit)
  \details Executes a exclusive LDR instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __LDREXW(volatile uint32_t *addr)
{
    uint32_t result;

   __ASM volatile ("ldrex %0, %1" : "=r" (result) : "Q" (*addr) );
   return(result);
}


/**
  \brief   STR Exclusive (8 bit)
  \details Executes a exclusive STR instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __STREXB(uint8_t value, volatile uint8_t *addr)
{
   uint32_t result;

   __ASM volatile ("strexb %0, %2, %1" : "=&r" (result), "=Q" (*addr) : "r" ((uint32_t)value) );
   return(result);
}


/**
  \brief   STR Exclusive (16 bit)
  \details Executes a exclusive STR instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __STREXH(uint16_t value, volatile uint16_t *addr)
{
   uint32_t result;

   __ASM volatile ("strexh %0, %2, %1" : "=&r" (result), "=Q" (*addr) : "r" ((uint32_t)value) );
   return(result);
}


/**
  \brief   STR Exclusive (32 bit)
  \details Executes a exclusive STR instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __STREXW(uint32_t value, volatile uint32_t *addr)
{
   uint32_t result;

   __ASM volatile ("strex %0, %2, %1" : "=&r" (result), "=Q" (*addr) : "r" (value) );
   return(result);
}


/**
  \brief   Remove the exclusive lock
  \details Removes the exclusive lock which is created by LDREX.
 */
__attribute__((always_inline)) __STATIC_INLINE void __CLREX(void)
{
  __ASM volatile ("clrex" ::: "memory");
}

/** \brief  Get CPSR Register
    \return               CPSR Register value
 */
__STATIC_INLINE uint32_t __get_CPSR(void)
{
  uint32_t result;
  __ASM volatile("MRS %0, cpsr" : "=r" (result) );
  return(result);
}

/** \brief  Set CPSR Register
    \param [in]    cpsr  CPSR value to set
 */
__STATIC_INLINE void __set_CPSR(uint32_t cpsr)
{
__ASM volatile ("MSR cpsr, %0" : : "r" (cpsr) : "memory");
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
__STATIC_INLINE void __set_mode(uint32_t mode) {
  __ASM volatile("MSR  cpsr_c, %0" : : "r" (mode) : "memory");
}

/** \brief  Set Stack Pointer
    \param [in]    stack  Stack Pointer value to set
 */
__STATIC_INLINE void __set_SP(uint32_t stack)
{
  __ASM volatile("MOV  sp, %0" : : "r" (stack) : "memory");
}

/** \brief  Set USR/SYS Stack Pointer
    \param [in]    topOfProcStack  USR/SYS Stack Pointer value to set
 */
__STATIC_INLINE void __set_SP_usr(uint32_t topOfProcStack)
{
  __ASM volatile(
    ".preserve8         \n"
    "BIC     r0, r0, #7 \n" // ensure stack is 8-byte aligned
    "MRS     r1, cpsr   \n"
    "CPS     #0x1F      \n" // no effect in USR mode
    "MOV     sp, r0     \n"
    "MSR     cpsr_c, r1 \n" // no effect in USR mode
    "ISB"
   );
}

/** \brief  Get FPEXC
    \return               Floating Point Exception Control register value
 */
__STATIC_INLINE uint32_t __get_FPEXC(void)
{
#if (__FPU_PRESENT == 1)
  uint32_t result;
  __ASM volatile("MRS %0, fpexc" : "=r" (result) );
  return(result);
#else
  return(0);
#endif
}

/** \brief  Set FPEXC
    \param [in]    fpexc  Floating Point Exception Control value to set
 */
__STATIC_INLINE void __set_FPEXC(uint32_t fpexc)
{
#if (__FPU_PRESENT == 1)
  __ASM volatile ("MSR fpexc, %0" : : "r" (fpexc) : "memory");
#endif
}

/** \brief  Get ACTLR
    \return               Auxiliary Control register value
 */
__STATIC_INLINE uint32_t __get_ACTLR(void)
{
  uint32_t result;
  __ASM volatile("MRS %0, actlr" : "=r" (result) );
  return(result);
}

/** \brief  Set ACTLR
    \param [in]    actlr  Auxiliary Control value to set
 */
__STATIC_INLINE void __set_ACTLR(uint32_t actlr)
{
  __ASM volatile ("MSR fpexc, %0" : : "r" (actlr) : "memory");
}
/** \brief  Get CPACR
    \return               Coprocessor Access Control register value
 */
__STATIC_INLINE uint32_t __get_CPACR(void)
{
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c1, c0, 2" : "=r"(result));
  return result;
}

/** \brief  Set CPACR
    \param [in]    cpacr  Coprocessor Access Control value to set
 */
__STATIC_INLINE void __set_CPACR(uint32_t cpacr)
{
  __ASM volatile("MCR p15, 0, %0, c1, c0, 2" : : "r"(cpacr) : "memory");
}

/** \brief  Get CBAR
    \return               Configuration Base Address register value
 */
__STATIC_INLINE uint32_t __get_CBAR() {
  uint32_t result;
  __ASM volatile("MRC p15, 4, %0, c15, c0, 0" : "=r"(result));
  return result;
}

/** \brief  Get TTBR0

    This function returns the value of the Translation Table Base Register 0.

    \return               Translation Table Base Register 0 value
 */
__STATIC_INLINE uint32_t __get_TTBR0() {
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c2, c0, 0" : "=r"(result));
  return result;
}

/** \brief  Set TTBR0

    This function assigns the given value to the Translation Table Base Register 0.

    \param [in]    ttbr0  Translation Table Base Register 0 value to set
 */
__STATIC_INLINE void __set_TTBR0(uint32_t ttbr0) {
  __ASM volatile("MCR p15, 0, %0, c2, c0, 0" : : "r"(ttbr0) : "memory");
}

/** \brief  Get DACR

    This function returns the value of the Domain Access Control Register.

    \return               Domain Access Control Register value
 */
__STATIC_INLINE uint32_t __get_DACR() {
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c3, c0, 0" : "=r"(result));
  return result;
}

/** \brief  Set DACR

    This function assigns the given value to the Domain Access Control Register.

    \param [in]    dacr   Domain Access Control Register value to set
 */
__STATIC_INLINE void __set_DACR(uint32_t dacr) {
  __ASM volatile("MCR p15, 0, %0, c3, c0, 0" : : "r"(dacr) : "memory");
}

/** \brief  Set SCTLR

    This function assigns the given value to the System Control Register.

    \param [in]    sctlr  System Control Register value to set
 */
__STATIC_INLINE void __set_SCTLR(uint32_t sctlr)
{
  __ASM volatile("MCR p15, 0, %0, c1, c0, 0" : : "r"(sctlr) : "memory");
}

/** \brief  Get SCTLR
    \return               System Control Register value
 */
__STATIC_INLINE uint32_t __get_SCTLR() {
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c1, c0, 0" : "=r"(result));
  return result;
}

/** \brief  Set ACTRL
    \param [in]    actrl  Auxiliary Control Register value to set
 */
__STATIC_INLINE void __set_ACTRL(uint32_t actrl)
{
  __ASM volatile("MCR p15, 0, %0, c1, c0, 1" : : "r"(actrl) : "memory");
}

/** \brief  Get ACTRL
    \return               Auxiliary Control Register value
 */
__STATIC_INLINE uint32_t __get_ACTRL(void)
{
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c1, c0, 1" : "=r"(result));
  return result;
}

/** \brief  Get MPIDR

    This function returns the value of the Multiprocessor Affinity Register.

    \return               Multiprocessor Affinity Register value
 */
__STATIC_INLINE uint32_t __get_MPIDR(void)
{
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c0, c0, 5" : "=r"(result));
  return result;
}

 /** \brief  Get VBAR

    This function returns the value of the Vector Base Address Register.

    \return               Vector Base Address Register
 */
__STATIC_INLINE uint32_t __get_VBAR(void)
{
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c12, c0, 0" : "=r"(result));
  return result;
}

/** \brief  Set VBAR

    This function assigns the given value to the Vector Base Address Register.

    \param [in]    vbar  Vector Base Address Register value to set
 */
__STATIC_INLINE void __set_VBAR(uint32_t vbar)
{
  __ASM volatile("MCR p15, 0, %0, c12, c0, 1" : : "r"(vbar) : "memory");
}

/** \brief  Set CNTFRQ

  This function assigns the given value to PL1 Physical Timer Counter Frequency Register (CNTFRQ).

  \param [in]    value  CNTFRQ Register value to set
*/
__STATIC_INLINE void __set_CNTFRQ(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c14, c0, 0" : : "r"(value) : "memory");
}

/** \brief  Set CNTP_TVAL

  This function assigns the given value to PL1 Physical Timer Value Register (CNTP_TVAL).

  \param [in]    value  CNTP_TVAL Register value to set
*/
__STATIC_INLINE void __set_CNTP_TVAL(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c14, c2, 0" : : "r"(value) : "memory");
}

/** \brief  Get CNTP_TVAL

    This function returns the value of the PL1 Physical Timer Value Register (CNTP_TVAL).

    \return               CNTP_TVAL Register value
 */
__STATIC_INLINE uint32_t __get_CNTP_TVAL() {
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c14, c2, 0" : "=r"(result));
  return result;
}

/** \brief  Set CNTP_CTL

  This function assigns the given value to PL1 Physical Timer Control Register (CNTP_CTL).

  \param [in]    value  CNTP_CTL Register value to set
*/
__STATIC_INLINE void __set_CNTP_CTL(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c14, c2, 1" : : "r"(value) : "memory");
}

/** \brief  Get CNTP_CTL register
    \return               CNTP_CTL Register value
 */
__STATIC_INLINE uint32_t __get_CNTP_CTL() {
  uint32_t result;
  __ASM volatile("MRC p15, 0, %0, c14, c2, 1" : "=r"(result));
  return result;
}

/** \brief  Set TLBIALL

  TLB Invalidate All
 */
__STATIC_INLINE void __set_TLBIALL(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c8, c7, 0" : : "r"(value) : "memory");
}

/** \brief  Set BPIALL.

  Branch Predictor Invalidate All
 */
__STATIC_INLINE void __set_BPIALL(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c7, c5, 6" : : "r"(value) : "memory");
}

/** \brief  Set ICIALLU

  Instruction Cache Invalidate All
 */
__STATIC_INLINE void __set_ICIALLU(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c7, c5, 0" : : "r"(value) : "memory");
}

/** \brief  Set DCCMVAC

  Data cache clean
 */
__STATIC_INLINE void __set_DCCMVAC(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c7, c10, 1" : : "r"(value) : "memory");
}

/** \brief  Set DCIMVAC

  Data cache invalidate
 */
__STATIC_INLINE void __set_DCIMVAC(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c7, c6, 1" : : "r"(value) : "memory");
}

/** \brief  Set DCCIMVAC

  Data cache clean and invalidate
 */
__STATIC_INLINE void __set_DCCIMVAC(uint32_t value) {
  __ASM volatile("MCR p15, 0, %0, c7, c14, 1" : : "r"(value) : "memory");
}


/** \brief  Set CCSIDR
 */
__STATIC_INLINE void __set_CCSIDR(uint32_t value) {
  __ASM volatile("MCR p15, 2, %0, c0, c0, 0" : : "r"(value) : "memory");
}

/** \brief  Get CCSIDR
    \return CCSIDR Register value
 */
__STATIC_INLINE uint32_t __get_CCSIDR() {
  uint32_t result;
  __ASM volatile("MRC p15, 1, %0, c0, c0, 0" : "=r"(result));
  return result;
}

/** \brief  Get CLIDR
    \return CLIDR Register value
 */
__STATIC_INLINE uint32_t __get_CLIDR() {
  uint32_t result;
  __ASM volatile("MRC p15, 1, %0, c0, c0, 1" : "=r"(result));
  return result;
}

__STATIC_INLINE int32_t log2_up(uint32_t n)
{
  int32_t log = -1;
  uint32_t t = n;
  while(t)
  {
    log++; t >>=1;
  }
  /* if n not power of 2 -> round up*/
  if ( n & (n - 1) ) log++;
  return log;
}

__STATIC_INLINE void __L1C_MaintainDCacheSetWay(uint32_t level, uint32_t maint)
{
  register volatile uint32_t Dummy;
  register volatile uint32_t ccsidr;
  uint32_t num_sets;
  uint32_t num_ways;
  uint32_t shift_way;
  uint32_t log2_linesize;
  uint32_t log2_num_ways;

  Dummy = level << 1;
  /* set csselr, select ccsidr register */
  __set_CCSIDR(Dummy);
  /* get current ccsidr register */
  ccsidr = __get_CCSIDR();
  num_sets = ((ccsidr & 0x0FFFE000) >> 13) + 1;
  num_ways = ((ccsidr & 0x00001FF8) >> 3) + 1;
  log2_linesize = (ccsidr & 0x00000007) + 2 + 2;
  log2_num_ways = log2_up(num_ways);
  shift_way = 32 - log2_num_ways;
  for(int way = num_ways-1; way >= 0; way--) {
    for(int set = num_sets-1; set >= 0; set--) {
      Dummy = (level << 1) | (set << log2_linesize) | (way << shift_way);
      switch (maint)
      {
        case 0:
          __ASM volatile("MCR p15, 0, %0, c7, c6, 2" : : "r"(Dummy) : "memory"); // DCISW. Invalidate by Set/Way
          break;

        case 1:
          __ASM volatile("MCR p15, 0, %0, c7, c10, 2" : : "r"(Dummy) : "memory"); // DCCSW. Clean by Set/Way
          break;

        default:
          __ASM volatile("MCR p15, 0, %0, c7, c14, 2" : : "r"(Dummy) : "memory"); // DCCISW. Clean and Invalidate by Set/Way
          break;

      }
    }
  }
  __DMB();
}

/** \brief  Clean and Invalidate the entire data or unified cache

  Generic mechanism for cleaning/invalidating the entire data or unified cache to the point of coherency
 */
__STATIC_INLINE void __L1C_CleanInvalidateCache(uint32_t op) {
  register volatile uint32_t clidr;
  uint32_t cache_type;
  clidr =  __get_CLIDR();
  for(uint32_t i = 0; i<7; i++)
  {
    cache_type = (clidr >> i*3) & 0x7UL;
    if ((cache_type >= 2) && (cache_type <= 4))
    {
      __L1C_MaintainDCacheSetWay(i, op);
    }
  }

}

/** \brief  Enable Floating Point Unit

  Critical section, called from undef handler, so systick is disabled
 */
__STATIC_INLINE void __FPU_Enable(void) {
  __ASM volatile(
        //Permit access to VFP/NEON, registers by modifying CPACR
    "        MRC     p15,0,R1,c1,c0,2  \n"
    "        ORR     R1,R1,#0x00F00000 \n"
    "        MCR     p15,0,R1,c1,c0,2  \n"

        //Ensure that subsequent instructions occur in the context of VFP/NEON access permitted
    "        ISB                       \n"

        //Enable VFP/NEON
    "        VMRS    R1,FPEXC          \n"
    "        ORR     R1,R1,#0x40000000 \n"
    "        VMSR    FPEXC,R1          \n"

        //Initialise VFP/NEON registers to 0
    "        MOV     R2,#0             \n"

#if TARGET_FEATURE_EXTENSION_REGISTER_COUNT >= 16
        //Initialise D16 registers to 0
    "        VMOV    D0, R2,R2         \n"
    "        VMOV    D1, R2,R2         \n"
    "        VMOV    D2, R2,R2         \n"
    "        VMOV    D3, R2,R2         \n"
    "        VMOV    D4, R2,R2         \n"
    "        VMOV    D5, R2,R2         \n"
    "        VMOV    D6, R2,R2         \n"
    "        VMOV    D7, R2,R2         \n"
    "        VMOV    D8, R2,R2         \n"
    "        VMOV    D9, R2,R2         \n"
    "        VMOV    D10,R2,R2         \n"
    "        VMOV    D11,R2,R2         \n"
    "        VMOV    D12,R2,R2         \n"
    "        VMOV    D13,R2,R2         \n"
    "        VMOV    D14,R2,R2         \n"
    "        VMOV    D15,R2,R2         \n"
#endif

#if TARGET_FEATURE_EXTENSION_REGISTER_COUNT == 32
        //Initialise D32 registers to 0
    "        VMOV    D16,R2,R2         \n"
    "        VMOV    D17,R2,R2         \n"
    "        VMOV    D18,R2,R2         \n"
    "        VMOV    D19,R2,R2         \n"
    "        VMOV    D20,R2,R2         \n"
    "        VMOV    D21,R2,R2         \n"
    "        VMOV    D22,R2,R2         \n"
    "        VMOV    D23,R2,R2         \n"
    "        VMOV    D24,R2,R2         \n"
    "        VMOV    D25,R2,R2         \n"
    "        VMOV    D26,R2,R2         \n"
    "        VMOV    D27,R2,R2         \n"
    "        VMOV    D28,R2,R2         \n"
    "        VMOV    D29,R2,R2         \n"
    "        VMOV    D30,R2,R2         \n"
    "        VMOV    D31,R2,R2         \n"
#endif
  );

  // Initialise FPSCR to a known state
  // Mask off all bits that do not have to be preserved. Non-preserved bits can/should be zero.
  __set_FPSCR(__get_FPSCR() & 0x00086060u);
}

#pragma GCC diagnostic pop

#endif /* __CMSIS_GCC_H */
