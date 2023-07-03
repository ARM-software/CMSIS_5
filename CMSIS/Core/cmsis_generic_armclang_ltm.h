/**************************************************************************//**
 * @file     armclang_ltm.h
 * @brief    CMSIS compiler armclang (Arm Compiler 6) header file
 * @version  V5.0.0
 * @date     04. December 2022
 ******************************************************************************/
/*
 * Copyright (c) 2009-2022 Arm Limited. All rights reserved.
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

/*lint -esym(9058, IRQn)*/ /* disable MISRA 2012 Rule 2.4 for IRQn */

#ifndef __CMSIS_GENERIC_ARMCLANG_LTM_H
#define __CMSIS_GENERIC_ARMCLANG_LTM_H

#pragma clang system_header   /* treat file as system include file */

/* CMSIS compiler specific defines */
#ifndef   __ASM
  #define __ASM                                  __asm
#endif
#ifndef   __INLINE
  #define __INLINE                               __inline
#endif
#ifndef   __FORCEINLINE
  #define __FORCEINLINE                          __attribute__((always_inline))
#endif
#ifndef   __STATIC_INLINE
  #define __STATIC_INLINE                        static __inline
#endif
#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE                   __attribute__((always_inline)) static __inline
#endif
#ifndef   __NO_RETURN
  #define __NO_RETURN                            __attribute__((__noreturn__))
#endif
#ifndef   CMSIS_DEPRECATED
  #define CMSIS_DEPRECATED                       __attribute__((deprecated))
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
#ifndef   __PACKED_UNION
  #define __PACKED_UNION                         union __attribute__((packed, aligned(1)))
#endif
#ifndef   __UNALIGNED_UINT32        /* deprecated */
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT32)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT32 */
  struct __attribute__((packed)) T_UINT32 { uint32_t v; };
  #pragma clang diagnostic pop
  #define __UNALIGNED_UINT32(x)                  (((struct T_UINT32 *)(x))->v)
#endif
#ifndef   __UNALIGNED_UINT16_WRITE
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT16_WRITE)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT16_WRITE */
  __PACKED_STRUCT T_UINT16_WRITE { uint16_t v; };
  #pragma clang diagnostic pop
  #define __UNALIGNED_UINT16_WRITE(addr, val)    (void)((((struct T_UINT16_WRITE *)(void *)(addr))->v) = (val))
#endif
#ifndef   __UNALIGNED_UINT16_READ
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT16_READ)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT16_READ */
  __PACKED_STRUCT T_UINT16_READ { uint16_t v; };
  #pragma clang diagnostic pop
  #define __UNALIGNED_UINT16_READ(addr)          (((const struct T_UINT16_READ *)(const void *)(addr))->v)
#endif
#ifndef   __UNALIGNED_UINT32_WRITE
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT32_WRITE)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT32_WRITE */
  __PACKED_STRUCT T_UINT32_WRITE { uint32_t v; };
  #pragma clang diagnostic pop
  #define __UNALIGNED_UINT32_WRITE(addr, val)    (void)((((struct T_UINT32_WRITE *)(void *)(addr))->v) = (val))
#endif
#ifndef   __UNALIGNED_UINT32_READ
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpacked"
/*lint -esym(9058, T_UINT32_READ)*/ /* disable MISRA 2012 Rule 2.4 for T_UINT32_READ */
  __PACKED_STRUCT T_UINT32_READ { uint32_t v; };
  #pragma clang diagnostic pop
  #define __UNALIGNED_UINT32_READ(addr)          (((const struct T_UINT32_READ *)(const void *)(addr))->v)
#endif
#ifndef   __ALIGNED
  #define __ALIGNED(x)                           __attribute__((aligned(x)))
#endif
#ifndef   __RESTRICT
  #define __RESTRICT                             __restrict
#endif
#ifndef   __COMPILER_BARRIER
  #define __COMPILER_BARRIER()                   __ASM volatile("":::"memory")
#endif
#ifndef __NO_INIT
  #define __NO_INIT                              __attribute__ ((section (".bss.noinit")))
#endif
#ifndef __ALIAS
  #define __ALIAS(x)                             __attribute__ ((alias(x)))
#endif

/* ##########################  Core Instruction Access  ######################### */
/** \defgroup CMSIS_Core_InstructionInterface CMSIS Core Instruction Interface
  Access to dedicated instructions
  @{
*/

/* Define macros for porting to both thumb1 and thumb2.
 * For thumb1, use low register (r0-r7), specified by constraint "l"
 * Otherwise, use general registers, specified by constraint "r" */
#if !defined (__arm__) && defined (__thumb__) && !defined (__thumb2__)
  #define __CMSIS_GCC_OUT_REG(r) "=l" (r)
  #define __CMSIS_GCC_RW_REG(r) "+l" (r)
  #define __CMSIS_GCC_USE_REG(r) "l" (r)
#else
  #define __CMSIS_GCC_OUT_REG(r) "=r" (r)
  #define __CMSIS_GCC_RW_REG(r) "+r" (r)
  #define __CMSIS_GCC_USE_REG(r) "r" (r)
#endif

/**
  \brief   No Operation
  \details No Operation does nothing. This instruction can be used for code alignment purposes.
 */
#define __NOP          __builtin_arm_nop

/**
  \brief   Wait For Interrupt
  \details Wait For Interrupt is a hint instruction that suspends execution until one of a number of events occurs.
 */
#define __WFI          __builtin_arm_wfi


/**
  \brief   Wait For Event
  \details Wait For Event is a hint instruction that permits the processor to enter
           a low-power state until one of a number of events occurs.
 */
#define __WFE          __builtin_arm_wfe


/**
  \brief   Send Event
  \details Send Event is a hint instruction. It causes an event to be signaled to the CPU.
 */
#define __SEV          __builtin_arm_sev


/**
  \brief   Instruction Synchronization Barrier
  \details Instruction Synchronization Barrier flushes the pipeline in the processor,
           so that all instructions following the ISB are fetched from cache or memory,
           after the instruction has been completed.
 */
#define __ISB()        __builtin_arm_isb(0xF)

/**
  \brief   Data Synchronization Barrier
  \details Acts as a special kind of Data Memory Barrier.
           It completes when all explicit memory accesses before this instruction complete.
 */
#define __DSB()        __builtin_arm_dsb(0xF)


/**
  \brief   Data Memory Barrier
  \details Ensures the apparent order of the explicit memory operations before
           and after the instruction, without ensuring their completion.
 */
#define __DMB()        __builtin_arm_dmb(0xF)


/**
  \brief   Reverse byte order (32 bit)
  \details Reverses the byte order in unsigned integer value. For example, 0x12345678 becomes 0x78563412.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REV(value)   __builtin_bswap32(value)


/**
  \brief   Reverse byte order (16 bit)
  \details Reverses the byte order within each halfword of a word. For example, 0x12345678 becomes 0x34127856.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REV16(value) __ROR(__REV(value), 16)


/**
  \brief   Reverse byte order (16 bit)
  \details Reverses the byte order in a 16-bit value and returns the signed 16-bit result. For example, 0x0080 becomes 0x8000.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REVSH(value) (int16_t)__builtin_bswap16(value)


/**
  \brief   Rotate Right in unsigned value (32 bit)
  \details Rotate Right (immediate) provides the value of the contents of a register rotated by a variable number of bits.
  \param [in]    op1  Value to rotate
  \param [in]    op2  Number of Bits to rotate
  \return               Rotated value
 */
__STATIC_FORCEINLINE uint32_t __ROR(uint32_t op1, uint32_t op2)
{
  op2 %= 32U;
  if (op2 == 0U)
  {
    return op1;
  }
  return (op1 >> op2) | (op1 << (32U - op2));
}

/**
  \brief   Breakpoint
  \details Causes the processor to enter Debug state.
           Debug tools can use this to investigate system state when the instruction at a particular address is reached.
  \param [in]    value  is ignored by the processor.
                 If required, a debugger can use it to store additional information about the breakpoint.
 */
#define __BKPT(value)     __ASM volatile ("bkpt "#value)

/**
  \brief   Reverse bit order of value
  \details Reverses the bit order of the given value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __RBIT            __builtin_arm_rbit

/**
  \brief   Count leading zeros
  \details Counts the number of leading zeros of a data value.
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
__STATIC_FORCEINLINE uint8_t __CLZ(uint32_t value)
{
  /* Even though __builtin_clz produces a CLZ instruction on ARM, formally
     __builtin_clz(0) is undefined behaviour, so handle this case specially.
     This guarantees ARM-compatible results if happening to compile on a non-ARM
     target, and ensures the compiler doesn't decide to activate any
     optimisations using the logic "value was passed to __builtin_clz, so it
     is non-zero".
     ARM Compiler 6.10 and possibly earlier will optimise this test away, leaving a
     single CLZ instruction.
   */
  if (value == 0U)
  {
    return 32U;
  }
  return __builtin_clz(value);
}


#if ( defined (__arm__               ) || \
     (defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
     (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
     (defined (__ARM_ARCH_8M_BASE__  ) && (__ARM_ARCH_8M_BASE__   == 1)) || \
     (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     )
  /**
    \brief   LDR Exclusive (8 bit)
    \details Executes a exclusive LDR instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  #define __LDREXB        (uint8_t)__builtin_arm_ldrex

  /**
    \brief   LDR Exclusive (16 bit)
    \details Executes a exclusive LDR instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  #define __LDREXH        (uint16_t)__builtin_arm_ldrex

  /**
    \brief   LDR Exclusive (32 bit)
    \details Executes a exclusive LDR instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  #define __LDREXW        (uint32_t)__builtin_arm_ldrex

  /**
    \brief   STR Exclusive (8 bit)
    \details Executes a exclusive STR instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  #define __STREXB        (uint32_t)__builtin_arm_strex

  /**
    \brief   STR Exclusive (16 bit)
    \details Executes a exclusive STR instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  #define __STREXH        (uint32_t)__builtin_arm_strex

  /**
    \brief   STR Exclusive (32 bit)
    \details Executes a exclusive STR instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  #define __STREXW        (uint32_t)__builtin_arm_strex

  /**
    \brief   Remove the exclusive lock
    \details Removes the exclusive lock which is created by LDREX.
   */
  #define __CLREX             __builtin_arm_clrex
#endif /* ( defined (__arm__               ) || \
           (defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
           (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
           (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
           (defined (__ARM_ARCH_8M_BASE__  ) && (__ARM_ARCH_8M_BASE__   == 1)) || \
           (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     ) */

#if ( defined (__arm__               ) || \
     (defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
     (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
     (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     )

  /**
    \brief   Signed Saturate
    \details Saturates a signed value.
    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (1..32)
    \return             Saturated value
   */
  #define __SSAT             __builtin_arm_ssat

  /**
    \brief   Unsigned Saturate
    \details Saturates an unsigned value.
    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (0..31)
    \return             Saturated value
   */
  #define __USAT             __builtin_arm_usat

  /**
    \brief   Rotate Right with Extend (32 bit)
    \details Moves each bit of a bitstring right by one bit.
             The carry input is shifted in at the left end of the bitstring.
    \param [in]    value  Value to rotate
    \return               Rotated value
   */
  __STATIC_FORCEINLINE uint32_t __RRX(uint32_t value)
  {
    uint32_t result;
  
    __ASM volatile ("rrx %0, %1" : __CMSIS_GCC_OUT_REG (result) : __CMSIS_GCC_USE_REG (value) );
    return(result);
  }


  /**
    \brief   LDRT Unprivileged (8 bit)
    \details Executes a Unprivileged LDRT instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  __STATIC_FORCEINLINE uint8_t __LDRBT(volatile uint8_t *ptr)
  {
    uint32_t result;

    __ASM volatile ("ldrbt %0, %1" : "=r" (result) : "Q" (*ptr) );
    return ((uint8_t) result);    /* Add explicit type cast here */
  }

  /**
    \brief   LDRT Unprivileged (16 bit)
    \details Executes a Unprivileged LDRT instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  __STATIC_FORCEINLINE uint16_t __LDRHT(volatile uint16_t *ptr)
  {
    uint32_t result;

    __ASM volatile ("ldrht %0, %1" : "=r" (result) : "Q" (*ptr) );
    return ((uint16_t) result);    /* Add explicit type cast here */
  }

  /**
    \brief   LDRT Unprivileged (32 bit)
    \details Executes a Unprivileged LDRT instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  __STATIC_FORCEINLINE uint32_t __LDRT(volatile uint32_t *ptr)
  {
    uint32_t result;

    __ASM volatile ("ldrt %0, %1" : "=r" (result) : "Q" (*ptr) );
    return(result);
  }

  /**
    \brief   STRT Unprivileged (8 bit)
    \details Executes a Unprivileged STRT instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __STATIC_FORCEINLINE void __STRBT(uint8_t value, volatile uint8_t *ptr)
  {
    __ASM volatile ("strbt %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
  }

  /**
    \brief   STRT Unprivileged (16 bit)
    \details Executes a Unprivileged STRT instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __STATIC_FORCEINLINE void __STRHT(uint16_t value, volatile uint16_t *ptr)
  {
    __ASM volatile ("strht %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
  }

  /**
    \brief   STRT Unprivileged (32 bit)
    \details Executes a Unprivileged STRT instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __STATIC_FORCEINLINE void __STRT(uint32_t value, volatile uint32_t *ptr)
  {
    __ASM volatile ("strt %1, %0" : "=Q" (*ptr) : "r" (value) );
  }
#else /* ( defined (__arm__               ) || \
          (defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
          (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
          (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
          (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     ) */
  /**
    \brief   Signed Saturate
    \details Saturates a signed value.
    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (1..32)
    \return             Saturated value
   */
  __STATIC_FORCEINLINE int32_t __SSAT(int32_t val, uint32_t sat)
  {
    if ((sat >= 1U) && (sat <= 32U))
    {
      const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
      const int32_t min = -1 - max ;
      if (val > max)
      {
        return max;
      }
      else if (val < min)
      {
        return min;
      }
    }
    return val;
  }

  /**
    \brief   Unsigned Saturate
    \details Saturates an unsigned value.
    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (0..31)
    \return             Saturated value
   */
  __STATIC_FORCEINLINE uint32_t __USAT(int32_t val, uint32_t sat)
  {
    if (sat <= 31U)
    {
      const uint32_t max = ((1U << sat) - 1U);
      if (val > (int32_t)max)
      {
        return max;
      }
      else if (val < 0)
      {
        return 0U;
      }
    }
    return (uint32_t)val;
  }

#endif /* ( defined (__arm__               ) || \
          (defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
          (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
          (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
          (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     ) */


#if ( defined (__arm__               ) || \
     (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
     (defined (__ARM_ARCH_8M_BASE__  ) && (__ARM_ARCH_8M_BASE__   == 1)) || \
     (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     )
  /**
    \brief   Load-Acquire (8 bit)
    \details Executes a LDAB instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  __STATIC_FORCEINLINE uint8_t __LDAB(volatile uint8_t *ptr)
  {
    uint32_t result;
  
    __ASM volatile ("ldab %0, %1" : "=r" (result) : "Q" (*ptr) : "memory" );
    return ((uint8_t) result);
  }

  /**
    \brief   Load-Acquire (16 bit)
    \details Executes a LDAH instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  __STATIC_FORCEINLINE uint16_t __LDAH(volatile uint16_t *ptr)
  {
    uint32_t result;
  
    __ASM volatile ("ldah %0, %1" : "=r" (result) : "Q" (*ptr) : "memory" );
    return ((uint16_t) result);
  }

  /**
    \brief   Load-Acquire (32 bit)
    \details Executes a LDA instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  __STATIC_FORCEINLINE uint32_t __LDA(volatile uint32_t *ptr)
  {
    uint32_t result;
  
    __ASM volatile ("lda %0, %1" : "=r" (result) : "Q" (*ptr) : "memory" );
    return(result);
  }

  /**
    \brief   Store-Release (8 bit)
    \details Executes a STLB instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __STATIC_FORCEINLINE void __STLB(uint8_t value, volatile uint8_t *ptr)
  {
    __ASM volatile ("stlb %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) : "memory" );
  }

  /**
    \brief   Store-Release (16 bit)
    \details Executes a STLH instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __STATIC_FORCEINLINE void __STLH(uint16_t value, volatile uint16_t *ptr)
  {
    __ASM volatile ("stlh %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) : "memory" );
  }

  /**
    \brief   Store-Release (32 bit)
    \details Executes a STL instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __STATIC_FORCEINLINE void __STL(uint32_t value, volatile uint32_t *ptr)
  {
    __ASM volatile ("stl %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) : "memory" );
  }

  /**
    \brief   Load-Acquire Exclusive (8 bit)
    \details Executes a LDAB exclusive instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  #define     __LDAEXB                 (uint8_t)__builtin_arm_ldaex

  /**
    \brief   Load-Acquire Exclusive (16 bit)
    \details Executes a LDAH exclusive instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  #define     __LDAEXH                 (uint16_t)__builtin_arm_ldaex

  /**
    \brief   Load-Acquire Exclusive (32 bit)
    \details Executes a LDA exclusive instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  #define     __LDAEX                  (uint32_t)__builtin_arm_ldaex

  /**
    \brief   Store-Release Exclusive (8 bit)
    \details Executes a STLB exclusive instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  #define     __STLEXB                 (uint32_t)__builtin_arm_stlex

  /**
    \brief   Store-Release Exclusive (16 bit)
    \details Executes a STLH exclusive instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  #define     __STLEXH                 (uint32_t)__builtin_arm_stlex

  /**
    \brief   Store-Release Exclusive (32 bit)
    \details Executes a STL exclusive instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  #define     __STLEX                  (uint32_t)__builtin_arm_stlex

#endif /* ( defined (__arm__               ) || \
           (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
           (defined (__ARM_ARCH_8M_BASE__  ) && (__ARM_ARCH_8M_BASE__   == 1)) || \
           (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     ) */

/** @}*/ /* end of group CMSIS_Core_InstructionInterface */


/* ###########################  Core Function Access  ########################### */
/** \ingroup  CMSIS_Core_FunctionInterface
    \defgroup CMSIS_Core_RegAccFunctions CMSIS Core Register Access Functions
  @{
 */

/**
  \brief   Enable IRQ Interrupts
  \details Enables IRQ interrupts by clearing special-purpose register PRIMASK.
           Can only be executed in Privileged modes.
 */
#ifndef __ARM_COMPAT_H
  __STATIC_FORCEINLINE void __enable_irq(void)
  {
    __ASM volatile ("cpsie i" : : : "memory");
  }
#endif

/**
  \brief   Disable IRQ Interrupts
  \details Disables IRQ interrupts by setting special-purpose register PRIMASK.
           Can only be executed in Privileged modes.
 */
#ifndef __ARM_COMPAT_H
  __STATIC_FORCEINLINE void __disable_irq(void)
  {
    __ASM volatile ("cpsid i" : : : "memory");
  }
#endif

#if ((defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
     (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
     (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
     (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     )
  /**
    \brief   Enable FIQ
    \details Enables FIQ interrupts by clearing special-purpose register FAULTMASK.
             Can only be executed in Privileged modes.
   */
  __STATIC_FORCEINLINE void __enable_fault_irq(void)
  {
    __ASM volatile ("cpsie f" : : : "memory");
  }

  /**
    \brief   Disable FIQ
    \details Disables FIQ interrupts by setting special-purpose register FAULTMASK.
             Can only be executed in Privileged modes.
   */
  __STATIC_FORCEINLINE void __disable_fault_irq(void)
  {
    __ASM volatile ("cpsid f" : : : "memory");
  }
#endif /* ((defined (__ARM_ARCH_7M__       ) && (__ARM_ARCH_7M__        == 1)) || \
           (defined (__ARM_ARCH_7EM__      ) && (__ARM_ARCH_7EM__       == 1)) || \
           (defined (__ARM_ARCH_8M_MAIN__  ) && (__ARM_ARCH_8M_MAIN__   == 1)) || \
           (defined (__ARM_ARCH_8_1M_MAIN__) && (__ARM_ARCH_8_1M_MAIN__ == 1))     ) */

/**
  \brief   Get FPSCR
  \details Returns the current value of the Floating Point Status/Control register.
  \return               Floating Point Status/Control register value
 */
#if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
     (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
  #define __get_FPSCR      (uint32_t)__builtin_arm_get_fpscr
#else
  #define __get_FPSCR()      ((uint32_t)0U)
#endif

/**
  \brief   Set FPSCR
  \details Assigns the given value to the Floating Point Status/Control register.
  \param [in]    fpscr  Floating Point Status/Control value to set
 */
#if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
     (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
  #define __set_FPSCR      __builtin_arm_set_fpscr
#else
  #define __set_FPSCR(fpscr)      ((void)(fpscr))
#endif

/** @} end of CMSIS_Core_RegAccFunctions */

/* ###################  Compiler specific Intrinsics  ########################### */
/** \defgroup CMSIS_SIMD_intrinsics CMSIS SIMD Intrinsics
  Access to dedicated SIMD instructions
  @{
*/

#if (defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))

__STATIC_FORCEINLINE uint32_t __SADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __QADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SHADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UQADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UHADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}


__STATIC_FORCEINLINE uint32_t __SSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("ssub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __QSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SHSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __USUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UQSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UHSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}


__STATIC_FORCEINLINE uint32_t __SADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __QADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SHADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UQADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UHADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("ssub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __QSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SHSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __USUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UQSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UHSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __QASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SHASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UQASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UHASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("ssax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __QSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SHSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __USAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UQSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UHSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __USAD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usad8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __USADA8(uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("usada8 %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

#define __SSAT16(ARG1,ARG2) \
({                          \
  int32_t __RES, __ARG1 = (ARG1); \
  __ASM ("ssat16 %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })

#define __USAT16(ARG1,ARG2) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1); \
  __ASM ("usat16 %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })

__STATIC_FORCEINLINE uint32_t __UXTB16(uint32_t op1)
{
  uint32_t result;

  __ASM volatile ("uxtb16 %0, %1" : "=r" (result) : "r" (op1));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __UXTAB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uxtab16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SXTB16(uint32_t op1)
{
  uint32_t result;

  __ASM volatile ("sxtb16 %0, %1" : "=r" (result) : "r" (op1));
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SXTAB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sxtab16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMUAD  (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smuad %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMUADX (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smuadx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMLAD (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMLADX (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smladx %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__STATIC_FORCEINLINE uint64_t __SMLALD (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlald %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlald %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__STATIC_FORCEINLINE uint64_t __SMLALDX (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlaldx %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlaldx %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__STATIC_FORCEINLINE uint32_t __SMUSD  (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smusd %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMUSDX (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smusdx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMLSD (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smlsd %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__STATIC_FORCEINLINE uint32_t __SMLSDX (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smlsdx %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__STATIC_FORCEINLINE uint64_t __SMLSLD (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlsld %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlsld %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__STATIC_FORCEINLINE uint64_t __SMLSLDX (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlsldx %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlsldx %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__STATIC_FORCEINLINE uint32_t __SEL  (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sel %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE  int32_t __QADD( int32_t op1,  int32_t op2)
{
  int32_t result;

  __ASM volatile ("qadd %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__STATIC_FORCEINLINE  int32_t __QSUB( int32_t op1,  int32_t op2)
{
  int32_t result;

  __ASM volatile ("qsub %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

  #define __PKHBT(ARG1,ARG2,ARG3)          ( ((((uint32_t)(ARG1))          ) & 0x0000FFFFUL) |  \
                                             ((((uint32_t)(ARG2)) << (ARG3)) & 0xFFFF0000UL)  )

  #define __PKHTB(ARG1,ARG2,ARG3)          ( ((((uint32_t)(ARG1))          ) & 0xFFFF0000UL) |  \
                                             ((((uint32_t)(ARG2)) >> (ARG3)) & 0x0000FFFFUL)  )

  #define __SXTB16_RORn(ARG1, ARG2)        __SXTB16(__ROR(ARG1, ARG2))

  #define __SXTAB16_RORn(ARG1, ARG2, ARG3) __SXTAB16(ARG1, __ROR(ARG2, ARG3))

  __STATIC_FORCEINLINE int32_t __SMMLA (int32_t op1, int32_t op2, int32_t op3)
  {
    int32_t result;

    __ASM volatile ("smmla %0, %1, %2, %3" : "=r" (result): "r"  (op1), "r" (op2), "r" (op3) );
    return(result);
  }
#endif /* (defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1)) */
/** @} end of group CMSIS_SIMD_intrinsics */

#endif /* __CMSIS_GENERIC_ARMCLANG_LTM_H */*/