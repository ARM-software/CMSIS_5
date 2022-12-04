/**************************************************************************//**
 * @file     iccarm.h
 * @brief    CMSIS compiler ICCARM (IAR Compiler for Arm) header file
 * @version  V5.0.0
 * @date     04. December 2022
 ******************************************************************************/

//------------------------------------------------------------------------------
//
// Copyright (c) 2017-2021 IAR Systems
// Copyright (c) 2017-2022 Arm Limited. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License")
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//------------------------------------------------------------------------------

#ifndef __CMSIS_GENERIC_ICCARM_H__
#define __CMSIS_GENERIC_ICCARM_H__

#ifndef __ICCARM__
  #error This file should only be compiled by ICCARM
#endif

#pragma system_include

#define __IAR_FT _Pragma("inline=forced") __intrinsic

#if (__VER__ >= 8000000)
  #define __ICCARM_V8 1
#else
  #define __ICCARM_V8 0
#endif

//#pragma language=extended

#ifndef __ALIGNED
  #if __ICCARM_V8
    #define __ALIGNED(x) __attribute__((aligned(x)))
  #elif (__VER__ >= 7080000)
    /* Needs IAR language extensions */
    #define __ALIGNED(x) __attribute__((aligned(x)))
  #else
    #warning No compiler specific solution for __ALIGNED.__ALIGNED is ignored.
    #define __ALIGNED(x)
  #endif
#endif


/* Define compiler macros for CPU architecture, used in CMSIS 5.
 */
#if __ARM_ARCH_6M__ || __ARM_ARCH_7M__ || __ARM_ARCH_7EM__ || __ARM_ARCH_8M_BASE__ || __ARM_ARCH_8M_MAIN__ || __ARM_ARCH_7A__
  /* Macros already defined */
#else
  #if defined(__ARM8M_MAINLINE__) || defined(__ARM8EM_MAINLINE__)
    #define __ARM_ARCH_8M_MAIN__ 1
  #elif defined(__ARM8M_BASELINE__)
    #define __ARM_ARCH_8M_BASE__ 1
  #elif defined(__ARM7A__)
    #define __ARM_ARCH_7A__ 1
  #elif defined(__ARM_ARCH_PROFILE) && __ARM_ARCH_PROFILE == 'M'
    #if __ARM_ARCH == 6
      #define __ARM_ARCH_6M__ 1
    #elif __ARM_ARCH == 7
      #if __ARM_FEATURE_DSP
        #define __ARM_ARCH_7EM__ 1
      #else
        #define __ARM_ARCH_7M__ 1
      #endif
    #endif /* __ARM_ARCH */
  #endif /* __ARM_ARCH_PROFILE == 'M' */
#endif

/* Alternativ core deduction for older ICCARM's */
#if !defined(__ARM_ARCH_6M__) && !defined(__ARM_ARCH_7M__) && !defined(__ARM_ARCH_7EM__) && \
    !defined(__ARM_ARCH_8M_BASE__) && !defined(__ARM_ARCH_8M_MAIN__)
  #if defined(__ARM6M__) && (__CORE__ == __ARM6M__)
    #define __ARM_ARCH_6M__ 1
  #elif defined(__ARM7M__) && (__CORE__ == __ARM7M__)
    #define __ARM_ARCH_7M__ 1
  #elif defined(__ARM7EM__) && (__CORE__ == __ARM7EM__)
    #define __ARM_ARCH_7EM__  1
  #elif defined(__ARM8M_BASELINE__) && (__CORE == __ARM8M_BASELINE__)
    #define __ARM_ARCH_8M_BASE__ 1
  #elif defined(__ARM8M_MAINLINE__) && (__CORE == __ARM8M_MAINLINE__)
    #define __ARM_ARCH_8M_MAIN__ 1
  #elif defined(__ARM8EM_MAINLINE__) && (__CORE == __ARM8EM_MAINLINE__)
    #define __ARM_ARCH_8M_MAIN__ 1
  #else
    #error "Unknown target."
  #endif
#endif



#if defined(__ARM_ARCH_6M__) && __ARM_ARCH_6M__==1
  #define __IAR_M0_FAMILY  1
#elif defined(__ARM_ARCH_8M_BASE__) && __ARM_ARCH_8M_BASE__==1
  #define __IAR_M0_FAMILY  1
#else
  #define __IAR_M0_FAMILY  0
#endif


#ifndef __ASM
  #define __ASM __asm
#endif

#ifndef   __COMPILER_BARRIER
  #define __COMPILER_BARRIER() __ASM volatile("":::"memory")
#endif

#ifndef __NO_INIT
  #define __NO_INIT __attribute__ ((section (".noinit")))
#endif
#ifndef __ALIAS
  #define __ALIAS(x) __attribute__ ((alias(x)))
#endif

#ifndef __INLINE
  #define __INLINE inline
#endif

#ifndef   __NO_RETURN
  #if __ICCARM_V8
    #define __NO_RETURN __attribute__((__noreturn__))
  #else
    #define __NO_RETURN _Pragma("object_attribute=__noreturn")
  #endif
#endif

#ifndef   __PACKED
  #if __ICCARM_V8
    #define __PACKED __attribute__((packed, aligned(1)))
  #else
    /* Needs IAR language extensions */
    #define __PACKED __packed
  #endif
#endif

#ifndef   __PACKED_STRUCT
  #if __ICCARM_V8
    #define __PACKED_STRUCT struct __attribute__((packed, aligned(1)))
  #else
    /* Needs IAR language extensions */
    #define __PACKED_STRUCT __packed struct
  #endif
#endif

#ifndef   __PACKED_UNION
  #if __ICCARM_V8
    #define __PACKED_UNION union __attribute__((packed, aligned(1)))
  #else
    /* Needs IAR language extensions */
    #define __PACKED_UNION __packed union
  #endif
#endif

#ifndef   __RESTRICT
  #if __ICCARM_V8
    #define __RESTRICT            __restrict
  #else
    /* Needs IAR language extensions */
    #define __RESTRICT            restrict
  #endif
#endif

#ifndef   __STATIC_INLINE
  #define __STATIC_INLINE       static inline
#endif

#ifndef   __FORCEINLINE
  #define __FORCEINLINE         _Pragma("inline=forced")
#endif

#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE  __FORCEINLINE __STATIC_INLINE
#endif

#ifndef   CMSIS_DEPRECATED
  #define CMSIS_DEPRECATED      __attribute__((deprecated))
#endif

#ifndef __UNALIGNED_UINT16_READ
  #pragma language=save
  #pragma language=extended
  __IAR_FT uint16_t __iar_uint16_read(void const *ptr)
  {
    return *(__packed uint16_t*)(ptr);
  }
  #pragma language=restore
  #define __UNALIGNED_UINT16_READ(PTR) __iar_uint16_read(PTR)
#endif


#ifndef __UNALIGNED_UINT16_WRITE
  #pragma language=save
  #pragma language=extended
  __IAR_FT void __iar_uint16_write(void const *ptr, uint16_t val)
  {
    *(__packed uint16_t*)(ptr) = val;;
  }
  #pragma language=restore
  #define __UNALIGNED_UINT16_WRITE(PTR,VAL) __iar_uint16_write(PTR,VAL)
#endif

#ifndef __UNALIGNED_UINT32_READ
  #pragma language=save
  #pragma language=extended
  __IAR_FT uint32_t __iar_uint32_read(void const *ptr)
  {
    return *(__packed uint32_t*)(ptr);
  }
  #pragma language=restore
  #define __UNALIGNED_UINT32_READ(PTR) __iar_uint32_read(PTR)
#endif

#ifndef __UNALIGNED_UINT32_WRITE
  #pragma language=save
  #pragma language=extended
  __IAR_FT void __iar_uint32_write(void const *ptr, uint32_t val)
  {
    *(__packed uint32_t*)(ptr) = val;;
  }
  #pragma language=restore
  #define __UNALIGNED_UINT32_WRITE(PTR,VAL) __iar_uint32_write(PTR,VAL)
#endif

#if !defined (__arm__)
  #ifndef __UNALIGNED_UINT32   /* deprecated */
    #pragma language=save
    #pragma language=extended
    __packed struct  __iar_u32 { uint32_t v; };
    #pragma language=restore
    #define __UNALIGNED_UINT32(PTR) (((struct __iar_u32 *)(PTR))->v)
  #endif
#endif

#ifndef   __USED
  #if __ICCARM_V8
    #define __USED __attribute__((used))
  #else
    #define __USED _Pragma("__root")
  #endif
#endif

#undef __WEAK                           /* undo the definition from DLib_Defaults.h */
#ifndef   __WEAK
  #if __ICCARM_V8
    #define __WEAK __attribute__((weak))
  #else
    #define __WEAK _Pragma("__weak")
  #endif
#endif

#ifndef __ICCARM_INTRINSICS_VERSION__
  #define __ICCARM_INTRINSICS_VERSION__  0
#endif

#if __ICCARM_INTRINSICS_VERSION__ == 2
  #if defined(__CLZ)
    #undef __CLZ
  #endif
  #if defined(__REVSH)
    #undef __REVSH
  #endif
  #if defined(__RBIT)
    #undef __RBIT
  #endif
  #if defined(__SSAT)
    #undef __SSAT
  #endif
  #if defined(__USAT)
    #undef __USAT
  #endif

  #include "iccarm_builtin.h"
#else /* __ICCARM_INTRINSICS_VERSION__ == 2 */
  #ifdef __INTRINSICS_INCLUDED
    #error intrinsics.h is already included previously!
  #endif

  #include <intrinsics.h>
#endif /* __ICCARM_INTRINSICS_VERSION__ == 2 */

#if __ICCARM_INTRINSICS_VERSION__ == 2
  /* ##########################  Core Instruction Access  ######################### */
  /** \defgroup CMSIS_Core_InstructionInterface CMSIS Core Instruction Interface
    Access to dedicated instructions
    @{
  */

  /**
    \brief   No Operation
    \details No Operation does nothing. This instruction can be used for code alignment purposes.
   */
  #define __NOP     __iar_builtin_no_operation
  
  /**
    \brief   Wait For Interrupt
    \details Wait For Interrupt is a hint instruction that suspends execution until one of a number of events occurs.
   */
  #define __WFI     __iar_builtin_WFI
  
  /**
    \brief   Wait For Event
    \details Wait For Event is a hint instruction that permits the processor to enter
             a low-power state until one of a number of events occurs.
   */
  #define __WFE     __iar_builtin_WFE
  
  /**
    \brief   Send Event
    \details Send Event is a hint instruction. It causes an event to be signaled to the CPU.
   */
  #define __SEV     __iar_builtin_SEV
  
  /**
    \brief   Instruction Synchronization Barrier
    \details Instruction Synchronization Barrier flushes the pipeline in the processor,
             so that all instructions following the ISB are fetched from cache or memory,
             after the instruction has been completed.
   */
  #define __ISB     __iar_builtin_ISB
  
  /**
    \brief   Data Synchronization Barrier
    \details Acts as a special kind of Data Memory Barrier.
             It completes when all explicit memory accesses before this instruction complete.
   */
  #define __DSB     __iar_builtin_DSB
  
  /**
    \brief   Data Memory Barrier
    \details Ensures the apparent order of the explicit memory operations before
             and after the instruction, without ensuring their completion.
   */
  #define __DMB     __iar_builtin_DMB
  
  /**
    \brief   Reverse byte order (32 bit)
    \details Reverses the byte order in unsigned integer value. For example, 0x12345678 becomes 0x78563412.
    \param [in]    value  Value to reverse
    \return               Reversed value
   */
  #define __REV     __iar_builtin_REV
  
  /**
    \brief   Reverse byte order (16 bit)
    \details Reverses the byte order within each halfword of a word. For example, 0x12345678 becomes 0x34127856.
    \param [in]    value  Value to reverse
    \return               Reversed value
   */
  #define __REV16   __iar_builtin_REV16

  /**
    \brief   Reverse byte order (16 bit)
    \details Reverses the byte order in a 16-bit value and returns the signed 16-bit result. For example, 0x0080 becomes 0x8000.
    \param [in]    value  Value to reverse
    \return               Reversed value
   */
  __IAR_FT int16_t __REVSH(int16_t val)
  {
    return (int16_t) __iar_builtin_REVSH(val);
  }
  
  /**
    \brief   Rotate Right in unsigned value (32 bit)
    \details Rotate Right (immediate) provides the value of the contents of a register rotated by a variable number of bits.
    \param [in]    op1  Value to rotate
    \param [in]    op2  Number of Bits to rotate
    \return               Rotated value
   */
    #define __ROR     __iar_builtin_ROR
#else
  __IAR_FT uint32_t __ROR(uint32_t op1, uint32_t op2)
  {
    return (op1 >> op2) | (op1 << ((sizeof(op1)*8)-op2));
  }
#endif

/**
  \brief   Breakpoint
  \details Causes the processor to enter Debug state.
           Debug tools can use this to investigate system state when the instruction at a particular address is reached.
  \param [in]    value  is ignored by the processor.
                 If required, a debugger can use it to store additional information about the breakpoint.
 */
#define __BKPT(value)    __asm volatile ("BKPT     %0" : : "i"(value))

/**
  \brief   Reverse bit order of value
  \details Reverses the bit order of the given value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __RBIT    __iar_builtin_RBIT
#elif __IAR_M0_FAMILY
  __STATIC_INLINE uint32_t __RBIT(uint32_t v)
  {
    uint8_t sc = 31U;
    uint32_t r = v;
    for (v >>= 1U; v; v >>= 1U)
    {
      r <<= 1U;
      r |= v & 1U;
      sc--;
    }
    return (r << sc);
  }
#endif

/**
  \brief   Count leading zeros
  \details Counts the number of leading zeros of a data value.
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __CLZ     __iar_builtin_CLZ
#elif __IAR_M0_FAMILY
  __STATIC_INLINE uint8_t __CLZ(uint32_t data)
  {
    if (data == 0U) { return 32U; }

    uint32_t count = 0U;
    uint32_t mask = 0x80000000U;

    while ((data & mask) == 0U)
    {
      count += 1U;
      mask = mask >> 1U;
    }
    return count;
  }
#endif

/**
  \brief   LDR Exclusive (8 bit)
  \details Executes a exclusive LDR instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
#define __LDREXB  __iar_builtin_LDREXB

/**
  \brief   LDR Exclusive (16 bit)
  \details Executes a exclusive LDR instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
#define __LDREXH  __iar_builtin_LDREXH

/**
  \brief   LDR Exclusive (32 bit)
  \details Executes a exclusive LDR instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __LDREXW  __iar_builtin_LDREX
#elif (!defined(__ARM_ARCH_6M__) || __ARM_ARCH_6M__==0)
  __IAR_FT uint32_t __LDREXW(uint32_t volatile *ptr)
  {
    return __LDREX((unsigned long *)ptr);
  }
#endif

/**
  \brief   STR Exclusive (8 bit)
  \details Executes a exclusive STR instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define __STREXB  __iar_builtin_STREXB

/**
  \brief   STR Exclusive (16 bit)
  \details Executes a exclusive STR instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define __STREXH  __iar_builtin_STREXH

/**
  \brief   STR Exclusive (32 bit)
  \details Executes a exclusive STR instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __STREXW  __iar_builtin_STREX
#elif (!defined(__ARM_ARCH_6M__) || __ARM_ARCH_6M__==0)
  __IAR_FT uint32_t __STREXW(uint32_t value, uint32_t volatile *ptr)
  {
    return __STREX(value, (unsigned long *)ptr);
  }
#endif

/**
  \brief   Remove the exclusive lock
  \details Removes the exclusive lock which is created by LDREX.
 */
#define __CLREX   __iar_builtin_CLREX

#if !__IAR_M0_FAMILY
  /**
    \brief   Signed Saturate
    \details Saturates a signed value.
    \param [in]  ARG1  Value to be saturated
    \param [in]  ARG2  Bit position to saturate to (1..32)
    \return             Saturated value
   */
  #define __SSAT    __iar_builtin_SSAT

  /**
    \brief   Unsigned Saturate
    \details Saturates an unsigned value.
    \param [in]  ARG1  Value to be saturated
    \param [in]  ARG2  Bit position to saturate to (0..31)
    \return             Saturated value
   */
  #define __USAT    __iar_builtin_USAT
#else /* !__IAR_M0_FAMILY */
  /**
    \brief   Signed Saturate
    \details Saturates a signed value.
    \param [in]  val  Value to be saturated
    \param [in]  sat  Bit position to saturate to (1..32)
    \return           Saturated value
   */
  __STATIC_INLINE int32_t __SSAT(int32_t val, uint32_t sat)
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
    \param [in]  val  Value to be saturated
    \param [in]  sat  Bit position to saturate to (0..31)
    \return           Saturated value
   */
  __STATIC_INLINE uint32_t __USAT(int32_t val, uint32_t sat)
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
#endif /* !__IAR_M0_FAMILY */

/**
  \brief   Rotate Right with Extend (32 bit)
  \details Moves each bit of a bitstring right by one bit.
           The carry input is shifted in at the left end of the bitstring.
  \param [in]    value  Value to rotate
  \return               Rotated value
 */
#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __RRX     __iar_builtin_RRX
#elif (defined (__CORTEX_M) && __CORTEX_M >= 0x03)   /* __CORTEX_M is defined in core_cm0.h, core_cm3.h and core_cm4.h. */
  __IAR_FT uint32_t __RRX(uint32_t value)
  {
    uint32_t result;
    __ASM volatile("RRX      %0, %1" : "=r"(result) : "r" (value));
    return(result);
  }
#endif

#if (defined (__CORTEX_M) && __CORTEX_M >= 0x03)   /* __CORTEX_M is defined in core_cm0.h, core_cm3.h and core_cm4.h. */
  /**
    \brief   LDRT Unprivileged (8 bit)
    \details Executes a Unprivileged LDRT instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  __IAR_FT uint8_t __LDRBT(volatile uint8_t *addr)
  {
    uint32_t res;
    __ASM volatile ("LDRBT %0, [%1]" : "=r" (res) : "r" (addr) : "memory");
    return ((uint8_t)res);
  }

  /**
    \brief   LDRT Unprivileged (16 bit)
    \details Executes a Unprivileged LDRT instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  __IAR_FT uint16_t __LDRHT(volatile uint16_t *addr)
  {
    uint32_t res;
    __ASM volatile ("LDRHT %0, [%1]" : "=r" (res) : "r" (addr) : "memory");
    return ((uint16_t)res);
  }

  /**
    \brief   LDRT Unprivileged (32 bit)
    \details Executes a Unprivileged LDRT instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  __IAR_FT uint32_t __LDRT(volatile uint32_t *addr)
  {
    uint32_t res;
    __ASM volatile ("LDRT %0, [%1]" : "=r" (res) : "r" (addr) : "memory");
    return res;
  }

  /**
    \brief   STRT Unprivileged (8 bit)
    \details Executes a Unprivileged STRT instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __IAR_FT void __STRBT(uint8_t value, volatile uint8_t *addr)
  {
    __ASM volatile ("STRBT %1, [%0]" : : "r" (addr), "r" ((uint32_t)value) : "memory");
  }

  /**
    \brief   STRT Unprivileged (16 bit)
    \details Executes a Unprivileged STRT instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __IAR_FT void __STRHT(uint16_t value, volatile uint16_t *addr)
  {
    __ASM volatile ("STRHT %1, [%0]" : : "r" (addr), "r" ((uint32_t)value) : "memory");
  }

  /**
    \brief   STRT Unprivileged (32 bit)
    \details Executes a Unprivileged STRT instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __IAR_FT void __STRT(uint32_t value, volatile uint32_t *addr)
  {
    __ASM volatile ("STRT %1, [%0]" : : "r" (addr), "r" (value) : "memory");
  }
#endif /* (defined (__CORTEX_M) && __CORTEX_M >= 0x03) */

#if ((defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) || \
     (defined (__ARM_ARCH_8M_BASE__ ) && (__ARM_ARCH_8M_BASE__ == 1))    )
  /**
    \brief   Load-Acquire (8 bit)
    \details Executes a LDAB instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  __IAR_FT uint8_t __LDAB(volatile uint8_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("LDAB %0, [%1]" : "=r" (res) : "r" (ptr) : "memory");
    return ((uint8_t)res);
  }

  /**
    \brief   Load-Acquire (16 bit)
    \details Executes a LDAH instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  __IAR_FT uint16_t __LDAH(volatile uint16_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("LDAH %0, [%1]" : "=r" (res) : "r" (ptr) : "memory");
    return ((uint16_t)res);
  }

  /**
    \brief   Load-Acquire (32 bit)
    \details Executes a LDA instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  __IAR_FT uint32_t __LDA(volatile uint32_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("LDA %0, [%1]" : "=r" (res) : "r" (ptr) : "memory");
    return res;
  }

  /**
    \brief   Store-Release (8 bit)
    \details Executes a STLB instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __IAR_FT void __STLB(uint8_t value, volatile uint8_t *ptr)
  {
    __ASM volatile ("STLB %1, [%0]" :: "r" (ptr), "r" (value) : "memory");
  }

  /**
    \brief   Store-Release (16 bit)
    \details Executes a STLH instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __IAR_FT void __STLH(uint16_t value, volatile uint16_t *ptr)
  {
    __ASM volatile ("STLH %1, [%0]" :: "r" (ptr), "r" (value) : "memory");
  }

  /**
    \brief   Store-Release (32 bit)
    \details Executes a STL instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
   */
  __IAR_FT void __STL(uint32_t value, volatile uint32_t *ptr)
  {
    __ASM volatile ("STL %1, [%0]" :: "r" (ptr), "r" (value) : "memory");
  }

  /**
    \brief   Load-Acquire Exclusive (8 bit)
    \details Executes a LDAB exclusive instruction for 8 bit value.
    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
   */
  __IAR_FT uint8_t __LDAEXB(volatile uint8_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("LDAEXB %0, [%1]" : "=r" (res) : "r" (ptr) : "memory");
    return ((uint8_t)res);
  }

  /**
    \brief   Load-Acquire Exclusive (16 bit)
    \details Executes a LDAH exclusive instruction for 16 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
   */
  __IAR_FT uint16_t __LDAEXH(volatile uint16_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("LDAEXH %0, [%1]" : "=r" (res) : "r" (ptr) : "memory");
    return ((uint16_t)res);
  }

  /**
    \brief   Load-Acquire Exclusive (32 bit)
    \details Executes a LDA exclusive instruction for 32 bit values.
    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
   */
  __IAR_FT uint32_t __LDAEX(volatile uint32_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("LDAEX %0, [%1]" : "=r" (res) : "r" (ptr) : "memory");
    return res;
  }

  /**
    \brief   Store-Release Exclusive (8 bit)
    \details Executes a STLB exclusive instruction for 8 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  __IAR_FT uint32_t __STLEXB(uint8_t value, volatile uint8_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("STLEXB %0, %2, [%1]" : "=r" (res) : "r" (ptr), "r" (value) : "memory");
    return res;
  }

  /**
    \brief   Store-Release Exclusive (16 bit)
    \details Executes a STLH exclusive instruction for 16 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  __IAR_FT uint32_t __STLEXH(uint16_t value, volatile uint16_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("STLEXH %0, %2, [%1]" : "=r" (res) : "r" (ptr), "r" (value) : "memory");
    return res;
  }

  /**
    \brief   Store-Release Exclusive (32 bit)
    \details Executes a STL exclusive instruction for 32 bit values.
    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
   */
  __IAR_FT uint32_t __STLEX(uint32_t value, volatile uint32_t *ptr)
  {
    uint32_t res;
    __ASM volatile ("STLEX %0, %2, [%1]" : "=r" (res) : "r" (ptr), "r" (value) : "memory");
    return res;
  }
#endif /* __ARM_ARCH_8M_MAIN__ or __ARM_ARCH_8M_BASE__ */


#if __ICCARM_INTRINSICS_VERSION__ == 2 && __ARM_MEDIA__
  #define __SADD8   __iar_builtin_SADD8
  #define __QADD8   __iar_builtin_QADD8
  #define __SHADD8  __iar_builtin_SHADD8
  #define __UADD8   __iar_builtin_UADD8
  #define __UQADD8  __iar_builtin_UQADD8
  #define __UHADD8  __iar_builtin_UHADD8
  #define __SSUB8   __iar_builtin_SSUB8
  #define __QSUB8   __iar_builtin_QSUB8
  #define __SHSUB8  __iar_builtin_SHSUB8
  #define __USUB8   __iar_builtin_USUB8
  #define __UQSUB8  __iar_builtin_UQSUB8
  #define __UHSUB8  __iar_builtin_UHSUB8
  #define __SADD16  __iar_builtin_SADD16
  #define __QADD16  __iar_builtin_QADD16
  #define __SHADD16 __iar_builtin_SHADD16
  #define __UADD16  __iar_builtin_UADD16
  #define __UQADD16 __iar_builtin_UQADD16
  #define __UHADD16 __iar_builtin_UHADD16
  #define __SSUB16  __iar_builtin_SSUB16
  #define __QSUB16  __iar_builtin_QSUB16
  #define __SHSUB16 __iar_builtin_SHSUB16
  #define __USUB16  __iar_builtin_USUB16
  #define __UQSUB16 __iar_builtin_UQSUB16
  #define __UHSUB16 __iar_builtin_UHSUB16
  #define __SASX    __iar_builtin_SASX
  #define __QASX    __iar_builtin_QASX
  #define __SHASX   __iar_builtin_SHASX
  #define __UASX    __iar_builtin_UASX
  #define __UQASX   __iar_builtin_UQASX
  #define __UHASX   __iar_builtin_UHASX
  #define __SSAX    __iar_builtin_SSAX
  #define __QSAX    __iar_builtin_QSAX
  #define __SHSAX   __iar_builtin_SHSAX
  #define __USAX    __iar_builtin_USAX
  #define __UQSAX   __iar_builtin_UQSAX
  #define __UHSAX   __iar_builtin_UHSAX
  #define __USAD8   __iar_builtin_USAD8
  #define __USADA8  __iar_builtin_USADA8
  #define __SSAT16  __iar_builtin_SSAT16
  #define __USAT16  __iar_builtin_USAT16
  #define __UXTB16  __iar_builtin_UXTB16
  #define __UXTAB16 __iar_builtin_UXTAB16
  #define __SXTB16  __iar_builtin_SXTB16
  #define __SXTAB16 __iar_builtin_SXTAB16
  #define __SMUAD   __iar_builtin_SMUAD
  #define __SMUADX  __iar_builtin_SMUADX
  #define __SMMLA   __iar_builtin_SMMLA
  #define __SMLAD   __iar_builtin_SMLAD
  #define __SMLADX  __iar_builtin_SMLADX
  #define __SMLALD  __iar_builtin_SMLALD
  #define __SMLALDX __iar_builtin_SMLALDX
  #define __SMUSD   __iar_builtin_SMUSD
  #define __SMUSDX  __iar_builtin_SMUSDX
  #define __SMLSD   __iar_builtin_SMLSD
  #define __SMLSDX  __iar_builtin_SMLSDX
  #define __SMLSLD  __iar_builtin_SMLSLD
  #define __SMLSLDX __iar_builtin_SMLSLDX
  #define __SEL     __iar_builtin_SEL
  #define __QADD    __iar_builtin_QADD
  #define __QSUB    __iar_builtin_QSUB
  #define __PKHBT   __iar_builtin_PKHBT
  #define __PKHTB   __iar_builtin_PKHTB
#endif /* __ICCARM_INTRINSICS_VERSION__ == 2 && __ARM_MEDIA__*/
#endif /* __CMSIS_GENERIC_ICCARM_H__ */
