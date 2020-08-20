/**************************************************************************//**
 * @file     cmsis_gcc.h
 * @brief    CMSIS compiler GCC header file
 * @version  V1.0.1
 * @date     20. August 2020
 ******************************************************************************/
/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
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
  #define __ASM                                  __asm
#endif
#ifndef   __INLINE
  #define __INLINE                               inline
#endif
#ifndef   __STATIC_INLINE
  #define __STATIC_INLINE                        static inline
#endif
#ifndef   __STATIC_FORCEINLINE
  #define __STATIC_FORCEINLINE                   __attribute__((always_inline)) static inline
#endif
#ifndef   __NO_RETURN
  #define __NO_RETURN                            __attribute__((__noreturn__))
#endif
#ifndef   __NAKED
  #define __NAKED                                __attribute__((naked))
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
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
  #pragma GCC diagnostic ignored "-Wattributes"
  struct __attribute__((packed)) T_UINT32 { uint32_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT32(x)                  (((struct T_UINT32 *)(x))->v)
#endif
#ifndef   __UNALIGNED_UINT16_WRITE
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
  #pragma GCC diagnostic ignored "-Wattributes"
  __PACKED_STRUCT T_UINT16_WRITE { uint16_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT16_WRITE(addr, val)    (void)((((struct T_UINT16_WRITE *)(void *)(addr))->v) = (val))
#endif
#ifndef   __UNALIGNED_UINT16_READ
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
  #pragma GCC diagnostic ignored "-Wattributes"
  __PACKED_STRUCT T_UINT16_READ { uint16_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT16_READ(addr)          (((const struct T_UINT16_READ *)(const void *)(addr))->v)
#endif
#ifndef   __UNALIGNED_UINT32_WRITE
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
  #pragma GCC diagnostic ignored "-Wattributes"
  __PACKED_STRUCT T_UINT32_WRITE { uint32_t v; };
  #pragma GCC diagnostic pop
  #define __UNALIGNED_UINT32_WRITE(addr, val)    (void)((((struct T_UINT32_WRITE *)(void *)(addr))->v) = (val))
#endif
#ifndef   __UNALIGNED_UINT32_READ
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpacked"
  #pragma GCC diagnostic ignored "-Wattributes"
  __PACKED_STRUCT T_UINT32_READ { uint32_t v; };
  #pragma GCC diagnostic pop
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

/* #########################  Startup and Lowlevel Init  ######################## */

#ifndef __EARLY_INIT
  /**
    \brief   Early system init: some useful things to be done on early stage.
    \details This default implementation.

   */
__STATIC_FORCEINLINE void __cmsis_cpu_init(void)
{

}

#define __EARLY_INIT __cmsis_cpu_init
#endif

#ifndef __PROGRAM_START

/**
  \brief   Initializes data and bss sections
  \details This default implementations initialized all data and additional bss
           sections relying on .copy.table and .zero.table specified properly
           in the used linker script.

 */
__STATIC_FORCEINLINE __NO_RETURN void __cmsis_start(void)
{
  extern void _start(void) __NO_RETURN;

  typedef struct {
    uint32_t const* src;
    uint32_t* dest;
    uint32_t  wlen;
  } __copy_table_t;

  typedef struct {
    uint32_t* dest;
    uint32_t  wlen;
  } __zero_table_t;

  extern const __copy_table_t __copy_table_start__;
  extern const __copy_table_t __copy_table_end__;
  extern const __zero_table_t __zero_table_start__;
  extern const __zero_table_t __zero_table_end__;

  for (__copy_table_t const* pTable = &__copy_table_start__; pTable < &__copy_table_end__; ++pTable) {
    for(uint32_t i=0u; i<pTable->wlen; ++i) {
      pTable->dest[i] = pTable->src[i];
    }
  }

  for (__zero_table_t const* pTable = &__zero_table_start__; pTable < &__zero_table_end__; ++pTable) {
    for(uint32_t i=0u; i<pTable->wlen; ++i) {
      pTable->dest[i] = 0u;
    }
  }

  _start();
}

#define __PROGRAM_START           __cmsis_start
#endif

#ifndef __INITIAL_SP
#define __INITIAL_SP              __EL3StackTop
#endif

#ifndef __STACK_LIMIT
#define __STACK_LIMIT             __EL3StackLimit
#endif

#ifndef __VECTOR_TABLE_EL3
#define __VECTOR_TABLE_EL3        __EL3_Vectors
#endif

#ifndef __VECTOR_TABLE_EL2
#define __VECTOR_TABLE_EL2        __EL2_Vectors
#endif

#ifndef __VECTOR_TABLE_EL1
#define __VECTOR_TABLE_EL1        __EL1_Vectors
#endif

#ifndef __VECTOR_TABLE_ATTRIBUTE
#define __VECTOR_TABLE_ATTRIBUTE  __attribute__((aligned(4096), used, section(".vectors")))
#endif

/* ###########################  Core Function Access  ########################### */
/** \ingroup  CMSIS_Core_FunctionInterface
    \defgroup CMSIS_Core_RegAccFunctions CMSIS Core Register Access Functions
  @{
 */

/*@} end of CMSIS_Core_RegAccFunctions */


/* ##########################  Core Instruction Access  ######################### */
/**
  \brief   No Operation
  \details No Operation does nothing. This instruction can be used for code alignment purposes.
 */
#define __NOP()                             __ASM volatile ("nop")

/**
  \brief   Wait For Interrupt
  \details Wait For Interrupt is a hint instruction that suspends execution until one of a number of events occurs.
 */
#define __WFI()                             __ASM volatile ("wfi":::"memory")


/**
  \brief   Wait For Event
  \details Wait For Event is a hint instruction that permits the processor to enter
           a low-power state until one of a number of events occurs.
 */
#define __WFE()                             __ASM volatile ("wfe":::"memory")


/**
  \brief   Send Event
  \details Send Event is a hint instruction. It causes an event to be signaled to the CPU.
 */
#define __SEV()                             __ASM volatile ("sev")


/**
  \brief   Instruction Synchronization Barrier
  \details Instruction Synchronization Barrier flushes the pipeline in the processor,
           so that all instructions following the ISB are fetched from cache or memory,
           after the instruction has been completed.
 */
__STATIC_FORCEINLINE void __ISB(void)
{
  __ASM volatile ("isb":::"memory");
}


/**
  \brief   Data Synchronization Barrier
  \details Acts as a special kind of Data Memory Barrier.
           It completes when all explicit memory accesses before this instruction complete.
 */
__STATIC_FORCEINLINE void __DSB(void)
{
  __ASM volatile ("dsb sy":::"memory");
}


/* ###########################  Core Function Access  ########################### */

/** \brief  Get MPIDR EL1
    \return Multiprocessor Affinity Register value
 */
__STATIC_FORCEINLINE uint64_t __get_MPIDR_EL1(void)
{
  uint64_t result;
  __ASM volatile("MRS %0, MPIDR_EL1" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Get MAIR EL3
    \return               MAIR value
 */
__STATIC_FORCEINLINE uint64_t __get_MAIR_EL3(void)
{
  uint64_t result;
  __ASM volatile("MRS  %0, mair_el3" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Set MAIR EL3
    \param [in]    mair  MAIR value to set
 */
__STATIC_FORCEINLINE void __set_MAIR_EL3(uint64_t mair)
{
  __ASM volatile("MSR  mair_el3, %0" : : "r" (mair) : "memory");
}

/** \brief  Get TCR EL3
    \return               TCR value
 */
__STATIC_FORCEINLINE uint64_t __get_TCR_EL3(void)
{
  uint64_t result;
  __ASM volatile("MRS  %0, tcr_el3" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Set TCR EL3
    \param [in]    tcr  TCR value to set
 */
__STATIC_FORCEINLINE void __set_TCR_EL3(uint64_t tcr)
{
  __ASM volatile("MSR  tcr_el3, %0" : : "r" (tcr) : "memory");
}

/** \brief  Get TTBR0 EL3
    \return               Translation Table Base Register 0 value
 */
__STATIC_FORCEINLINE uint64_t __get_TTBR0_EL3(void)
{
  uint64_t result;
  __ASM volatile("MRS  %0, ttbr0_el3" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Set TTBR0 EL3
    \param [in]    ttbr0  Translation Table Base Register 0 value to set
 */
__STATIC_FORCEINLINE void __set_TTBR0_EL3(uint64_t ttbr0)
{
  __ASM volatile("MSR  ttbr0_el3, %0" : : "r" (ttbr0) : "memory");
}

/** \brief  Get SCTLR EL3
    \return STRLR EL3 value
 */
__STATIC_FORCEINLINE uint64_t __get_SCTLR_EL3(void)
{
  uint64_t result;
  __ASM volatile("MRS  %0, sctlr_el3" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Set SCTLR EL3
    \param [in]    vbar  SCTLR value to set
 */
__STATIC_FORCEINLINE void __set_SCTLR_EL3(uint64_t sctlr)
{
  __ASM volatile("MSR  sctlr_el3, %0" : : "r" (sctlr) : "memory");
}

/** \brief  Set VBAR EL3
    \param [in]    vbar  VBAR value to set
 */
__STATIC_FORCEINLINE void __set_VBAR_EL3(uint64_t vbar)
{
  __ASM volatile("MSR  vbar_el3, %0" : : "r" (vbar) : "memory");
}

/** \brief  Set VBAR EL2
    \param [in]    vbar  VBAR value to set
 */
__STATIC_FORCEINLINE void __set_VBAR_EL2(uint64_t vbar)
{
  __ASM volatile("MSR  vbar_el2, %0" : : "r" (vbar) : "memory");
}

/** \brief  Set VBAR EL1
    \param [in]    vbar  VBAR value to set
 */
__STATIC_FORCEINLINE void __set_VBAR_EL1(uint64_t vbar)
{
  __ASM volatile("MSR  vbar_el1, %0" : : "r" (vbar) : "memory");
}

/** \brief  Get Stack Pointer
    \return Stack Pointer value
 */
__STATIC_FORCEINLINE uint64_t __get_SP(void)
{
  uint64_t result;
  __ASM volatile("MOV  %0, sp" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Set Stack Pointer
    \param [in]    stack  Stack Pointer value to set
 */
__STATIC_FORCEINLINE void __set_SP(uint64_t stack)
{
  __ASM volatile("MOV  sp, %0" : : "r" (stack) : "memory");
}


/*
 * Include common core functions to access Coprocessor 15 registers
 */

#define __get_CP(cp, op1, Rt, CRn, CRm, op2) __ASM volatile("MRC p" # cp ", " # op1 ", %0, c" # CRn ", c" # CRm ", " # op2 : "=r" (Rt) : : "memory" )
#define __set_CP(cp, op1, Rt, CRn, CRm, op2) __ASM volatile("MCR p" # cp ", " # op1 ", %0, c" # CRn ", c" # CRm ", " # op2 : : "r" (Rt) : "memory" )
#define __get_CP64(cp, op1, Rt, CRm) __ASM volatile("MRRC p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : "=r" (Rt) : : "memory" )
#define __set_CP64(cp, op1, Rt, CRm) __ASM volatile("MCRR p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : : "r" (Rt) : "memory" )

#include "cmsis_cp15.h"

/** \brief  Enable Floating Point Unit

  Critical section, called from undef handler, so systick is disabled
 */
__STATIC_INLINE void __FPU_Enable(void)
{
  __ASM volatile(
    //In AArch64, you do not need to enable access to the NEON and FP registers.
    //However, access to  the NEON and FP registers can still be trapped.

    // Disable trapping of   accessing in EL3 and EL2.
    "        MSR    CPTR_EL3, XZR    \n"
    "        MSR    CPTR_EL2, XZR    \n"

    // Disable access trapping in EL1 and EL0.
    "        MOV    X1, #(0x3 << 20) \n"

    // FPEN disables trapping to EL1.
    "        MSR    CPACR_EL1, X1    \n"

    //Ensure that subsequent instructions occur in the context of VFP/NEON access permitted
    "        ISB                       "

    : : : "cc", "x1"
  );
}

#pragma GCC diagnostic pop

#endif /* __CMSIS_GCC_H */
