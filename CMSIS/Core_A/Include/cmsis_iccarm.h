/**************************************************************************//**
 * @file     cmsis_iccarm.h
 * @brief    CMSIS compiler ICCARM (IAR Compiler for Arm) header file
 * @version  V5.1.0
 * @date     04. December 2022
 ******************************************************************************/

//------------------------------------------------------------------------------
//
// Copyright (c) 2017-2018 IAR Systems
// Copyright (c) 2018-2022 Arm Limited. All rights reserved.
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


#ifndef __CMSIS_ICCARM_H__
#define __CMSIS_ICCARM_H__

// Include the generic settigs:
#include "../../Core/cmsis_generic_iccarm.h"

#ifndef __ICCARM__
  #error This file should only be compiled by ICCARM
#endif

#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __disable_fault_irq __iar_builtin_disable_fiq
  #define __disable_irq       __iar_builtin_disable_interrupt
  #define __enable_fault_irq  __iar_builtin_enable_fiq
  #define __enable_irq        __iar_builtin_enable_interrupt
  #define __arm_rsr           __iar_builtin_rsr
  #define __arm_wsr           __iar_builtin_wsr

  #if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
       (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
    #define __get_FPSCR()             (__arm_rsr("FPSCR"))
    #define __set_FPSCR(VALUE)        (__arm_wsr("FPSCR", (VALUE)))
  #else
    #define __get_FPSCR()             ( 0 )
    #define __set_FPSCR(VALUE)        ((void)VALUE)
  #endif

  #define __get_CPSR()                (__arm_rsr("CPSR"))
  #define __get_mode()                (__get_CPSR() & 0x1FU)

  #define __set_CPSR(VALUE)           (__arm_wsr("CPSR", (VALUE)))
  #define __set_mode(VALUE)           (__arm_wsr("CPSR_c", (VALUE)))


  #define __get_FPEXC()       (__arm_rsr("FPEXC"))
  #define __set_FPEXC(VALUE)    (__arm_wsr("FPEXC", VALUE))

  #define __get_CP(cp, op1, RT, CRn, CRm, op2) \
    ((RT) = __arm_rsr("p" # cp ":" # op1 ":c" # CRn ":c" # CRm ":" # op2))

  #define __set_CP(cp, op1, RT, CRn, CRm, op2) \
    (__arm_wsr("p" # cp ":" # op1 ":c" # CRn ":c" # CRm ":" # op2, (RT)))

  #define __get_CP64(cp, op1, Rt, CRm) \
    __ASM volatile("MRRC p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : "=r" (Rt) : : "memory" )

  #define __set_CP64(cp, op1, Rt, CRm) \
    __ASM volatile("MCRR p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : : "r" (Rt) : "memory" )

  #include "cmsis_cp15.h"


  #if !((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)))
    #define __get_FPSCR __cmsis_iar_get_FPSR_not_active
  #endif

  #ifdef __INTRINSICS_INCLUDED
  #error intrinsics.h is already included previously!
  #endif

  #include <intrinsics.h>

  #if !((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)))
    #define __get_FPSCR() (0)
  #endif

  #pragma diag_suppress=Pe940
  #pragma diag_suppress=Pe177

  #define __enable_irq        __enable_interrupt
  #define __disable_irq       __disable_interrupt
  #define __enable_fault_irq    __enable_fiq
  #define __disable_fault_irq   __disable_fiq
  #define __NOP               __no_operation

  #define __get_xPSR          __get_PSR

  __IAR_FT void __set_mode(uint32_t mode)
  {
    __ASM volatile("MSR  cpsr_c, %0" : : "r" (mode) : "memory");
  }

  __IAR_FT uint32_t __LDREXW(uint32_t volatile *ptr)
  {
    return __LDREX((unsigned long *)ptr);
  }

  __IAR_FT uint32_t __STREXW(uint32_t value, uint32_t volatile *ptr)
  {
    return __STREX(value, (unsigned long *)ptr);
  }


  __IAR_FT uint32_t __RRX(uint32_t value)
  {
    uint32_t result;
    __ASM("RRX      %0, %1" : "=r"(result) : "r" (value) : "cc");
    return(result);
  }


  __IAR_FT uint32_t __ROR(uint32_t op1, uint32_t op2)
  {
    return (op1 >> op2) | (op1 << ((sizeof(op1)*8)-op2));
  }

  __IAR_FT uint32_t __get_FPEXC(void)
  {
  #if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)))
    uint32_t result;
    __ASM volatile("VMRS %0, fpexc" : "=r" (result) : : "memory");
    return(result);
  #else
    return(0);
  #endif
  }

  __IAR_FT void __set_FPEXC(uint32_t fpexc)
  {
  #if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)))
    __ASM volatile ("VMSR fpexc, %0" : : "r" (fpexc) : "memory");
  #endif
  }


  #define __get_CP(cp, op1, Rt, CRn, CRm, op2) \
    __ASM volatile("MRC p" # cp ", " # op1 ", %0, c" # CRn ", c" # CRm ", " # op2 : "=r" (Rt) : : "memory" )
  #define __set_CP(cp, op1, Rt, CRn, CRm, op2) \
    __ASM volatile("MCR p" # cp ", " # op1 ", %0, c" # CRn ", c" # CRm ", " # op2 : : "r" (Rt) : "memory" )
  #define __get_CP64(cp, op1, Rt, CRm) \
    __ASM volatile("MRRC p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : "=r" (Rt) : : "memory" )
  #define __set_CP64(cp, op1, Rt, CRm) \
    __ASM volatile("MCRR p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : : "r" (Rt) : "memory" )

  #include "cmsis_cp15.h"

#endif   /* __ICCARM_INTRINSICS_VERSION__ == 2 */


__IAR_FT uint32_t __get_SP_usr(void)
{
  uint32_t cpsr;
  uint32_t result;
  __ASM volatile(
    "MRS     %0, cpsr   \n"
    "CPS     #0x1F      \n" // no effect in USR mode
    "MOV     %1, sp     \n"
    "MSR     cpsr_c, %2 \n" // no effect in USR mode
    "ISB" :  "=r"(cpsr), "=r"(result) : "r"(cpsr) : "memory"
   );
  return result;
}

__IAR_FT void __set_SP_usr(uint32_t topOfProcStack)
{
  uint32_t cpsr;
  __ASM volatile(
    "MRS     %0, cpsr   \n"
    "CPS     #0x1F      \n" // no effect in USR mode
    "MOV     sp, %1     \n"
    "MSR     cpsr_c, %2 \n" // no effect in USR mode
    "ISB" : "=r"(cpsr) : "r" (topOfProcStack), "r"(cpsr) : "memory"
   );
}

#define __get_mode()                (__get_CPSR() & 0x1FU)

__STATIC_INLINE
void __FPU_Enable(void)
{
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

#ifdef __ARM_ADVANCED_SIMD__
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

    //Initialise FPSCR to a known state
    "        VMRS    R1,FPSCR          \n"
    "        MOV32   R2,#0x00086060    \n" //Mask off all bits that do not have to be preserved. Non-preserved bits can/should be zero.
    "        AND     R1,R1,R2          \n"
    "        VMSR    FPSCR,R1          \n"
    : : : "cc", "r1", "r2"
  );
}



#undef __IAR_FT
#undef __ICCARM_V8

#pragma diag_default=Pe940
#pragma diag_default=Pe177

#endif /* __CMSIS_ICCARM_H__ */
