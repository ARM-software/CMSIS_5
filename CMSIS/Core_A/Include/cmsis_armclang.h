/**************************************************************************//**
 * @file     cmsis_armclang.h
 * @brief    CMSIS compiler armclang (Arm Compiler 6) header file
 * @version  V5.5.0
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

#ifndef __CMSIS_ARMCLANG_H
#define __CMSIS_ARMCLANG_H

#pragma clang system_header   /* treat file as system include file */

// Include the generic settigs:
#include "../../Core/cmsis_generic_armclang.h"


/* ###########################  Core Function Access  ########################### */
/** \ingroup  CMSIS_Core_FunctionInterface
    \defgroup CMSIS_Core_RegAccFunctions CMSIS Core Register Access Functions
  @{
 */

/** \brief  Get CPSR Register
    \return               CPSR Register value
 */
__STATIC_FORCEINLINE uint32_t __get_CPSR(void)
{
  uint32_t result;
  __ASM volatile("MRS %0, cpsr" : "=r" (result) );
  return(result);
}

/** \brief  Set CPSR Register
    \param [in]    cpsr  CPSR value to set
 */
__STATIC_FORCEINLINE void __set_CPSR(uint32_t cpsr)
{
  __ASM volatile ("MSR cpsr, %0" : : "r" (cpsr) : "cc", "memory");
}

/** \brief  Get Mode
    \return                Processor Mode
 */
__STATIC_FORCEINLINE uint32_t __get_mode(void)
{
  return (__get_CPSR() & 0x1FU);
}

/** \brief  Set Mode
    \param [in]    mode  Mode value to set
 */
__STATIC_FORCEINLINE void __set_mode(uint32_t mode)
{
  __ASM volatile("MSR  cpsr_c, %0" : : "r" (mode) : "memory");
}

/** \brief  Get Stack Pointer
    \return Stack Pointer value
 */
__STATIC_FORCEINLINE uint32_t __get_SP(void)
{
  uint32_t result;
  __ASM volatile("MOV  %0, sp" : "=r" (result) : : "memory");
  return result;
}

/** \brief  Set Stack Pointer
    \param [in]    stack  Stack Pointer value to set
 */
__STATIC_FORCEINLINE void __set_SP(uint32_t stack)
{
  __ASM volatile("MOV  sp, %0" : : "r" (stack) : "memory");
}

/** \brief  Get USR/SYS Stack Pointer
    \return USR/SYS Stack Pointer value
 */
__STATIC_FORCEINLINE uint32_t __get_SP_usr(void)
{
  uint32_t cpsr;
  uint32_t result;
  __ASM volatile(
    "MRS     %0, cpsr   \n"
    "CPS     #0x1F      \n" // no effect in USR mode
    "MOV     %1, sp     \n"
    "MSR     cpsr_c, %0 \n" // no effect in USR mode
    "ISB" :  "=r"(cpsr), "=r"(result) : : "memory"
   );
  return result;
}

/** \brief  Set USR/SYS Stack Pointer
    \param [in]    topOfProcStack  USR/SYS Stack Pointer value to set
 */
__STATIC_FORCEINLINE void __set_SP_usr(uint32_t topOfProcStack)
{
  uint32_t cpsr;
  __ASM volatile(
    "MRS     %0, cpsr   \n"
    "CPS     #0x1F      \n" // no effect in USR mode
    "MOV     sp, %1     \n"
    "MSR     cpsr_c, %0 \n" // no effect in USR mode
    "ISB" : "=r"(cpsr) : "r" (topOfProcStack) : "memory"
   );
}

/** \brief  Get FPEXC
    \return               Floating Point Exception Control register value
 */
__STATIC_FORCEINLINE uint32_t __get_FPEXC(void)
{
#if (__FPU_PRESENT == 1)
  uint32_t result;
  __ASM volatile("VMRS %0, fpexc" : "=r" (result) : : "memory");
  return(result);
#else
  return(0);
#endif
}

/** \brief  Set FPEXC
    \param [in]    fpexc  Floating Point Exception Control value to set
 */
__STATIC_FORCEINLINE void __set_FPEXC(uint32_t fpexc)
{
#if (__FPU_PRESENT == 1)
  __ASM volatile ("VMSR fpexc, %0" : : "r" (fpexc) : "memory");
#endif
}


/*
 * Include common core functions to access Coprocessor 15 registers
 */

#define __get_CP(cp, op1, Rt, CRn, CRm, op2) __ASM volatile("MRC p" # cp ", " # op1 ", %0, c" # CRn ", c" # CRm ", " # op2 : "=r" (Rt) : : "memory" )
#define __set_CP(cp, op1, Rt, CRn, CRm, op2) __ASM volatile("MCR p" # cp ", " # op1 ", %0, c" # CRn ", c" # CRm ", " # op2 : : "r" (Rt) : "memory" )
#define __get_CP64(cp, op1, Rt, CRm)         __ASM volatile("MRRC p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : "=r" (Rt) : : "memory" )
#define __set_CP64(cp, op1, Rt, CRm)         __ASM volatile("MCRR p" # cp ", " # op1 ", %Q0, %R0, c" # CRm  : : "r" (Rt) : "memory" )

#include "cmsis_cp15.h"

/** \brief  Enable Floating Point Unit

  Critical section, called from undef handler, so systick is disabled
 */
__STATIC_INLINE void __FPU_Enable(void)
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

#if (defined(__ARM_NEON) && (__ARM_NEON == 1))
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
    "        LDR     R2,=0x00086060    \n" //Mask off all bits that do not have to be preserved. Non-preserved bits can/should be zero.
    "        AND     R1,R1,R2          \n"
    "        VMSR    FPSCR,R1            "
    : : : "cc", "r1", "r2"
  );
}

#endif /* __CMSIS_ARMCLANG_H */
