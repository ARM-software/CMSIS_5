/**************************************************************************//**
 * @file     cmsis_armcc.h
 * @brief    CMSIS compiler ARMCC (Arm Compiler 5) header file
 * @version  V5.4.0
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

#ifndef __CMSIS_ARMCC_H
#define __CMSIS_ARMCC_H

// Include the generic settigs:
#include "../../Core/cmsis_generic_armcc.h"


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
__STATIC_INLINE uint32_t __get_mode(void)
{
  return (__get_CPSR() & 0x1FU);
}

/** \brief  Set Mode
    \param [in]    mode  Mode value to set
 */
__STATIC_INLINE __ASM void __set_mode(uint32_t mode)
{
  MOV  r1, lr
  MSR  CPSR_C, r0
  BX   r1
}

/** \brief  Get Stack Pointer
    \return Stack Pointer
 */
__STATIC_INLINE __ASM uint32_t __get_SP(void)
{
  MOV  r0, sp
  BX   lr
}

/** \brief  Set Stack Pointer
    \param [in]    stack  Stack Pointer value to set
 */
__STATIC_INLINE __ASM void __set_SP(uint32_t stack)
{
  MOV  sp, r0
  BX   lr
}


/** \brief  Get USR/SYS Stack Pointer
    \return USR/SYSStack Pointer
 */
__STATIC_INLINE __ASM uint32_t __get_SP_usr(void)
{
  ARM
  PRESERVE8

  MRS     R1, CPSR
  CPS     #0x1F       ;no effect in USR mode
  MOV     R0, SP
  MSR     CPSR_c, R1  ;no effect in USR mode
  ISB
  BX      LR
}

/** \brief  Set USR/SYS Stack Pointer
    \param [in]    topOfProcStack  USR/SYS Stack Pointer value to set
 */
__STATIC_INLINE __ASM void __set_SP_usr(uint32_t topOfProcStack)
{
  ARM
  PRESERVE8

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

/*
 * Include common core functions to access Coprocessor 15 registers
 */

#define __get_CP(cp, op1, Rt, CRn, CRm, op2) do { register volatile uint32_t tmp __ASM("cp" # cp ":" # op1 ":c" # CRn ":c" # CRm ":" # op2); (Rt) = tmp; } while(0)
#define __set_CP(cp, op1, Rt, CRn, CRm, op2) do { register volatile uint32_t tmp __ASM("cp" # cp ":" # op1 ":c" # CRn ":c" # CRm ":" # op2); tmp = (Rt); } while(0)
#define __get_CP64(cp, op1, Rt, CRm) \
  do { \
    uint32_t ltmp, htmp; \
    __ASM volatile("MRRC p" # cp ", " # op1 ", ltmp, htmp, c" # CRm); \
    (Rt) = ((((uint64_t)htmp) << 32U) | ((uint64_t)ltmp)); \
  } while(0)

#define __set_CP64(cp, op1, Rt, CRm) \
  do { \
    const uint64_t tmp = (Rt); \
    const uint32_t ltmp = (uint32_t)(tmp); \
    const uint32_t htmp = (uint32_t)(tmp >> 32U); \
    __ASM volatile("MCRR p" # cp ", " # op1 ", ltmp, htmp, c" # CRm); \
  } while(0)

#include "cmsis_cp15.h"

/** \brief  Enable Floating Point Unit

  Critical section, called from undef handler, so systick is disabled
 */
__STATIC_INLINE __ASM void __FPU_Enable(void)
{
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
        VMRS    R1,FPSCR
        LDR     R2,=0x00086060 //Mask off all bits that do not have to be preserved. Non-preserved bits can/should be zero.
        AND     R1,R1,R2
        VMSR    FPSCR,R1

        BX      LR
}

#endif /* __CMSIS_ARMCC_H */
