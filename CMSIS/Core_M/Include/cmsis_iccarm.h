/**************************************************************************//**
 * @file     cmsis_iccarm.h
 * @brief    CMSIS compiler ICCARM (IAR Compiler for Arm) header file
 * @version  V5.4.0
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


#ifndef __CMSIS_ICCARM_H__
#define __CMSIS_ICCARM_H__

// Include the generic settigs:
#include "../../Core/cmsis_generic_iccarm.h"

#ifndef __ICCARM__
  #error This file should only be compiled by ICCARM
#endif


#ifndef __PROGRAM_START
  #define __PROGRAM_START           __iar_program_start
#endif

#ifndef __INITIAL_SP
  #define __INITIAL_SP              CSTACK$$Limit
#endif

#ifndef __STACK_LIMIT
  #define __STACK_LIMIT             CSTACK$$Base
#endif

#ifndef __VECTOR_TABLE
  #define __VECTOR_TABLE            __vector_table
#endif

#ifndef __VECTOR_TABLE_ATTRIBUTE
  #define __VECTOR_TABLE_ATTRIBUTE  @".intvec"
#endif

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  #ifndef __STACK_SEAL
    #define __STACK_SEAL              STACKSEAL$$Base
  #endif

  #ifndef __TZ_STACK_SEAL_SIZE
    #define __TZ_STACK_SEAL_SIZE      8U
  #endif

  #ifndef __TZ_STACK_SEAL_VALUE
    #define __TZ_STACK_SEAL_VALUE     0xFEF5EDA5FEF5EDA5ULL
  #endif

  __STATIC_FORCEINLINE void __TZ_set_STACKSEAL_S (uint32_t* stackTop) {
    *((uint64_t *)stackTop) = __TZ_STACK_SEAL_VALUE;
  }
#endif

#if __ICCARM_INTRINSICS_VERSION__ == 2
  #define __disable_fault_irq __iar_builtin_disable_fiq
  #define __disable_irq       __iar_builtin_disable_interrupt
  #define __enable_fault_irq  __iar_builtin_enable_fiq
  #define __enable_irq        __iar_builtin_enable_interrupt
  #define __arm_rsr           __iar_builtin_rsr
  #define __arm_wsr           __iar_builtin_wsr


  #define __get_APSR()                (__arm_rsr("APSR"))
  #define __get_BASEPRI()             (__arm_rsr("BASEPRI"))
  #define __get_CONTROL()             (__arm_rsr("CONTROL"))
  #define __get_FAULTMASK()           (__arm_rsr("FAULTMASK"))

  #if ((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
       (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     )
    #define __get_FPSCR()             (__arm_rsr("FPSCR"))
    #define __set_FPSCR(VALUE)        (__arm_wsr("FPSCR", (VALUE)))
  #else
    #define __get_FPSCR()             ( 0 )
    #define __set_FPSCR(VALUE)        ((void)VALUE)
  #endif

  #define __get_IPSR()                (__arm_rsr("IPSR"))
  #define __get_MSP()                 (__arm_rsr("MSP"))
  #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
       (!defined (__ARM_FEATURE_CMSE) || (__ARM_FEATURE_CMSE < 3)))
    // without main extensions, the non-secure MSPLIM is RAZ/WI
    #define __get_MSPLIM()            (0U)
  #else
    #define __get_MSPLIM()            (__arm_rsr("MSPLIM"))
  #endif
  #define __get_PRIMASK()             (__arm_rsr("PRIMASK"))
  #define __get_PSP()                 (__arm_rsr("PSP"))

  #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
       (!defined (__ARM_FEATURE_CMSE) || (__ARM_FEATURE_CMSE < 3)))
    // without main extensions, the non-secure PSPLIM is RAZ/WI
    #define __get_PSPLIM()            (0U)
  #else
    #define __get_PSPLIM()            (__arm_rsr("PSPLIM"))
  #endif

  #define __get_xPSR()                (__arm_rsr("xPSR"))

  #define __set_BASEPRI(VALUE)        (__arm_wsr("BASEPRI", (VALUE)))
  #define __set_BASEPRI_MAX(VALUE)    (__arm_wsr("BASEPRI_MAX", (VALUE)))

  __STATIC_FORCEINLINE void __set_CONTROL(uint32_t control)
  {
    __arm_wsr("CONTROL", control);
    __iar_builtin_ISB();
  }

  #define __set_FAULTMASK(VALUE)      (__arm_wsr("FAULTMASK", (VALUE)))
  #define __set_MSP(VALUE)            (__arm_wsr("MSP", (VALUE)))

  #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
       (!defined (__ARM_FEATURE_CMSE) || (__ARM_FEATURE_CMSE < 3)))
    // without main extensions, the non-secure MSPLIM is RAZ/WI
    #define __set_MSPLIM(VALUE)       ((void)(VALUE))
  #else
    #define __set_MSPLIM(VALUE)       (__arm_wsr("MSPLIM", (VALUE)))
  #endif
  #define __set_PRIMASK(VALUE)        (__arm_wsr("PRIMASK", (VALUE)))
  #define __set_PSP(VALUE)            (__arm_wsr("PSP", (VALUE)))
  #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
       (!defined (__ARM_FEATURE_CMSE) || (__ARM_FEATURE_CMSE < 3)))
    // without main extensions, the non-secure PSPLIM is RAZ/WI
    #define __set_PSPLIM(VALUE)       ((void)(VALUE))
  #else
    #define __set_PSPLIM(VALUE)       (__arm_wsr("PSPLIM", (VALUE)))
  #endif

  #define __TZ_get_CONTROL_NS()       (__arm_rsr("CONTROL_NS"))

  __STATIC_FORCEINLINE void __TZ_set_CONTROL_NS(uint32_t control)
  {
    __arm_wsr("CONTROL_NS", control);
    __iar_builtin_ISB();
  }

  #define __TZ_get_PSP_NS()           (__arm_rsr("PSP_NS"))
  #define __TZ_set_PSP_NS(VALUE)      (__arm_wsr("PSP_NS", (VALUE)))
  #define __TZ_get_MSP_NS()           (__arm_rsr("MSP_NS"))
  #define __TZ_set_MSP_NS(VALUE)      (__arm_wsr("MSP_NS", (VALUE)))
  #define __TZ_get_SP_NS()            (__arm_rsr("SP_NS"))
  #define __TZ_set_SP_NS(VALUE)       (__arm_wsr("SP_NS", (VALUE)))
  #define __TZ_get_PRIMASK_NS()       (__arm_rsr("PRIMASK_NS"))
  #define __TZ_set_PRIMASK_NS(VALUE)  (__arm_wsr("PRIMASK_NS", (VALUE)))
  #define __TZ_get_BASEPRI_NS()       (__arm_rsr("BASEPRI_NS"))
  #define __TZ_set_BASEPRI_NS(VALUE)  (__arm_wsr("BASEPRI_NS", (VALUE)))
  #define __TZ_get_FAULTMASK_NS()     (__arm_rsr("FAULTMASK_NS"))
  #define __TZ_set_FAULTMASK_NS(VALUE)(__arm_wsr("FAULTMASK_NS", (VALUE)))

  #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
       (!defined (__ARM_FEATURE_CMSE) || (__ARM_FEATURE_CMSE < 3)))
    // without main extensions, the non-secure PSPLIM is RAZ/WI
    #define __TZ_get_PSPLIM_NS()      (0U)
    #define __TZ_set_PSPLIM_NS(VALUE) ((void)(VALUE))
  #else
    #define __TZ_get_PSPLIM_NS()      (__arm_rsr("PSPLIM_NS"))
    #define __TZ_set_PSPLIM_NS(VALUE) (__arm_wsr("PSPLIM_NS", (VALUE)))
  #endif

  #define __TZ_get_MSPLIM_NS()        (__arm_rsr("MSPLIM_NS"))
  #define __TZ_set_MSPLIM_NS(VALUE)   (__arm_wsr("MSPLIM_NS", (VALUE)))


  #if (!((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
         (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     ))
    #define __get_FPSCR __cmsis_iar_get_FPSR_not_active
    #define __set_FPSCR __cmsis_iar_set_FPSR_not_active
  #endif

  #ifdef __INTRINSICS_INCLUDED
    #error intrinsics.h is already included previously!
  #endif

  #include <intrinsics.h>

  #if __IAR_M0_FAMILY
   /* Avoid clash between intrinsics.h and arm_math.h when compiling for Cortex-M0. */
    #undef __CLZ
    #undef __SSAT
    #undef __USAT
    #undef __RBIT
    #undef __get_APSR

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

    __STATIC_INLINE  uint32_t __get_APSR(void)
    {
      uint32_t res;
      __asm("MRS      %0,APSR" : "=r" (res));
      return res;
    }

  #endif /* __IAR_M0_FAMILY */
#else /* __ICCARM_INTRINSICS_VERSION__ == 2 */
  #if (!((defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)) && \
         (defined (__FPU_USED   ) && (__FPU_USED    == 1U))     ))
    #undef __get_FPSCR
    #undef __set_FPSCR
    #define __get_FPSCR()       (0)
    #define __set_FPSCR(VALUE)  ((void)VALUE)
  #endif

  #pragma diag_suppress=Pe940
  #pragma diag_suppress=Pe177

  #define __enable_irq    __enable_interrupt
  #define __disable_irq   __disable_interrupt
  #define __NOP           __no_operation

  #define __get_xPSR      __get_PSR

  /* __CORTEX_M is defined in core_cm0.h, core_cm3.h and core_cm4.h. */
  #if (defined (__CORTEX_M) && __CORTEX_M >= 0x03)
    __IAR_FT void __set_BASEPRI_MAX(uint32_t value)
    {
      __asm volatile("MSR      BASEPRI_MAX,%0"::"r" (value));
    }


    #define __enable_fault_irq  __enable_fiq
    #define __disable_fault_irq __disable_fiq
  #endif /* (__CORTEX_M >= 0x03) */

  #if ((defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) || \
       (defined (__ARM_ARCH_8M_BASE__ ) && (__ARM_ARCH_8M_BASE__ == 1))    )

   __IAR_FT uint32_t __get_MSPLIM(void)
    {
      uint32_t res;
    #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
         (!defined (__ARM_FEATURE_CMSE  ) || (__ARM_FEATURE_CMSE   < 3)))
      // without main extensions, the non-secure MSPLIM is RAZ/WI
      res = 0U;
    #else
      __asm volatile("MRS      %0,MSPLIM" : "=r" (res));
    #endif
      return res;
    }

    __IAR_FT void   __set_MSPLIM(uint32_t value)
    {
    #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
         (!defined (__ARM_FEATURE_CMSE  ) || (__ARM_FEATURE_CMSE   < 3)))
      // without main extensions, the non-secure MSPLIM is RAZ/WI
      (void)value;
    #else
      __asm volatile("MSR      MSPLIM,%0" :: "r" (value));
    #endif
    }

    __IAR_FT uint32_t __get_PSPLIM(void)
    {
      uint32_t res;
    #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
         (!defined (__ARM_FEATURE_CMSE  ) || (__ARM_FEATURE_CMSE   < 3)))
      // without main extensions, the non-secure PSPLIM is RAZ/WI
      res = 0U;
    #else
      __asm volatile("MRS      %0,PSPLIM" : "=r" (res));
    #endif
      return res;
    }

    __IAR_FT void   __set_PSPLIM(uint32_t value)
    {
    #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
         (!defined (__ARM_FEATURE_CMSE  ) || (__ARM_FEATURE_CMSE   < 3)))
      // without main extensions, the non-secure PSPLIM is RAZ/WI
      (void)value;
    #else
      __asm volatile("MSR      PSPLIM,%0" :: "r" (value));
    #endif
    }

    __IAR_FT uint32_t __TZ_get_CONTROL_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,CONTROL_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_CONTROL_NS(uint32_t value)
    {
      __asm volatile("MSR      CONTROL_NS,%0" :: "r" (value));
      __iar_builtin_ISB();
    }

    __IAR_FT uint32_t   __TZ_get_PSP_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,PSP_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_PSP_NS(uint32_t value)
    {
      __asm volatile("MSR      PSP_NS,%0" :: "r" (value));
    }

    __IAR_FT uint32_t   __TZ_get_MSP_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,MSP_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_MSP_NS(uint32_t value)
    {
      __asm volatile("MSR      MSP_NS,%0" :: "r" (value));
    }

    __IAR_FT uint32_t   __TZ_get_SP_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,SP_NS" : "=r" (res));
      return res;
    }
    __IAR_FT void   __TZ_set_SP_NS(uint32_t value)
    {
      __asm volatile("MSR      SP_NS,%0" :: "r" (value));
    }

    __IAR_FT uint32_t   __TZ_get_PRIMASK_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,PRIMASK_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_PRIMASK_NS(uint32_t value)
    {
      __asm volatile("MSR      PRIMASK_NS,%0" :: "r" (value));
    }

    __IAR_FT uint32_t   __TZ_get_BASEPRI_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,BASEPRI_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_BASEPRI_NS(uint32_t value)
    {
      __asm volatile("MSR      BASEPRI_NS,%0" :: "r" (value));
    }

    __IAR_FT uint32_t   __TZ_get_FAULTMASK_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,FAULTMASK_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_FAULTMASK_NS(uint32_t value)
    {
      __asm volatile("MSR      FAULTMASK_NS,%0" :: "r" (value));
    }

    __IAR_FT uint32_t   __TZ_get_PSPLIM_NS(void)
    {
      uint32_t res;
    #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
         (!defined (__ARM_FEATURE_CMSE  ) || (__ARM_FEATURE_CMSE   < 3)))
      // without main extensions, the non-secure PSPLIM is RAZ/WI
      res = 0U;
    #else
      __asm volatile("MRS      %0,PSPLIM_NS" : "=r" (res));
    #endif
      return res;
    }

    __IAR_FT void   __TZ_set_PSPLIM_NS(uint32_t value)
    {
    #if (!(defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) && \
         (!defined (__ARM_FEATURE_CMSE  ) || (__ARM_FEATURE_CMSE   < 3)))
      // without main extensions, the non-secure PSPLIM is RAZ/WI
      (void)value;
    #else
      __asm volatile("MSR      PSPLIM_NS,%0" :: "r" (value));
    #endif
    }

    __IAR_FT uint32_t   __TZ_get_MSPLIM_NS(void)
    {
      uint32_t res;
      __asm volatile("MRS      %0,MSPLIM_NS" : "=r" (res));
      return res;
    }

    __IAR_FT void   __TZ_set_MSPLIM_NS(uint32_t value)
    {
      __asm volatile("MSR      MSPLIM_NS,%0" :: "r" (value));
    }

  #endif /* (defined (__ARM_ARCH_8M_MAIN__ ) && (__ARM_ARCH_8M_MAIN__ == 1)) || \
            (defined (__ARM_ARCH_8M_BASE__ ) && (__ARM_ARCH_8M_BASE__ == 1))    ) */
#endif   /* __ICCARM_INTRINSICS_VERSION__ == 2 */


#if __IAR_M0_FAMILY

#endif

#undef __IAR_FT
#undef __IAR_M0_FAMILY
#undef __ICCARM_V8

#pragma diag_default=Pe940
#pragma diag_default=Pe177

#define __SXTB16_RORn(ARG1, ARG2) __SXTB16(__ROR(ARG1, ARG2))

#define __SXTAB16_RORn(ARG1, ARG2, ARG3) __SXTAB16(ARG1, __ROR(ARG2, ARG3))

#endif /* __CMSIS_ICCARM_H__ */
