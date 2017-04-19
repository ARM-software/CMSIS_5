/*
 * Copyright (c) 2013-2017 ARM Limited. All rights reserved.
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
 *
 * -----------------------------------------------------------------------------
 *
 * Project:     CMSIS-RTOS RTX
 * Title:       Cortex-A Core definitions
 *
 * -----------------------------------------------------------------------------
 */

#ifndef RTX_CORE_CA_H_
#define RTX_CORE_CA_H_

#include "RTE_Components.h"
#include CMSIS_device_header

#define __DOMAIN_NS             0U
#define __EXCLUSIVE_ACCESS      1U

/* CPSR initial state */
#define CPSR_INIT_USER          0x00000010U
#define CPSR_INIT_SYSTEM        0x0000001FU

/* CPSR bit definitions */
#define CPSR_T_BIT              0x20U
#define CPSR_I_BIT              0x80U
#define CPSR_F_BIT              0x40U

/* CPSR mode bitmasks */
#define CPSR_MODE_USER          0x10U
#define CPSR_MODE_SYSTEM        0x1FU

/* Determine privilege level */
#define IS_PRIVILEGED()          (__get_mode() != CPSR_MODE_USER)
#define IS_IRQ_MODE()           ((__get_mode() != CPSR_MODE_USER) && (__get_mode() != CPSR_MODE_SYSTEM))
#define IS_IRQ_MASKED()         (0U)

#define STACK_FRAME_INIT        0x00U

#define IS_VFP_D32_STACK_FRAME(n) (((n) & 0x04U) != 0U)
#define IS_VFP_D16_STACK_FRAME(n) (((n) & 0x02U) != 0U)

/* Emulate M profile get_PSP: SP_usr - (8*4) */
#if defined(__CC_ARM)
static __asm    uint32_t __get_PSP (void) {
  arm
  sub   sp, sp, #4
  stm   sp, {sp}^
  pop   {r0}
  sub   r0, r0, #32
  bx    lr
}
#else
__STATIC_INLINE uint32_t __get_PSP (void) {
  register uint32_t ret;

  __asm volatile (
    ".syntax unified\n\t"
    ".arm\n\t"
    "sub  sp,sp,#4\n\t"
    "stm  sp,{sp}^\n\t"
    "pop  {%[ret]}\n\t"
    "sub  %[ret],%[ret],#32\n\t"
    : [ret] "=&l" (ret)
    :
    : "memory"
  );

  return ret;
}
#endif


//  ==== Service Calls definitions ====

#if defined(__CC_ARM)

#define __SVC_INDIRECT(n) __svc_indirect(n)

#define SVC0_0N(f,t)                                                           \
__SVC_INDIRECT(0) t    svc##f (t(*)());                                        \
                  t svcRtx##f (void);                                          \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (void) {                                         \
  svc##f(svcRtx##f);                                                           \
}

#define SVC0_0(f,t)                                                            \
__SVC_INDIRECT(0) t    svc##f (t(*)());                                        \
                  t svcRtx##f (void);                                          \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (void) {                                         \
  return svc##f(svcRtx##f);                                                    \
}

#define SVC0_0D SVC0_0

#define SVC0_1N(f,t,t1)                                                        \
__SVC_INDIRECT(0) t    svc##f (t(*)(t1),t1);                                   \
                  t svcRtx##f (t1 a1);                                         \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1) {                                        \
  svc##f(svcRtx##f,a1);                                                        \
}

#define SVC0_1(f,t,t1)                                                         \
__SVC_INDIRECT(0) t    svc##f (t(*)(t1),t1);                                   \
                  t svcRtx##f (t1 a1);                                         \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1) {                                        \
  return svc##f(svcRtx##f,a1);                                                 \
}

#define SVC0_2(f,t,t1,t2)                                                      \
__SVC_INDIRECT(0) t    svc##f (t(*)(t1,t2),t1,t2);                             \
                  t svcRtx##f (t1 a1, t2 a2);                                  \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1, t2 a2) {                                 \
  return svc##f(svcRtx##f,a1,a2);                                              \
}

#define SVC0_3(f,t,t1,t2,t3)                                                   \
__SVC_INDIRECT(0) t    svc##f (t(*)(t1,t2,t3),t1,t2,t3);                       \
                  t svcRtx##f (t1 a1, t2 a2, t3 a3);                           \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1, t2 a2, t3 a3) {                          \
  return svc##f(svcRtx##f,a1,a2,a3);                                           \
}

#define SVC0_4(f,t,t1,t2,t3,t4)                                                \
__SVC_INDIRECT(0) t    svc##f (t(*)(t1,t2,t3,t4),t1,t2,t3,t4);                 \
                  t svcRtx##f (t1 a1, t2 a2, t3 a3, t4 a4);                    \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1, t2 a2, t3 a3, t4 a4) {                   \
  return svc##f(svcRtx##f,a1,a2,a3,a4);                                        \
}

#define SVC0_0M SVC0_0
#define SVC0_1M SVC0_1
#define SVC0_2M SVC0_2
#define SVC0_3M SVC0_3
#define SVC0_4M SVC0_4

#elif defined(__ICCARM__)

#define SVC_Setup(f)                                                           \
  __asm(                                                                       \
    "mov r12,%0\n"                                                             \
    :: "r"(&f): "r12"                                                          \
  );

#define STRINGIFY(a) #a
#define __SVC_INDIRECT(n) _Pragma(STRINGIFY(swi_number = n)) __swi

#define SVC0_0N(f,t)                                                           \
__SVC_INDIRECT(0) t    svc##f ();                                              \
                  t svcRtx##f (void);                                          \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (void) {                                         \
  SVC_Setup(svcRtx##f);                                                        \
  svc##f();                                                                    \
}

#define SVC0_0(f,t)                                                            \
__SVC_INDIRECT(0) t    svc##f ();                                              \
                  t svcRtx##f (void);                                          \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (void) {                                         \
  SVC_Setup(svcRtx##f);                                                        \
  return svc##f();                                                             \
}

#define SVC0_0D SVC0_0

#define SVC0_1N(f,t,t1)                                                        \
__SVC_INDIRECT(0) t    svc##f (t1 a1);                                         \
                  t svcRtx##f (t1 a1);                                         \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1) {                                        \
  SVC_Setup(svcRtx##f);                                                        \
  svc##f(a1);                                                                  \
}

#define SVC0_1(f,t,t1)                                                         \
__SVC_INDIRECT(0) t    svc##f (t1 a1);                                         \
                  t svcRtx##f (t1 a1);                                         \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1) {                                        \
  SVC_Setup(svcRtx##f);                                                        \
  return svc##f(a1);                                                           \
}

#define SVC0_2(f,t,t1,t2)                                                      \
__SVC_INDIRECT(0) t    svc##f (t1 a1, t2 a2);                                  \
                  t svcRtx##f (t1 a1, t2 a2);                                  \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1, t2 a2) {                                 \
  SVC_Setup(svcRtx##f);                                                        \
  return svc##f(a1,a2);                                                        \
}

#define SVC0_3(f,t,t1,t2,t3)                                                   \
__SVC_INDIRECT(0) t    svc##f (t1 a1, t2 a2, t3 a3);                           \
                  t svcRtx##f (t1 a1, t2 a2, t3 a3);                           \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1, t2 a2, t3 a3) {                          \
  SVC_Setup(svcRtx##f);                                                        \
  return svc##f(a1,a2,a3);                                                     \
}

#define SVC0_4(f,t,t1,t2,t3,t4)                                                \
__SVC_INDIRECT(0) t    svc##f (t1 a1, t2 a2, t3 a3, t4 a4);                    \
                  t svcRtx##f (t1 a1, t2 a2, t3 a3, t4 a4);                    \
__attribute__((always_inline))                                                 \
__STATIC_INLINE   t  __svc##f (t1 a1, t2 a2, t3 a3, t4 a4) {                   \
  SVC_Setup(svcRtx##f);                                                        \
  return svc##f(a1,a2,a3,a4);                                                  \
}

#define SVC0_0M SVC0_0
#define SVC0_1M SVC0_1
#define SVC0_2M SVC0_2
#define SVC0_3M SVC0_3
#define SVC0_4M SVC0_4

#else   // !(defined(__CC_ARM) || defined(__ICCARM__))

#define SVC_RegF "r12"

#define SVC_ArgN(n) \
register uint32_t __r##n __ASM("r"#n)

#define SVC_ArgR(n,a) \
register uint32_t __r##n __ASM("r"#n) = (uint32_t)a

#define SVC_ArgF(f) \
register uint32_t __rf   __ASM(SVC_RegF) = (uint32_t)f

#define SVC_In0 "r"(__rf)
#define SVC_In1 "r"(__rf),"r"(__r0)
#define SVC_In2 "r"(__rf),"r"(__r0),"r"(__r1)
#define SVC_In3 "r"(__rf),"r"(__r0),"r"(__r1),"r"(__r2)
#define SVC_In4 "r"(__rf),"r"(__r0),"r"(__r1),"r"(__r2),"r"(__r3)

#define SVC_Out0
#define SVC_Out1 "=r"(__r0)
#define SVC_Out2 "=r"(__r0),"=r"(__r1)

#define SVC_CL0
#define SVC_CL1 "r1"
#define SVC_CL2 "r0","r1"

#define SVC_Call0(in, out, cl)                                                 \
  __ASM volatile ("svc 0" : out : in : cl)

#define SVC0_0N(f,t)                                                           \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (void) {                                            \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In0, SVC_Out0, SVC_CL2);                                       \
}

#define SVC0_0(f,t)                                                            \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (void) {                                            \
  SVC_ArgN(0);                                                                 \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In0, SVC_Out1, SVC_CL1);                                       \
  return (t) __r0;                                                             \
}

#define SVC0_0D(f,t)                                                           \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (void) {                                            \
  SVC_ArgN(0);                                                                 \
  SVC_ArgN(1);                                                                 \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In0, SVC_Out2, SVC_CL0);                                       \
  return (((t) __r0) | (((t) __r1) << 32));                                    \
}

#define SVC0_1N(f,t,t1)                                                        \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (t1 a1) {                                           \
  SVC_ArgR(0,a1);                                                              \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In1, SVC_Out0, SVC_CL1);                                       \
}

#define SVC0_1(f,t,t1)                                                         \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (t1 a1) {                                           \
  SVC_ArgR(0,a1);                                                              \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In1, SVC_Out1, SVC_CL1);                                       \
  return (t) __r0;                                                             \
}

#define SVC0_2(f,t,t1,t2)                                                      \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (t1 a1, t2 a2) {                                    \
  SVC_ArgR(0,a1);                                                              \
  SVC_ArgR(1,a2);                                                              \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In2, SVC_Out1, SVC_CL0);                                       \
  return (t) __r0;                                                             \
}

#define SVC0_3(f,t,t1,t2,t3)                                                   \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (t1 a1, t2 a2, t3 a3) {                             \
  SVC_ArgR(0,a1);                                                              \
  SVC_ArgR(1,a2);                                                              \
  SVC_ArgR(2,a3);                                                              \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In3, SVC_Out1, SVC_CL0);                                       \
  return (t) __r0;                                                             \
}

#define SVC0_4(f,t,t1,t2,t3,t4)                                                \
__attribute__((always_inline))                                                 \
__STATIC_INLINE t __svc##f (t1 a1, t2 a2, t3 a3, t4 a4) {                      \
  SVC_ArgR(0,a1);                                                              \
  SVC_ArgR(1,a2);                                                              \
  SVC_ArgR(2,a3);                                                              \
  SVC_ArgR(3,a4);                                                              \
  SVC_ArgF(svcRtx##f);                                                         \
  SVC_Call0(SVC_In4, SVC_Out1, SVC_CL0);                                       \
  return (t) __r0;                                                             \
}

#define SVC0_0M SVC0_0
#define SVC0_1M SVC0_1
#define SVC0_2M SVC0_2
#define SVC0_3M SVC0_3
#define SVC0_4M SVC0_4

#endif


//  ==== Core Peripherals functions ====

extern uint32_t SystemCoreClock;        // System Clock Frequency (Core Clock)

/// Initialize SVC and PendSV System Service Calls (not needed on Cortex-A)
__STATIC_INLINE void SVC_Initialize (void) {
}

/// Setup External Tick Timer Interrupt
/// \param[in] irqn  Interrupt number
extern void ExtTick_SetupIRQ (int32_t irqn);

/// Enable External Tick Timer Interrupt
/// \param[in] irqn  Interrupt number
extern void ExtTick_EnableIRQ (int32_t irqn);

/// Disable External Tick Timer Interrupt
/// \param[in] irqn  Interrupt number
extern void ExtTick_DisableIRQ (int32_t irqn);

/// Get Pending SV (Service Call) and ST (SysTick) Flags
/// \return    Pending SV&ST Flags
__STATIC_INLINE uint8_t GetPendSV_ST (void) {
  return (0U);
}

/// Get Pending SV (Service Call) Flag
/// \return    Pending SV Flag
extern uint8_t GetPendSV (void);

/// Clear Pending SV (Service Call) and ST (SysTick) Flags
__STATIC_INLINE void ClrPendSV_ST (void) {
}

/// Clear Pending SV (Service Call) Flag
extern void ClrPendSV (void);

/// Set Pending SV (Service Call) Flag
extern void SetPendSV (void);

/// Set Pending Flags
/// \param[in] flags  Flags to set
extern void SetPendFlags (uint8_t flags);


//  ==== Exclusive Access Operation ====

#if (__EXCLUSIVE_ACCESS == 1U)

/// Atomic Access Operation: Write (8-bit)
/// \param[in]  mem             Memory address
/// \param[in]  val             Value to write
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint8_t atomic_wr8 (uint8_t *mem, uint8_t val) {
  mov    r2,r0
1
  ldrexb r0,[r2]
  strexb r3,r1,[r2]
  cmp    r3,#0
  bne    %B1
  bx     lr
}
#else
__STATIC_INLINE uint8_t atomic_wr8 (uint8_t *mem, uint8_t val) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint8_t  ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrexb %[ret],[%[mem]]\n\t"
    "strexb %[res],%[val],[%[mem]]\n\t"
    "cmp    %[res],#0\n\t"
    "bne    1b\n\t"
  : [ret] "=&l" (ret),
    [res] "=&l" (res)
  : [mem] "l"   (mem),
    [val] "l"   (val)
  : "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Set bits (32-bit)
/// \param[in]  mem             Memory address
/// \param[in]  bits            Bit mask
/// \return                     New value
#if defined(__CC_ARM)
static __asm    uint32_t atomic_set32 (uint32_t *mem, uint32_t bits) {
  mov   r2,r0
1
  ldrex r0,[r2]
  orr   r0,r0,r1
  strex r3,r0,[r2]
  cmp   r3,#0
  bne   %B1
  bx    lr
}
#else
__STATIC_INLINE uint32_t atomic_set32 (uint32_t *mem, uint32_t bits) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[val],[%[mem]]\n\t"
    "orr   %[ret],%[val],%[bits]\n\t"
    "strex %[res],%[ret],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  : [ret]  "=&l" (ret),
    [val]  "=&l" (val),
    [res]  "=&l" (res)
  : [mem]  "l"   (mem),
    [bits] "l"   (bits)
  : "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Clear bits (32-bit)
/// \param[in]  mem             Memory address
/// \param[in]  bits            Bit mask
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint32_t atomic_clr32 (uint32_t *mem, uint32_t bits) {
  push  {r4,lr}
  mov   r2,r0
1
  ldrex r0,[r2]
  bic   r4,r0,r1
  strex r3,r4,[r2]
  cmp   r3,#0
  bne   %B1
  pop   {r4,pc}
}
#else
__STATIC_INLINE uint32_t atomic_clr32 (uint32_t *mem, uint32_t bits) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[mem]]\n\t"
    "bic   %[val],%[ret],%[bits]\n\t"
    "strex %[res],%[val],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  : [ret]  "=&l" (ret),
    [val]  "=&l" (val),
    [res]  "=&l" (res)
  : [mem]  "l"   (mem),
    [bits] "l"   (bits)
  : "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Check if all specified bits (32-bit) are active and clear them
/// \param[in]  mem             Memory address
/// \param[in]  bits            Bit mask
/// \return                     Active bits before clearing or 0 if not active
#if defined(__CC_ARM)
static __asm    uint32_t atomic_chk32_all (uint32_t *mem, uint32_t bits) {
  push  {r4,lr}
  mov   r2,r0
1
  ldrex r0,[r2]
  and   r4,r0,r1
  cmp   r4,r1
  beq   %F2
  clrex
  movs  r0,#0
  pop   {r4,pc}
2
  bic   r4,r0,r1
  strex r3,r4,[r2]
  cmp   r3,#0
  bne   %B1
  pop   {r4,pc}
}
#else
__STATIC_INLINE uint32_t atomic_chk32_all (uint32_t *mem, uint32_t bits) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[mem]]\n\t"
    "and   %[val],%[ret],%[bits]\n\t"
    "cmp   %[val],%[bits]\n\t"
    "beq   2f\n\t"
    "clrex\n\t"
    "movs  %[ret],#0\n\t"
    "b     3f\n"
  "2:\n\t"
    "bic   %[val],%[ret],%[bits]\n\t"
    "strex %[res],%[val],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  "3:"
  : [ret]  "=&l" (ret),
    [val]  "=&l" (val),
    [res]  "=&l" (res)
  : [mem]  "l"   (mem),
    [bits] "l"   (bits)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Check if any specified bits (32-bit) are active and clear them
/// \param[in]  mem             Memory address
/// \param[in]  bits            Bit mask
/// \return                     Active bits before clearing or 0 if not active
#if defined(__CC_ARM)
static __asm    uint32_t atomic_chk32_any (uint32_t *mem, uint32_t bits) {
  push  {r4,lr}
  mov   r2,r0
1
  ldrex r0,[r2]
  tst   r0,r1
  bne   %F2
  clrex
  movs  r0,#0
  pop   {r4,pc}
2
  bic   r4,r0,r1
  strex r3,r4,[r2]
  cmp    r3,#0
  bne   %B1
  pop   {r4,pc}
}
#else
__STATIC_INLINE uint32_t atomic_chk32_any (uint32_t *mem, uint32_t bits) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[mem]]\n\t"
    "tst   %[ret],%[bits]\n\t"
    "bne   2f\n\t"
    "clrex\n\t"
    "movs  %[ret],#0\n\t"
    "b     3f\n"
  "2:\n\t"
    "bic   %[val],%[ret],%[bits]\n\t"
    "strex %[res],%[val],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  "3:"
  : [ret]  "=&l" (ret),
    [val]  "=&l" (val),
    [res]  "=&l" (res)
  : [mem]  "l"   (mem),
    [bits] "l"   (bits)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Increment (32-bit)
/// \param[in]  mem             Memory address
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint32_t atomic_inc32 (uint32_t *mem) {
  mov   r2,r0
1
  ldrex r0,[r2]
  adds  r1,r0,#1
  strex r3,r1,[r2]
  cmp   r3,#0
  bne   %B1
  bx    lr
}
#else
__STATIC_INLINE uint32_t atomic_inc32 (uint32_t *mem) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[mem]]\n\t"
    "adds  %[val],%[ret],#1\n\t"
    "strex %[res],%[val],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  : [ret] "=&l" (ret),
    [val] "=&l" (val),
    [res] "=&l" (res)
  : [mem] "l"   (mem)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// atomic Access Operation: Increment (32-bit) if Less Than
/// \param[in]  mem             Memory address
/// \param[in]  max             Maximum value
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint32_t atomic_inc32_lt (uint32_t *mem, uint32_t max) {
  push  {r4,lr}
  mov   r2,r0
1
  ldrex r0,[r2]
  cmp   r1,r0
  bhi   %F2
  clrex
  pop   {r4,pc}
2
  adds  r4,r0,#1
  strex r3,r4,[r2]
  cmp   r3,#0
  bne   %B1
  pop   {r4,pc}
}
#else
__STATIC_INLINE uint32_t atomic_inc32_lt (uint32_t *mem, uint32_t max) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[mem]]\n\t"
    "cmp   %[max],%[ret]\n\t"
    "bhi    2f\n\t"
    "clrex\n\t"
    "b      3f\n"
  "2:\n\t"
    "adds  %[val],%[ret],#1\n\t"
    "strex %[res],%[val],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  "3:"
  : [ret] "=&l" (ret),
    [val] "=&l" (val),
    [res] "=&l" (res)
  : [mem] "l"   (mem),
    [max] "l"   (max)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Increment (16-bit) if Less Than
/// \param[in]  mem             Memory address
/// \param[in]  max             Maximum value
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint16_t atomic_inc16_lt (uint16_t *mem, uint16_t max) {
  push   {r4,lr}
  mov    r2,r0
1
  ldrexh r0,[r2]
  cmp    r1,r0
  bhi    %F2
  clrex
  pop    {r4,pc}
2
  adds   r4,r0,#1
  strexh r3,r4,[r2]
  cmp    r3,#0
  bne    %B1
  pop    {r4,pc}
}
#else
__STATIC_INLINE uint16_t atomic_inc16_lt (uint16_t *mem, uint16_t max) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint16_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrexh %[ret],[%[mem]]\n\t"
    "cmp    %[max],%[ret]\n\t"
    "bhi    2f\n\t"
    "clrex\n\t"
    "b      3f\n"
  "2:\n\t"
    "adds   %[val],%[ret],#1\n\t"
    "strexh %[res],%[val],[%[mem]]\n\t"
    "cmp    %[res],#0\n\t"
    "bne    1b\n"
  "3:"
  : [ret] "=&l" (ret),
    [val] "=&l" (val),
    [res] "=&l" (res)
  : [mem] "l"   (mem),
    [max] "l"   (max)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Increment (16-bit) and clear on Limit
/// \param[in]  mem             Memory address
/// \param[in]  max             Maximum value
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint16_t atomic_inc16_lim (uint16_t *mem, uint16_t lim) {
  push   {r4,lr}
  mov    r2,r0
1
  ldrexh r0,[r2]
  adds   r4,r0,#1
  cmp    r1,r4
  bhi    %F2
  movs   r4,#0
2
  strexh r3,r4,[r2]
  cmp    r3,#0
  bne    %B1
  pop    {r4,pc}
}
#else
__STATIC_INLINE uint16_t atomic_inc16_lim (uint16_t *mem, uint16_t lim) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint16_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrexh %[ret],[%[mem]]\n\t"
    "adds   %[val],%[ret],#1\n\t"
    "cmp    %[lim],%[val]\n\t"
    "bhi    2f\n\t"
    "movs   %[val],#0\n"
  "2:\n\t"
    "strexh %[res],%[val],[%[mem]]\n\t"
    "cmp    %[res],#0\n\t"
    "bne    1b\n"
  : [ret] "=&l" (ret),
    [val] "=&l" (val),
    [res] "=&l" (res)
  : [mem] "l"   (mem),
    [lim] "l"   (lim)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Decrement (32-bit) if Not Zero
/// \param[in]  mem             Memory address
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint32_t atomic_dec32_nz (uint32_t *mem) {
  mov   r2,r0
1
  ldrex r0,[r2]
  cmp   r0,#0
  bne   %F2
  clrex
  bx    lr
2
  subs  r1,r0,#1
  strex r3,r1,[r2]
  cmp   r3,#0
  bne   %B1
  bx    lr
}
#else
__STATIC_INLINE uint32_t atomic_dec32_nz (uint32_t *mem) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint32_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[mem]]\n\t"
    "cmp   %[ret],#0\n\t"
    "bne   2f\n"
    "clrex\n\t"
    "b     3f\n"
  "2:\n\t"
    "subs  %[val],%[ret],#1\n\t"
    "strex %[res],%[val],[%[mem]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  "3:"
  : [ret] "=&l" (ret),
    [val] "=&l" (val),
    [res] "=&l" (res)
  : [mem] "l"   (mem)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Decrement (16-bit) if Not Zero
/// \param[in]  mem             Memory address
/// \return                     Previous value
#if defined(__CC_ARM)
static __asm    uint16_t atomic_dec16_nz (uint16_t *mem) {
  mov    r2,r0
1
  ldrexh r0,[r2]
  cmp    r0,#0
  bne    %F2
  clrex
  bx     lr
2
  subs   r1,r0,#1
  strexh r3,r1,[r2]
  cmp    r3,#0
  bne    %B1
  bx      lr
}
#else
__STATIC_INLINE uint16_t atomic_dec16_nz (uint16_t *mem) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register uint16_t ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrexh %[ret],[%[mem]]\n\t"
    "cmp    %[ret],#0\n\t"
    "bne    2f\n\t"
    "clrex\n\t"
    "b      3f\n"
  "2:\n\t"
    "subs   %[val],%[ret],#1\n\t"
    "strexh %[res],%[val],[%[mem]]\n\t"
    "cmp    %[res],#0\n\t"
    "bne    1b\n"
  "3:"
  : [ret] "=&l" (ret),
    [val] "=&l" (val),
    [res] "=&l" (res)
  : [mem] "l"   (mem)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Link Get
/// \param[in]  root            Root address
/// \return                     Link
#if defined(__CC_ARM)
static __asm    void *atomic_link_get (void **root) {
  mov   r2,r0
1
  ldrex r0,[r2]
  cmp   r0,#0
  bne   %F2
  clrex
  bx    lr
2
  ldr   r1,[r0]
  strex r3,r1,[r2]
  cmp   r3,#0
  bne   %B1
  bx    lr
}
#else
__STATIC_INLINE void *atomic_link_get (void **root) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif
  register void    *ret;

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldrex %[ret],[%[root]]\n\t"
    "cmp   %[ret],#0\n\t"
    "bne   2f\n\t"
    "clrex\n\t"
    "b     3f\n"
  "2:\n\t"
    "ldr   %[val],[%[ret]]\n\t"
    "strex %[res],%[val],[%[root]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  "3:"
  : [ret]  "=&l" (ret),
    [val]  "=&l" (val),
    [res]  "=&l" (res)
  : [root] "l"   (root)
  : "cc", "memory"
  );

  return ret;
}
#endif

/// Atomic Access Operation: Link Put
/// \param[in]  root            Root address
/// \param[in]  lnk             Link
#if defined(__CC_ARM)
static __asm    void atomic_link_put (void **root, void *link) {
1
  ldr   r2,[r0]
  str   r2,[r1]
  dmb
  ldrex r2,[r0]
  ldr   r3,[r1]
  cmp   r3,r2
  bne   %B1
  strex r3,r1,[r0]
  cmp   r3,#0
  bne   %B1
  bx    lr
}
#else
__STATIC_INLINE void atomic_link_put (void **root, void *link) {
#ifdef  __ICCARM__
#pragma diag_suppress=Pe550
#endif
  register uint32_t val1, val2, res;
#ifdef  __ICCARM__
#pragma diag_default=Pe550
#endif

  __ASM volatile (
#ifndef __ICCARM__
  ".syntax unified\n\t"
#endif
  "1:\n\t"
    "ldr   %[val1],[%[root]]\n\t"
    "str   %[val1],[%[link]]\n\t"
    "dmb\n\t"
    "ldrex %[val1],[%[root]]\n\t"
    "ldr   %[val2],[%[link]]\n\t"
    "cmp   %[val2],%[val1]\n\t"
    "bne   1b\n\t"
    "strex %[res],%[link],[%[root]]\n\t"
    "cmp   %[res],#0\n\t"
    "bne   1b\n"
  : [val1] "=&l" (val1),
    [val2] "=&l" (val2),
    [res]  "=&l" (res)
  : [root] "l"   (root),
    [link] "l"   (link)
  : "cc", "memory"
  );
}
#endif

#endif  // (__EXCLUSIVE_ACCESS == 1U)


#endif  // RTX_CORE_CA_H_
