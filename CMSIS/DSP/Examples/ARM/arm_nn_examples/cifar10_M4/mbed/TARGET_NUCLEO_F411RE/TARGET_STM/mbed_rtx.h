/* mbed Microcontroller Library
 * Copyright (c) 2016 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MBED_MBED_RTX_H
#define MBED_MBED_RTX_H

#if defined(TARGET_STM32F051R8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20002000UL)
#endif

#elif defined(TARGET_STM32L031K6)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20002000UL)
#endif

#elif defined(TARGET_STM32F070RB)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20004000UL)
#endif

#elif defined(TARGET_STM32F072RB)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20004000UL)
#endif

#elif defined(TARGET_STM32F091RC)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20008000UL)
#endif

#elif defined(TARGET_STM32F100RB)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20002000UL)
#endif

#elif defined(TARGET_STM32F103RB)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20005000UL)
#endif

#elif defined(TARGET_STM32F207ZG)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F303VC)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x2000A000UL)
#endif

#elif defined(TARGET_STM32F334C8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20003000UL)
#endif

#elif defined(TARGET_STM32F302R8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20004000UL)
#endif

#elif defined(TARGET_STM32F303K8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20003000UL)
#endif

#elif defined(TARGET_STM32F303RE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20010000UL)
#endif

#elif defined(TARGET_STM32F303ZE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20010000UL)
#endif

#elif defined(TARGET_STM32F334R8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20003000UL)
#endif

#elif defined(TARGET_STM32F446VE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F401VC)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20010000UL)
#endif

#elif (defined(TARGET_STM32F429ZI) || defined(TARGET_STM32F439ZI))

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20030000UL)
#endif

#elif defined(TARGET_UBLOX_EVK_ODIN_W2)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20030000UL)
#endif

#elif defined(TARGET_UBLOX_C030)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20030000UL)
#endif

#elif defined(TARGET_STM32F469NI)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20050000UL)
#endif

#elif defined(TARGET_STM32F405RG)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F401RE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20018000UL)
#endif

#elif defined(TARGET_STM32F410RB)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20008000UL)
#endif

#elif defined(TARGET_MTS_MDOT_F411RE) || defined (TARGET_MTS_DRAGONFLY_F411RE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F411RE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F412ZG)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20040000UL)
#endif

#elif defined(TARGET_STM32F446RE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F446ZE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F407VG)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20020000UL)
#endif

#elif defined(TARGET_STM32F746NG)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20050000UL)
#endif

#elif (defined(TARGET_STM32F746ZG) || defined(TARGET_STM32F756ZG))

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20050000UL)
#endif

#elif defined(TARGET_STM32F767ZI)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20080000UL)
#endif

#elif defined(TARGET_STM32F769NI)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20080000UL)
#endif

#elif defined(TARGET_STM32L053C8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20002000UL)
#endif

#elif defined(TARGET_STM32L031K6)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20002000UL)
#endif

#elif defined(TARGET_STM32L053R8)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20002000UL)
#endif

#elif defined(TARGET_STM32L072CZ)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20005000UL)
#endif

#elif defined(TARGET_STM32L073RZ)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20005000UL)
#endif

#elif defined(TARGET_STM32L152RC)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20008000UL)
#endif

#elif defined(TARGET_STM32L152RE)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20014000UL)
#endif

#elif defined(TARGET_NZ32_SC151)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20008000UL)
#endif

#elif defined(TARGET_XDOT_L151CC)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20008000UL)
#endif

#elif defined(TARGET_STM32L476VG) || defined(TARGET_STM32L475VG)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20018000UL)
#endif

#elif defined(TARGET_STM32L432KC)

#ifndef INITIAL_SP
#define INITIAL_SP              (0x2000C000UL)
#endif

#elif (defined(TARGET_STM32L476RG) || defined(TARGET_STM32L486RG))

#ifndef INITIAL_SP
#define INITIAL_SP              (0x20018000UL)
#endif

#endif

#endif  // MBED_MBED_RTX_H
