/**************************************************************************//**
 * @file     mem_ARMCA5.h
 * @brief    Memory base and size definitions (used in scatter file)
 * @version  V1.1.0
 * @date     15. May 2019
 *
 * @note
 *
 ******************************************************************************/
/*
 * Copyright (c) 2009-2019 Arm Limited. All rights reserved.
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

#ifndef __MEM_ARMCA5_H
#define __MEM_ARMCA5_H

/*----------------------------------------------------------------------------
  User Stack & Heap size definition
 *----------------------------------------------------------------------------*/
/*
//-------- <<< Use Configuration Wizard in Context Menu >>> ------------------
*/

/*--------------------- ROM Configuration ------------------------------------
//
// <h> ROM Configuration
//   <i> For compatibility with MMU config the sections must be multiple of 1MB
//   <o0> ROM Base Address <0x0-0xFFFFFFFF:0x100000>
//   <o1> ROM Size (in Bytes) <0x0-0xFFFFFFFF:0x100000>
// </h>
 *----------------------------------------------------------------------------*/
#define __ROM_BASE       0x80000000
#define __ROM_SIZE       0x00400000

/*--------------------- RAM Configuration -----------------------------------
// <h> RAM Configuration
//   <i> For compatibility with MMU config the sections must be multiple of 1MB
//   <o0> RAM Base Address    <0x0-0xFFFFFFFF:0x100000>
//   <o1> RAM Total Size (in Bytes) <0x0-0xFFFFFFFF:0x100000>
//   <h> Data Sections
//     <o2> RW_DATA Size (in Bytes) <0x0-0xFFFFFFFF:8>
//     <o3> ZI_DATA Size (in Bytes) <0x0-0xFFFFFFFF:8>
//   </h>
//   <h> Stack / Heap Configuration
//     <o4>  Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//     <o5>  Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
//     <h> Exceptional Modes
//       <o6> UND Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//       <o7> ABT Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//       <o8> SVC Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//       <o9> IRQ Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//       <o10> FIQ Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//     </h>
//   </h>
// </h>
 *----------------------------------------------------------------------------*/
//#define __RAM_BASE       0x80200000#
#define __RAM_BASE       0x80600000
#define __RAM_SIZE       0x00700000

#define __RW_DATA_SIZE   0x00200000
#define __ZI_DATA_SIZE   0x00200000

#define __STACK_SIZE     0x00007000
#define __HEAP_SIZE      0x00200000

#define __UND_STACK_SIZE 0x00000100
#define __ABT_STACK_SIZE 0x00000100
#define __SVC_STACK_SIZE 0x00000100
#define __IRQ_STACK_SIZE 0x00000100
#define __FIQ_STACK_SIZE 0x00000100

/*----------------------------------------------------------------------------*/

/*--------------------- TTB Configuration ------------------------------------
//
// <h> TTB Configuration
//   <i> The TLB L1 contains 4096 32-bit entries and must be 16kB aligned
//   <i> The TLB L2 entries are placed after the L1 in the MMU config
//   <o0> TTB Base Address <0x0-0xFFFFFFFF:0x4000>
//   <o1> TTB Size (in Bytes) <0x0-0xFFFFFFFF:8>
// </h>
 *----------------------------------------------------------------------------*/
#define __TTB_BASE       0x80500000
#define __TTB_SIZE       0x00005000

#endif /* __MEM_ARMCA5_H */
