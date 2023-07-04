/*
 * Copyright (c) 2023 Arm Limited. All rights reserved.
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

#ifndef __MEM_ARMCM3_H
#define __MEM_ARMCM3_H

/*---------------------------------------------------------------------------
// <<< Use Configuration Wizard in Context Menu >>>
  ---------------------------------------------------------------------------*/

/*--------------------- ROM Configuration -----------------------------------
// <h> Flash Configuration
//   <o0> Flash Base Address <0x0-0xFFFFFFFF:8>
//   <o1> Flash Size (in Bytes) <0x0-0xFFFFFFFF:8>
// </h>
  ---------------------------------------------------------------------------*/
#define __ROM_BASE       0x00000000
#define __ROM_SIZE       0x00040000

/*--------------------- RAM Configuration -----------------------------------
// <h> RAM Configuration
//   <o0> RAM Base Address    <0x0-0xFFFFFFFF:8>
//   <o1> RAM Size (in Bytes) <0x0-0xFFFFFFFF:8>
// </h>
 ----------------------------------------------------------------------------*/
#define __RAM_BASE       0x20000000
#define __RAM_SIZE       0x00020000

/*--------------------- Stack / Heap Configuration --------------------------
// <h> Stack / Heap Configuration
//   <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//   <o1> Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
// </h>
  ---------------------------------------------------------------------------*/
#define __STACK_SIZE     0x00000400
#define __HEAP_SIZE      0x00000C00

/*---------------------------------------------------------------------------
// <<< end of configuration section >>>
  ---------------------------------------------------------------------------*/

#endif /* __MEM_ARMCM3_H */
