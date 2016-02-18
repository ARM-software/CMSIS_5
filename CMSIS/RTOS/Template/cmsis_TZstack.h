/*
 * Copyright (c) 2015-2016 ARM Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ----------------------------------------------------------------------
 *
 * $Date:        18. February 2016
 * $Revision:    V0.01
 *
 * Project:      CMSIS-RTOS extensions for ARMv8-M TrustZone 
 * Title:        Secure Stack Management for RTOS on ARMv8-M TrustZone  
 *
 * Version 0.01
 *    Initial Proposal Phase
 *---------------------------------------------------------------------------*/
 
#ifndef __cmsis_TZstack_H
#define __cmsis_TZstack_H

#include <stdint.h>

/// Initialize Secure Process Stack Management
/// \return execution status (0: success, <0: error)
int32_t TZ_Init_Stack_S (void);

/// Allocate Memory for Secure Process Stack Management
/// \param[in]  box_id  secure box identifier
/// \return value >= 0 context identifier
/// \return value <  0 no memory available or internal error
int32_t TZ_Alloc_Stack_S (uint32_t box_id);

/// Free Memory for Secure Process Stack Management
/// \param[in]  context_id  context identifier
/// \return execution status (0: success, <0: error)
int32_t TZ_Free_Stack_S (int32_t context_id);

/// Load Secure Context
/// \param[in]  context_id  context identifier
/// \return execution status (0: success, <0: error)
int32_t TZ_Load_Context_S (int32_t context_id);

/// Store Secure Context
/// \param[in]  context_id  context identifier
/// \return execution status (0: success, <0: error)
int32_t TZ_Store_Context_S (int32_t context_id);

#endif  // __cmsis_TZstack_H
