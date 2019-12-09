/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_median_filter_init_f32.c
 * Description:  Floating-point median filter initialization function
 *
 * $Date:        
 * $Revision:    
 *
 * Target Processor: 
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2019 ARM Limited or its affiliates. All rights reserved.
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

/**
  @ingroup groupFilters
 */

/**
  @addtogroup MedianFilters
  @{
 */

/**
 * @brief  Initialization function for the floating-point median filter.
 * @param[in,out] S           points to an instance of the floating-point median filter structure.
 * @param[in]     windowSize  size of the window.
 * @param[in]     pBuffer     points to the working array
 * @param[in]     pDelay      points to the delay samples used at the beginning
 */

#include "arm_math.h"
#if defined(ARM_ERROR_HANDLER)
#include "arm_error.h"
#endif

void arm_median_filter_init_f32(arm_median_filter_instance_f32 * S, uint32_t windowSize, float32_t * pBuffer, float32_t * pDelay)
{
#if defined (ARM_ERROR_HANDLER)
    if( windowsize<1 )
       arm_error_handler(ARM_ERROR_MATH, "The window size cannot be less than or equal to 0.");
#endif    
    S->windowSize = windowSize;
    S->pBuffer    = pBuffer;
    S->pDelay     = pDelay;
}

/**
  @} end of MedianFilters group
 */
