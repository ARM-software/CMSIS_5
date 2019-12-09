/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_exponential_smoothing_init_f32.c
 * Description:  Floating-point exponential smoothing initialization function
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

#include "arm_math.h"

/**
  @ingroup groupFilters
 */

/**
  @addtogroup ExponentialSmoothings
  @{
 */

/*
 * @brief  Initialization function for the floating-point exponential smoothing.
 * @param[in,out]  S           points to an instance of the exponential smoothing structure.
 * @param[in]      alpha       smoothing factor (IIR filter coefficient).
 * @param[in]      vectorSize  number of samples of one vector.
 */

void arm_exponential_smoothing_init_f32(arm_exp_smooth_instance_f32* S, float32_t alpha, uint32_t vectorSize)
{
    S->alpha = alpha;
    S->vectorSize = vectorSize;
}

/**
  @} end of ExponentialSmoothings group
 */
