/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_req_f32.c
 * Description:  IIR filter control function
 *
 * $Date:        2019
 * $Revision:    
 *
 * Target Processor: Cortex-M cores
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
  @addtogroup IIRs
  @{
 */

/**
 * @brief  Control function for the floating-point IIR filter.
 * @param[in]     iirType      type of IIR filter.
 * @param[in]     order        order of IIR filter.
 * @param[in]     nbCascaded   number of stages
 * @param[in]     fastFlag     flag for fast version
 * @param[out]    stateSize    points to the needed size for the state
 * @param[out]    coeffSize    points to the needed size for the coefficients
 * @param[out]    stateAlign   points to the needed alignment for the state
 * @param[out]    coeffAlign   points to the needed alignment for the coefficients
 *
 * @par   
 *        This function returns the amount of memory to allocate for the state and for the 
 *        coefficients in bytes and the required alignments. The values are computed according
 *        to the type of filter, the order, the number of stages and wheter or not the fast 
 *        algorithm has been chosen.
 *
 * @par   
 *        For example:
 * <pre>arm_iir_req_f32(type, order, stages, fastFlag, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
 * float32_t * state = (float32_t *)malloc(stateSize + stateAlign -1);
 * state = (uint32_t)(state + statealign -1) & (0xfffffff0); *
 * float32_t * coeffs = (float32_t *)malloc(coeffSize + coeffAlign -1);
 * coeffs = (uint32_t)(coeffs + coeffalign -1) & (0xfffffff0);
 * </pre>  
 * 
 * @par
 * The function \ref arm_iir_init_f32() will be used to initialize the allocated memory spaces.
 *
 */

void arm_iir_req_f32(arm_iir_type iirType, uint16_t order, uint32_t nbCascaded, uint16_t simdFlag, uint32_t * stateSize, uint32_t * coeffSize, uint32_t * stateAlign, uint32_t * coeffAlign)
{
    if (iirType == ARM_IIR_DF1) 
    {
        /* x[n-order] ... x[n-1] , y[n-order] ... y[n-1] */
        *stateSize = sizeof(float32_t)*2*order*nbCascaded;

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )       
        if(simdFlag)
        {
            *coeffSize = sizeof(float32_t)*(24+8*(order-1))*nbCascaded;
            /* | 0,   0,   0,     b_0,   ...    b_ord, a1,    ...    a_ord |
               | 0,   0,   b_0,   ...    b_ord, 0,     ...    a_ord, 0     |
               | 0,   b_0, ...    b_ord, 0,     0,     a_ord, 0,     0     |
               | b_0, ...  b_ord, 0,     0,     0,     0,     0,     0     | */
        }
        else
#else
        /* To avoid warning */
        (void)simdFlag;
#endif
        {
            /* b_0 b_1 ... b_order , a_1 ... a_order */
            *coeffSize = sizeof(float32_t)*(order+order+1)*nbCascaded;
        }
    }
    else if (iirType == ARM_IIR_DF2T)
    {
        /* Used only for biquad filters */
        *stateSize = sizeof(float32_t)*order*nbCascaded;
        *coeffSize = sizeof(float32_t)*32*nbCascaded;
    }

    *stateAlign = sizeof(float32_t)*4;
    *coeffAlign = sizeof(float32_t)*4;
}

/**
  @} end of IIRs group
 */
