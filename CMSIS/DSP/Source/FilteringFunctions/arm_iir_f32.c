/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_f32.c
 * Description:  IIR filter processing function
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
#if defined(ARM_ERROR_HANDLER)
#include "arm_error.h"
#endif

void arm_iir_1st_f32(const arm_iir_instance_f32 * S, float32_t * pSrc, float32_t * pDst, uint32_t blockSize);
void arm_iir_2nd_f32(const arm_iir_instance_f32 * S, float32_t * pSrc, float32_t * pDst, uint32_t blockSize);
void arm_iir_3rd_f32(const arm_iir_instance_f32 * S, float32_t * pSrc, float32_t * pDst, uint32_t blockSize);
void arm_iir_generic_f32(const arm_iir_instance_f32 * S, float32_t * pSrc, float32_t * pDst, uint32_t blockSize);

/**
  @ingroup groupFilters
 */

/**
  @defgroup IIRs 

  This set of functions implements cascades of IIR filters of any order
  for floating-point data types.

  @par           Algorithm
                   Each stage implements an IIR filter of order k
                   using the difference equation:
  @par
  <pre>
      y[n] = b0 * x[n] + ... bk * x[n-k] + a1 * y[n-1] + ... + ak * y[n-k] 
  </pre>

  @par
                   The <code>pState</code> points to state variables array.
                   Each stage has 2k state variables. The state variables 
                   for all stages are arranged in the <code>pState</code> array as:
  <pre>
      x[n-1], ... x[n-k], y[n-1], ... y[n-k]
  </pre>

  @par
                   The state variables for stage 1 are first, then the state variables for stage 2, and so on.
                   The state array has a total length of <code>2*k*numStages</code> values.
                   The state variables are updated after each block of data is processed, the coefficients are untouched.

  @par           Instance Structure
                   The coefficients and state variables for a filter are stored together in an instance data structure.
                   A separate instance structure must be defined for each filter.
                   Coefficient arrays may be shared among several instances while state variable arrays cannot be shared.
                   There are separate instance structure declarations for each of the 3 supported data types.

  @par           Init Function
                   There is also an associated initialization function for each data type.
                   The initialization function performs the following operations:
                   - Sets the values of the internal structure fields.
                   - Zeros out the values in the state buffer.
 */

/**
  @addtogroup IIRs Vector IIR filters
  @{
 */

/**
   * @brief  Processing function for the floating-point IIR filter.
   * @param[in]     S          points to an instance of the floating-point IIR filter.
   * @param[in]     pSrc       points to the block of input data.
   * @param[out]    pDst       points to the block of output data.
   * @param[in]     blockSize  number of samples to process.
   * @return        none
   *
   * @par Initialization function
   *   Refer to \ref arm_iir_init_f32().
   *
 */

void arm_iir_f32(
    const arm_iir_instance_f32 * S,
          float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
    uint16_t order = S->order;

#if defined(ARM_ERROR_HANDLER)
    if(S->debugFlag)
    {
        if((uint32_t)(S->pState) & 0x000f )
            arm_error_handler(ARM_ERROR_ALIGNMENT, "Unaligned state");

        if((uint32_t)(S->pCoeffs) & 0x000f )
            arm_error_handler(ARM_ERROR_ALIGNMENT, "Unaligned coefficients");
    }
#endif

    switch(order)
    {
        case 1:
        arm_iir_1st_f32(S, pSrc, pDst, blockSize);
        break;
        
        case 2:
        arm_iir_2nd_f32(S, pSrc, pDst, blockSize);
        break;
        
        case 3:
        arm_iir_3rd_f32(S, pSrc, pDst, blockSize);
        break;
        
        default:
        arm_iir_generic_f32(S, pSrc, pDst, blockSize);
        break;
    }
}

/**
  @} end of IIRs group
 */
