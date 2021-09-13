/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_fir_f64.c
 * Description:  Floating-point FIR filter processing function
 *
 * $Date:        13 September 2021
 * $Revision:    V1.10.0
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
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

#include "dsp/filtering_functions.h"

/**
  @ingroup groupFilters
 */

/**
  @defgroup FIR Finite Impulse Response (FIR) Filters

  This set of functions implements Finite Impulse Response (FIR) filters
  for Q7, Q15, Q31, and floating-point data types.  Fast versions of Q15 and Q31 are also provided.
  The functions operate on blocks of input and output data and each call to the function processes
  <code>blockSize</code> samples through the filter.  <code>pSrc</code> and
  <code>pDst</code> points to input and output arrays containing <code>blockSize</code> values.

  @par           Algorithm
                   The FIR filter algorithm is based upon a sequence of multiply-accumulate (MAC) operations.
                   Each filter coefficient <code>b[n]</code> is multiplied by a state variable which equals a previous input sample <code>x[n]</code>.
  <pre>
      y[n] = b[0] * x[n] + b[1] * x[n-1] + b[2] * x[n-2] + ...+ b[numTaps-1] * x[n-numTaps+1]
  </pre>
  @par
                   \image html FIR.GIF "Finite Impulse Response filter"
  @par
                   <code>pCoeffs</code> points to a coefficient array of size <code>numTaps</code>.
                   Coefficients are stored in time reversed order.
  @par
  <pre>
      {b[numTaps-1], b[numTaps-2], b[N-2], ..., b[1], b[0]}
  </pre>
  @par
                   <code>pState</code> points to a state array of size <code>numTaps + blockSize - 1</code>.
                   Samples in the state buffer are stored in the following order.
  @par
  <pre>
      {x[n-numTaps+1], x[n-numTaps], x[n-numTaps-1], x[n-numTaps-2]....x[n](==pSrc[0]), x[n+1](==pSrc[1]), ..., x[n+blockSize-1](==pSrc[blockSize-1])}
  </pre>
  @par
                   Note that the length of the state buffer exceeds the length of the coefficient array by <code>blockSize-1</code>.
                   The increased state buffer length allows circular addressing, which is traditionally used in the FIR filters,
                   to be avoided and yields a significant speed improvement.
                   The state variables are updated after each block of data is processed; the coefficients are untouched.

  @par           Instance Structure
                   The coefficients and state variables for a filter are stored together in an instance data structure.
                   A separate instance structure must be defined for each filter.
                   Coefficient arrays may be shared among several instances while state variable arrays cannot be shared.
                   There are separate instance structure declarations for each of the 4 supported data types.

  @par           Initialization Functions
                   There is also an associated initialization function for each data type.
                   The initialization function performs the following operations:
                   - Sets the values of the internal structure fields.
                   - Zeros out the values in the state buffer.
                   To do this manually without calling the init function, assign the follow subfields of the instance structure:
                   numTaps, pCoeffs, pState. Also set all of the values in pState to zero.
  @par
                   Use of the initialization function is optional.
                   However, if the initialization function is used, then the instance structure cannot be placed into a const data section.
                   To place an instance structure into a const data section, the instance structure must be manually initialized.
                   Set the values in the state buffer to zeros before static initialization.
                   The code below statically initializes each of the 4 different data type filter instance structures
  <pre>
      arm_fir_instance_f32 S = {numTaps, pState, pCoeffs};
      arm_fir_instance_q31 S = {numTaps, pState, pCoeffs};
      arm_fir_instance_q15 S = {numTaps, pState, pCoeffs};
      arm_fir_instance_q7 S =  {numTaps, pState, pCoeffs};
  </pre>
                   where <code>numTaps</code> is the number of filter coefficients in the filter; <code>pState</code> is the address of the state buffer;
                   <code>pCoeffs</code> is the address of the coefficient buffer.
  @par          Initialization of Helium version
                 For Helium version the array of coefficients must be padded with zero to contain
                 a full number of lanes.

                 The array length L must be a multiple of x. L = x * a :
                 - x is 4  for f32
                 - x is 4  for q31
                 - x is 4  for f16 (so managed like the f32 version and not like the q15 one)
                 - x is 8  for q15
                 - x is 16 for q7

                 The additional coefficients 
                 (x * a - numTaps) must be set to 0.
                 numTaps is still set to its right value in the init function. It means that
                 the implementation may require to read more coefficients due to the vectorization and
                 to avoid having to manage too many different cases in the code.

                
  @par          Helium state buffer
                 The state buffer must contain some additional temporary data
                 used during the computation but which is not the state of the FIR.
                 The first A samples are temporary data.
                 The remaining samples are the state of the FIR filter.
  @par                 
                 So the state buffer has size <code> numTaps + A + blockSize - 1 </code> :
                 - A is blockSize for f32
                 - A is 8*ceil(blockSize/8) for f16
                 - A is 8*ceil(blockSize/4) for q31
                 - A is 0 for other datatypes (q15 and q7)


  @par           Fixed-Point Behavior
                   Care must be taken when using the fixed-point versions of the FIR filter functions.
                   In particular, the overflow and saturation behavior of the accumulator used in each function must be considered.
                   Refer to the function specific documentation below for usage guidelines.

 */

/**
  @addtogroup FIR
  @{
 */

/**
  @brief         Processing function for floating-point FIR filter.
  @param[in]     S          points to an instance of the floating-point FIR filter structure
  @param[in]     pSrc       points to the block of input data
  @param[out]    pDst       points to the block of output data
  @param[in]     blockSize  number of samples to process
  @return        none
 */

void arm_fir_f64(
  const arm_fir_instance_f64 * S,
  const float64_t * pSrc,
        float64_t * pDst,
        uint32_t blockSize)
{
        float64_t *pState = S->pState;                 /* State pointer */
  const float64_t *pCoeffs = S->pCoeffs;               /* Coefficient pointer */
        float64_t *pStateCurnt;                        /* Points to the current sample of the state */
        float64_t *px;                                 /* Temporary pointer for state buffer */
  const float64_t *pb;                                 /* Temporary pointer for coefficient buffer */
        float64_t acc0;                                /* Accumulator */
        uint32_t numTaps = S->numTaps;                 /* Number of filter coefficients in the filter */
        uint32_t i, tapCnt, blkCnt;                    /* Loop counters */

  /* S->pState points to state array which contains previous frame (numTaps - 1) samples */
  /* pStateCurnt points to the location where the new input data should be written */
  pStateCurnt = &(S->pState[(numTaps - 1U)]);

  /* Initialize blkCnt with number of taps */
  blkCnt = blockSize;

  while (blkCnt > 0U)
  {
    /* Copy one sample at a time into state buffer */
    *pStateCurnt++ = *pSrc++;

    /* Set the accumulator to zero */
    acc0 = 0.0f;

    /* Initialize state pointer */
    px = pState;

    /* Initialize Coefficient pointer */
    pb = pCoeffs;

    i = numTaps;

    /* Perform the multiply-accumulates */
    while (i > 0U)
    {
      /* acc =  b[numTaps-1] * x[n-numTaps-1] + b[numTaps-2] * x[n-numTaps-2] + b[numTaps-3] * x[n-numTaps-3] +...+ b[0] * x[0] */
      acc0 += *px++ * *pb++;

      i--;
    }

    /* Store result in destination buffer. */
    *pDst++ = acc0;

    /* Advance state pointer by 1 for the next sample */
    pState = pState + 1U;

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Processing is complete.
     Now copy the last numTaps - 1 samples to the start of the state buffer.
     This prepares the state buffer for the next function call. */

  /* Points to the start of the state buffer */
  pStateCurnt = S->pState;

  /* Initialize tapCnt with number of taps */
  tapCnt = (numTaps - 1U);

  /* Copy remaining data */
  while (tapCnt > 0U)
  {
    *pStateCurnt++ = *pState++;

    /* Decrement loop counter */
    tapCnt--;
  }

}

/**
* @} end of FIR group
*/
