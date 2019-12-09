/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_median_filter_f32.c
 * Description:  Floating-point median filter
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
  @defgroup MedianFilters
 
  This set of functions implements median filters.
 
  @par           Algorithm
  Given the size of the window, the ouput array is computed as   
  <pre>
      y[n] = med(x[n-(windowsize-1)], ..., x[n]) 
  </pre>
  
  @par      Usage
  A working buffer with the size of the windows is needed and it 
  must be allocated by the user and given to the init function \ref arm_median_filter_init_f32().

  @par
  In order to compute the first (windowsize-1) elements of the output vector,
  (windowsize-1) extra elements are needed. They should be stored in the pDelay
  filled of the instance through the initialization function.
  
  @par
  For example, if the window size is 3, pDelay will contain 2 elements and:
  <pre>
      pDst[0] = med(pDelay[0], pDelay[1], pSrc[0])
      pDst[1] = med(pDelay[1],  pSrc[0],  pSrc[1])
      pDst[2] = med( pSrc[0],   pSrc[1],  pSrc[2])
      ...
  </pre>
  If the pDelay buffer is not defined (it's NULL), the first element of pSrc
  will be duplicated:
  <pre>
      pDst[0] = med(pSrc[0], pSrc[0], pSrc[0])
      pDst[1] = med(pSrc[0], pSrc[0], pSrc[1])
      pDst[2] = med(pSrc[0], pSrc[1], pSrc[2])
      ...
  </pre>

*/

/**
  @addtogroup MedianFilters Vector median filters
  @{
 */

/**
   * @brief  Processing function for the floating-point median filter.
   * @param[in,out] S          points to an instance of the floating-point median filter.
   * @param[in]     pSrc       points to the block of input data.
   * @param[out]    pDst       points to the block of output data.
   * @param[in]     blockSize  number of samples to process.
 */

#include "arm_math.h"

void arm_median_filter_f32(
          arm_median_filter_instance_f32 * S,
    const float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
    uint32_t windowSize = S->windowSize;
    float32_t * pDelay = S->pDelay;
    float32_t * buf = S->pBuffer;
    int32_t i, j;

    arm_sort_instance_f32 S_sort;
    S_sort.alg = ARM_SORT_BUBBLE;
    S_sort.dir = 1;

    /* Compute median for first (windowSize-1) values */
    if(pDelay != NULL)
    {
        /* If the buffer delay has been defined, use it */
        for(i=0; i<(int32_t)windowSize-1; i++)
        {
            memcpy(buf, pDelay++, (windowSize-1-i)*sizeof(float32_t));
            memcpy(buf+(windowSize-1-i), pSrc, (i+1)*sizeof(float32_t));
    
            arm_sort_f32(&S_sort, buf, buf, windowSize);
            *pDst = buf[windowSize/2];
    
            pDst++;
        }
    }
    else
    {
        /* Otherwise duplicate the first value pSrc[0] */
        for(i=0; i<(int32_t)windowSize-1; i++)
        {
            for(j=0; j<(int32_t)(windowSize-1-i); j++)
                buf[j] = pSrc[0];        
            memcpy(buf+(windowSize-1-i), pSrc, (i+1)*sizeof(float32_t));
    
            arm_sort_f32(&S_sort, buf, buf, windowSize);
            *pDst = buf[windowSize/2];
    
            pDst++;
        }
    }

    /* Compute median for the remaining (blockSize-windowSize) values */
    for(i=(int32_t)windowSize; i<=(int32_t)blockSize; i++)
    {
        arm_sort_f32(&S_sort, (float32_t *)pSrc, buf, windowSize);
        *pDst = buf[windowSize/2];

        pSrc++;
        pDst++;
    }
}

/**
  @} end of IIRs group
 */
