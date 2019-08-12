
/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_jensenshannon_distance_f32.c
 * Description:  Jensen-Shannon distance between two vectors
 *
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
#include <limits.h>
#include <math.h>


static inline double rel_entr(double x, double y)
{
    return (x * log(x / y));
}


/**
  @addtogroup FloatDist
  @{
 */



#if defined(ARM_MATH_NEON)

#include "NEMath.h"


/**
 * @brief        Jensen-Shannon distance between two vectors
 *
 * This function is assuming that elements of second vector are > 0
 * and 0 only when the corresponding element of first vector is 0.
 * Otherwise the result of the computation does not make sense
 * and for speed reasons, the cases returning NaN or Infinity are not
 * managed.
 *
 * When the function is computing x log (x / y) with x == 0 and y == 0,
 * it will compute the right result (0) but a division by zero will occur
 * and should be ignored in client code.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */


float32_t arm_jensenshannon_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize)
{
    float32_t accum, result, tmp,a,b;
    uint32_t i, blkCnt;
    float32x4_t aV,bV,t, tmpV, accumV;
    float32x2_t accumV2;

    accum = 0.0; 
    accumV = vdupq_n_f32(0.0);

    blkCnt = blockSize >> 2;
    while(blkCnt > 0)
    {
      aV = vld1q_f32(pA);
      bV = vld1q_f32(pB);
      t = vaddq_f32(aV,bV);
      t = vmulq_n_f32(t, 0.5);

      tmpV = vmulq_f32(aV, vinvq_f32(t));
      tmpV = vlogq_f32(tmpV);
      accumV = vmlaq_f32(accumV, aV, tmpV);


      tmpV = vmulq_f32(bV, vinvq_f32(t));
      tmpV = vlogq_f32(tmpV);
      accumV = vmlaq_f32(accumV, bV, tmpV);

      pA += 4;
      pB += 4;


      blkCnt --;
    }

    accumV2 = vpadd_f32(vget_low_f32(accumV),vget_high_f32(accumV));
    accum = accumV2[0] + accumV2[1];

    blkCnt = blockSize & 3;
    while(blkCnt > 0)
    {
      a = *pA;
      b = *pB;
      tmp = (a + b) / 2.0;
      accum += rel_entr(a, tmp);
      accum += rel_entr(b, tmp);

      pA++;
      pB++;

      blkCnt --;
    }


    arm_sqrt_f32(accum/2.0, &result);
    return(result);

}

#else


/**
 * @brief        Jensen-Shannon distance between two vectors
 *
 * This function is assuming that elements of second vector are > 0
 * and 0 only when the corresponding element of first vector is 0.
 * Otherwise the result of the computation does not make sense
 * and for speed reasons, the cases returning NaN or Infinity are not
 * managed.
 *
 * When the function is computing x log (x / y) with x == 0 and y == 0,
 * it will compute the right result (0) but a division by zero will occur
 * and should be ignored in client code.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */


float32_t arm_jensenshannon_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize)
{
    float32_t left, right,sum, result, tmp;
    uint32_t i;

    left = 0.0; 
    right = 0.0;
    for(i=0; i < blockSize; i++)
    {
      tmp = (pA[i] + pB[i]) / 2.0;
      left  += rel_entr(pA[i], tmp);
      right += rel_entr(pB[i], tmp);
    }


    sum = left + right;
    arm_sqrt_f32(sum/2.0, &result);
    return(result);

}

#endif

/**
 * @} end of FloatDist group
 */
