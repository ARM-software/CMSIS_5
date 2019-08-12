
/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_minkowski_distance_f32.c
 * Description:  Minkowski distance between two vectors
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


/**
  @addtogroup FloatDist
  @{
 */


/**
 * @brief        Minkowski distance between two vectors
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    order      Distance order
 * @param[in]    blockSize  Number of samples
 * @return distance
 *
 */
#if defined(ARM_MATH_NEON)

#include "NEMath.h"

float32_t arm_minkowski_distance_f32(const float32_t *pA,const float32_t *pB, int order, uint32_t blockSize)
{
    float32_t sum,diff;
    uint32_t i, blkCnt;
    float32x4_t sumV,aV,bV, tmpV, n;
    float32x2_t sumV2;

    sum = 0.0; 
    sumV = vdupq_n_f32(0.0);
    n = vdupq_n_f32(order);

    blkCnt = blockSize >> 2;
    while(blkCnt > 0)
    {
       aV = vld1q_f32(pA);
       bV = vld1q_f32(pB);
       pA += 4;
       pB += 4;

       tmpV = vabdq_f32(aV,bV);
       tmpV = vpowq_f32(tmpV,n);
       sumV = vaddq_f32(sumV, tmpV);


       blkCnt --;
    }

    sumV2 = vpadd_f32(vget_low_f32(sumV),vget_high_f32(sumV));
    sum = sumV2[0] + sumV2[1];

    blkCnt = blockSize & 3;
    while(blkCnt > 0)
    {
       sum += pow(fabs(*pA++ - *pB++),order);

       blkCnt --;
    }


    return(pow(sum,(1.0/order)));

}

#else


float32_t arm_minkowski_distance_f32(const float32_t *pA,const float32_t *pB, int order, uint32_t blockSize)
{
    float32_t sum,diff;
    uint32_t i;

    sum = 0.0; 
    for(i=0; i < blockSize; i++)
    {
       sum += pow(fabs(pA[i] - pB[i]),order);
    }


    return(pow(sum,(1.0/order)));

}
#endif


/**
 * @} end of FloatDist group
 */
