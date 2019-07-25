/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_logsumexp_f32.c
 * Description:  LogSumExp
 *
 *
 * Target Processor: Cortex-M and Cortex-A cores
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
 * @addtogroup groupStats
 * @{
 */


/**
 * @brief Kullback-Leibler
 *
 * Distribution A may contain 0 with Neon version.
 * Result will be right but some exception flags will be set.
 *
 * Distribution B must not contain 0 probability.
 *
 * @param[in]  *pSrcA         points to an array of input values for probaility distribution A.
 * @param[in]  *pSrcB         points to an array of input values for probaility distribution B.
 * @param[in]  blockSize      number of samples in the input array.
 * @return Kullback-Leibler divergence D(A || B)
 *
 */

#if defined(ARM_MATH_NEON)

#include "NEMath.h"

float32_t arm_kullback_leibler_f32(const float32_t * pSrcA,const float32_t * pSrcB,uint32_t blockSize)
{
    const float32_t *pInA, *pInB;
    uint32_t blkCnt;
    float32_t accum, pA,pB;

    float32x4_t accumV;
    float32x2_t accumV2;
    float32x4_t tmpVA, tmpVB,tmpV;
 
    pInA = pSrcA;
    pInB = pSrcB;

    accum = 0.0;
    accumV = vdupq_n_f32(0.0);

    blkCnt = blockSize >> 2;
    while(blkCnt > 0)
    {
      tmpVA = vld1q_f32(pInA);
      pInA += 4;

      tmpVB = vld1q_f32(pInB);
      pInB += 4;

      tmpV = vinvq_f32(tmpVA);
      tmpVB = vmulq_f32(tmpVB, tmpV);
      tmpVB = vlogq_f32(tmpVB);

      accumV = vmlaq_f32(accumV, tmpVA, tmpVB);
       
      blkCnt--;
    
    }

    accumV2 = vpadd_f32(vget_low_f32(accumV),vget_high_f32(accumV));
    accum = accumV2[0] + accumV2[1];

    blkCnt = blockSize & 3;
    while(blkCnt > 0)
    {
       pA = *pInA++;
       pB = *pInB++;
       accum += pA * log(pB/pA);
       
       blkCnt--;
    
    }

    return(-accum);
}

#else
float32_t arm_kullback_leibler_f32(const float32_t * pSrcA,const float32_t * pSrcB,uint32_t blockSize)
{
    const float32_t *pInA, *pInB;
    uint32_t blkCnt;
    float32_t accum, pA,pB;
 
    pInA = pSrcA;
    pInB = pSrcB;
    blkCnt = blockSize;

    accum = 0.0;

    while(blkCnt > 0)
    {
       pA = *pInA++;
       pB = *pInB++;
       accum += pA * log(pB / pA);
       
       blkCnt--;
    
    }

    return(-accum);
}
#endif
/**
 * @} end of groupStats group
 */
