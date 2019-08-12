
/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_euclidean_distance_f32.c
 * Description:  Euclidean distance between two vectors
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
 * @brief        Euclidean distance between two vectors
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */

#if defined(ARM_MATH_NEON)

#include "NEMath.h"

float32_t arm_euclidean_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize)
{
   float32_t accum=0.0,tmp;
   uint32_t i,blkCnt;
   float32x4_t a,b,accumV;
   float32x2_t accumV2;

   accumV = vdupq_n_f32(0.0);
   blkCnt = blockSize >> 2;
   while(blkCnt > 0)
   {
        a = vld1q_f32(pA);
        b = vld1q_f32(pB);

        a = vsubq_f32(a,b);
        accumV = vmlaq_f32(accumV,a,a);
        pA += 4;
        pB += 4;
        blkCnt --;
   }
   accumV2 = vpadd_f32(vget_low_f32(accumV),vget_high_f32(accumV));
   accum = accumV2[0] + accumV2[1];

   blkCnt = blockSize & 3;
   while(blkCnt > 0)
   {
      tmp = *pA++ - *pB++;
      accum += SQ(tmp);
      blkCnt --;
   }
   arm_sqrt_f32(accum,&tmp);
   return(tmp);
}

#else
float32_t arm_euclidean_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize)
{
   float32_t accum=0.0,tmp;
   uint32_t i;

   while(blockSize > 0)
   {
      tmp = *pA++ - *pB++;
      accum += SQ(tmp);
      blockSize --;
   }
   arm_sqrt_f32(accum,&tmp);
   return(tmp);
}
#endif


/**
 * @} end of FloatDist group
 */
