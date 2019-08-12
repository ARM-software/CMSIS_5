
/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_canberra_distance_f32.c
 * Description:  Canberra distance between two vectors
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
 * @brief        Canberra distance between two vectors
 *
 * This function may divide by zero when samples pA[i] and pB[i] are both zero.
 * The result of the computation will be correct. So the division per zero may be
 * ignored.
 *
 * @param[in]    pA         First vector
 * @param[in]    pB         Second vector
 * @param[in]    blockSize  vector length
 * @return distance
 *
 */

#if defined(ARM_MATH_NEON)

#include "NEMath.h"

float32_t arm_canberra_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize)
{
   float32_t accum=0.0, tmpA, tmpB,diff,sum;
   uint32_t i,blkCnt;
   float32x4_t a,b,c,d,accumV;
   float32x2_t accumV2;
   int32x4_t   isZeroV;
   float32x4_t zeroV = vdupq_n_f32(0.0);

   accumV = vdupq_n_f32(0.0);

   blkCnt = blockSize >> 2;
   while(blkCnt > 0)
   {
        a = vld1q_f32(pA);
        b = vld1q_f32(pB);

        c = vabdq_f32(a,b);

        a = vabsq_f32(a);
        b = vabsq_f32(b);
        a = vaddq_f32(a,b);
        isZeroV = vceqq_f32(a,zeroV);

        /* 
         * May divide by zero when a and b have both the same lane at zero.
         */
        a = vinvq_f32(a);
        
        /*
         * Force result of a division by 0 to 0. It the behavior of the
         * sklearn canberra function.
         */
        a = vbicq_s32(a,isZeroV);
        c = vmulq_f32(c,a);
        accumV = vaddq_f32(accumV,c);

        pA += 4;
        pB += 4;
        blkCnt --;
   }
   accumV2 = vpadd_f32(vget_low_f32(accumV),vget_high_f32(accumV));
   accum = accumV2[0] + accumV2[1];


   blkCnt = blockSize & 3;
   while(blkCnt > 0)
   {
      tmpA = *pA++;
      tmpB = *pB++;

      diff = fabs(tmpA - tmpB);
      sum = fabs(tmpA) + fabs(tmpB);
      if ((tmpA != 0.0) || (tmpB != 0.0))
      {
         accum += (diff / sum);
      }
      blkCnt --;
   }
   return(accum);
}

#else
float32_t arm_canberra_distance_f32(const float32_t *pA,const float32_t *pB, uint32_t blockSize)
{
   float32_t accum=0.0, tmpA, tmpB,diff,sum;

   while(blockSize > 0)
   {
      tmpA = *pA++;
      tmpB = *pB++;

      diff = fabs(tmpA - tmpB);
      sum = fabs(tmpA) + fabs(tmpB);
      if ((tmpA != 0.0) || (tmpB != 0.0))
      {
         accum += (diff / sum);
      }
      blockSize --;
   }
   return(accum);
}
#endif


/**
 * @} end of FloatDist group
 */
