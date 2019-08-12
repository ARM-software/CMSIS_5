/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_svm_rbf_predict_f32.c
 * Description:  SVM Radial Basis Function Classifier
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
 * @addtogroup groupSVM
 * @{
 */


/**
 * @brief SVM rbf prediction
 * @param[in]    S         Pointer to an instance of the rbf SVM structure.
 * @param[in]    in        Pointer to input vector
 * @param[out]   pResult   decision value
 * @return none.
 *
 */
#if defined(ARM_MATH_NEON)

#include "NEMath.h"

void arm_svm_rbf_predict_f32(
    const arm_svm_rbf_instance_f32 *S,
    const float32_t * in,
    int * pResult)
{
    float32_t sum = S->intercept;
   
    float32_t dot;
    float32x4_t dotV; 

    float32x4_t accuma,accumb,accumc,accumd,accum;
    float32x2_t accum2;
    float32x4_t temp;
    float32x4_t vec1;

    float32x4_t vec2,vec2a,vec2b,vec2c,vec2d;

    uint32_t blkCnt;   
    uint32_t vectorBlkCnt;   

    const float32_t *pIn = in;

    const float32_t *pSupport = S->supportVectors;

    const float32_t *pSupporta = S->supportVectors;
    const float32_t *pSupportb;
    const float32_t *pSupportc;
    const float32_t *pSupportd;

    pSupportb = pSupporta + S->vectorDimension;
    pSupportc = pSupportb + S->vectorDimension;
    pSupportd = pSupportc + S->vectorDimension;

    const float32_t *pDualCoefs = S->dualCoefficients;


    vectorBlkCnt = S->nbOfSupportVectors >> 2;
    while (vectorBlkCnt > 0U)
    {
        accuma = vdupq_n_f32(0);
        accumb = vdupq_n_f32(0);
        accumc = vdupq_n_f32(0);
        accumd = vdupq_n_f32(0);

        pIn = in;

        blkCnt = S->vectorDimension >> 2;
        while (blkCnt > 0U)
        {
        
            vec1 = vld1q_f32(pIn);
            vec2a = vld1q_f32(pSupporta);
            vec2b = vld1q_f32(pSupportb);
            vec2c = vld1q_f32(pSupportc);
            vec2d = vld1q_f32(pSupportd);

            pIn += 4;
            pSupporta += 4;
            pSupportb += 4;
            pSupportc += 4;
            pSupportd += 4;

            temp = vsubq_f32(vec1, vec2a);
            accuma = vmlaq_f32(accuma, temp, temp);

            temp = vsubq_f32(vec1, vec2b);
            accumb = vmlaq_f32(accumb, temp, temp);

            temp = vsubq_f32(vec1, vec2c);
            accumc = vmlaq_f32(accumc, temp, temp);

            temp = vsubq_f32(vec1, vec2d);
            accumd = vmlaq_f32(accumd, temp, temp);

            blkCnt -- ;
        }
        accum2 = vpadd_f32(vget_low_f32(accuma),vget_high_f32(accuma));
        dotV[0] = accum2[0] + accum2[1];

        accum2 = vpadd_f32(vget_low_f32(accumb),vget_high_f32(accumb));
        dotV[1] = accum2[0] + accum2[1];

        accum2 = vpadd_f32(vget_low_f32(accumc),vget_high_f32(accumc));
        dotV[2] = accum2[0] + accum2[1];

        accum2 = vpadd_f32(vget_low_f32(accumd),vget_high_f32(accumd));
        dotV[3] = accum2[0] + accum2[1];


        blkCnt = S->vectorDimension & 3;
        while (blkCnt > 0U)
        {
            dotV[0] = dotV[0] + SQ(*pIn - *pSupporta);
            dotV[1] = dotV[1] + SQ(*pIn - *pSupportb);
            dotV[2] = dotV[2] + SQ(*pIn - *pSupportc);
            dotV[3] = dotV[3] + SQ(*pIn - *pSupportd);
            pSupporta++;
            pSupportb++;
            pSupportc++;
            pSupportd++;

            pIn++;

            blkCnt -- ;
        }

        vec1 = vld1q_f32(pDualCoefs);
        pDualCoefs += 4; 

        // To vectorize later
        dotV = vmulq_n_f32(dotV, -S->gamma);
        dotV = vexpq_f32(dotV);

        accum = vmulq_f32(vec1,dotV);
        accum2 = vpadd_f32(vget_low_f32(accum),vget_high_f32(accum));
        sum += accum2[0] + accum2[1];

        pSupporta += 3*S->vectorDimension;
        pSupportb += 3*S->vectorDimension;
        pSupportc += 3*S->vectorDimension;
        pSupportd += 3*S->vectorDimension;

        vectorBlkCnt -- ;
    }

    pSupport = pSupporta;
    vectorBlkCnt = S->nbOfSupportVectors & 3;

    while (vectorBlkCnt > 0U)
    {
        accum = vdupq_n_f32(0);
        dot = 0.0;
        pIn = in;

        blkCnt = S->vectorDimension >> 2;
        while (blkCnt > 0U)
        {
        
            vec1 = vld1q_f32(pIn);
            vec2 = vld1q_f32(pSupport);
            pIn += 4;
            pSupport += 4;

            temp = vsubq_f32(vec1,vec2);
            accum = vmlaq_f32(accum, temp,temp);

            blkCnt -- ;
        }
        accum2 = vpadd_f32(vget_low_f32(accum),vget_high_f32(accum));
        dot = accum2[0] + accum2[1];


        blkCnt = S->vectorDimension & 3;
        while (blkCnt > 0U)
        {

            dot = dot + SQ(*pIn - *pSupport);
            pIn++;
            pSupport++;

            blkCnt -- ;
        }

        sum += *pDualCoefs++ * exp(-S->gamma * dot);
        vectorBlkCnt -- ;
    }

    *pResult=S->classes[STEP(sum)];
}
#else
void arm_svm_rbf_predict_f32(
    const arm_svm_rbf_instance_f32 *S,
    const float32_t * in,
    int * pResult)
{
    float32_t sum=S->intercept;
    float32_t dot=0;
    const float32_t *pSupport = S->supportVectors;

    for(int i=0; i < S->nbOfSupportVectors; i++)
    {
        dot=0;
        for(int j=0; j < S->vectorDimension; j++)
        {
            dot = dot + SQ(in[j] - *pSupport);
            pSupport++;
        }
        sum += S->dualCoefficients[i] * exp(-S->gamma * dot);
    }
    *pResult=S->classes[STEP(sum)];
}
#endif


/**
 * @} end of groupSVM group
 */
