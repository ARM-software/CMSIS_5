/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_naive_gaussian_bayes_predict_f32
 * Description:  Naive Gaussian Bayesian Estimator
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

#define PI_F 3.1415926535897932384626433832795f
#define DPI_F (2*3.1415926535897932384626433832795f)

/**
 * @addtogroup groupBayes
 * @{
 */


#if defined(ARM_MATH_NEON)

#include "NEMath.h"

/**
 * @brief Naive Gaussian Bayesian Estimator
 *
 * @param[in]  *S         points to a naive bayes instance structure
 * @param[in]  *in        points to the elements of the input vector.
 * @param[in]  *pBuffer   points to a buffer of length numberOfClasses
 * @return The predicted class
 *
 */

uint32_t arm_gaussian_naive_bayes_predict_f32(const arm_gaussian_naive_bayes_instance_f32 *S, 
   const float32_t * in, 
   float32_t *pBuffer)
{
    int nbClass;
    int nbDim;
    const float32_t *pPrior = S->classPriors;

    const float32_t *pTheta = S->theta;
    const float32_t *pSigma = S->sigma;

    const float32_t *pTheta1 = S->theta + S->vectorDimension;
    const float32_t *pSigma1 = S->sigma + S->vectorDimension;

    float32_t *buffer = pBuffer;
    const float32_t *pIn=in;

    float32_t result;
    float32_t sigma,sigma1;
    float32_t tmp,tmp1;
    uint32_t index;
    uint32_t vecBlkCnt;
    uint32_t classBlkCnt;
    float32x4_t epsilonV;
    float32x4_t sigmaV,sigmaV1;
    float32x4_t tmpV,tmpVb,tmpV1;
    float32x2_t tmpV2;
    float32x4_t thetaV,thetaV1;
    float32x4_t inV;

    epsilonV = vdupq_n_f32(S->epsilon);

    classBlkCnt = S->numberOfClasses >> 1;
    while(classBlkCnt > 0)
    {

        
        pIn = in;

        tmp = log(*pPrior++);
        tmp1 = log(*pPrior++);
        tmpV = vdupq_n_f32(0.0);
        tmpV1 = vdupq_n_f32(0.0);

        vecBlkCnt = S->vectorDimension >> 2;
        while(vecBlkCnt > 0)
        {
           sigmaV = vld1q_f32(pSigma);
           thetaV = vld1q_f32(pTheta);

           sigmaV1 = vld1q_f32(pSigma1);
           thetaV1 = vld1q_f32(pTheta1);

           inV = vld1q_f32(pIn);

           sigmaV = vaddq_f32(sigmaV, epsilonV);
           sigmaV1 = vaddq_f32(sigmaV1, epsilonV);

           tmpVb = vmulq_n_f32(sigmaV,DPI_F);
           tmpVb = vlogq_f32(tmpVb);
           tmpV = vmlsq_n_f32(tmpV,tmpVb,0.5);

           tmpVb = vmulq_n_f32(sigmaV1,DPI_F);
           tmpVb = vlogq_f32(tmpVb);
           tmpV1 = vmlsq_n_f32(tmpV1,tmpVb,0.5);
           
           tmpVb = vsubq_f32(inV,thetaV);
           tmpVb = vmulq_f32(tmpVb,tmpVb);
           tmpVb = vmulq_f32(tmpVb, vinvq_f32(sigmaV));
           tmpV = vmlsq_n_f32(tmpV,tmpVb,0.5);

           tmpVb = vsubq_f32(inV,thetaV1);
           tmpVb = vmulq_f32(tmpVb,tmpVb);
           tmpVb = vmulq_f32(tmpVb, vinvq_f32(sigmaV1));
           tmpV1 = vmlsq_n_f32(tmpV1,tmpVb,0.5);

           pIn += 4;
           pTheta += 4;
           pSigma += 4;
           pTheta1 += 4;
           pSigma1 += 4;

           vecBlkCnt--;
        }
        tmpV2 = vpadd_f32(vget_low_f32(tmpV),vget_high_f32(tmpV));
        tmp += tmpV2[0] + tmpV2[1];

        tmpV2 = vpadd_f32(vget_low_f32(tmpV1),vget_high_f32(tmpV1));
        tmp1 += tmpV2[0] + tmpV2[1];

        vecBlkCnt = S->vectorDimension & 3;
        while(vecBlkCnt > 0)
        {
           sigma = *pSigma + S->epsilon;
           sigma1 = *pSigma1 + S->epsilon;

           tmp -= 0.5*log(2.0 * PI_F * sigma);
           tmp -= 0.5*(*pIn - *pTheta) * (*pIn - *pTheta) / sigma;

           tmp1 -= 0.5*log(2.0 * PI_F * sigma1);
           tmp1 -= 0.5*(*pIn - *pTheta1) * (*pIn - *pTheta1) / sigma1;

           pIn++;
           pTheta++;
           pSigma++;
           pTheta1++;
           pSigma1++;
           vecBlkCnt--;
        }

        *buffer++ = tmp;
        *buffer++ = tmp1;

        pSigma += S->vectorDimension;
        pTheta += S->vectorDimension;
        pSigma1 += S->vectorDimension;
        pTheta1 += S->vectorDimension;
        
        classBlkCnt--;
    }

    classBlkCnt = S->numberOfClasses & 1;

    while(classBlkCnt > 0)
    {

        
        pIn = in;

        tmp = log(*pPrior++);
        tmpV = vdupq_n_f32(0.0);

        vecBlkCnt = S->vectorDimension >> 2;
        while(vecBlkCnt > 0)
        {
           sigmaV = vld1q_f32(pSigma);
           thetaV = vld1q_f32(pTheta);
           inV = vld1q_f32(pIn);

           sigmaV = vaddq_f32(sigmaV, epsilonV);

           tmpVb = vmulq_n_f32(sigmaV,DPI_F);
           tmpVb = vlogq_f32(tmpVb);
           tmpV = vmlsq_n_f32(tmpV,tmpVb,0.5);
           
           tmpVb = vsubq_f32(inV,thetaV);
           tmpVb = vmulq_f32(tmpVb,tmpVb);
           tmpVb = vmulq_f32(tmpVb, vinvq_f32(sigmaV));
           tmpV = vmlsq_n_f32(tmpV,tmpVb,0.5);

           pIn += 4;
           pTheta += 4;
           pSigma += 4;

           vecBlkCnt--;
        }
        tmpV2 = vpadd_f32(vget_low_f32(tmpV),vget_high_f32(tmpV));
        tmp += tmpV2[0] + tmpV2[1];

        vecBlkCnt = S->vectorDimension & 3;
        while(vecBlkCnt > 0)
        {
           sigma = *pSigma + S->epsilon;
           tmp -= 0.5*log(2.0 * PI_F * sigma);
           tmp -= 0.5*(*pIn - *pTheta) * (*pIn - *pTheta) / sigma;

           pIn++;
           pTheta++;
           pSigma++;
           vecBlkCnt--;
        }

        *buffer++ = tmp;
        
        classBlkCnt--;
    }

    arm_max_f32(pBuffer,S->numberOfClasses,&result,&index);

    return(index);
}

#else

/**
 * @brief Naive Gaussian Bayesian Estimator
 *
 * @param[in]  *S         points to a naive bayes instance structure
 * @param[in]  *in        points to the elements of the input vector.
 * @param[in]  *pBuffer   points to a buffer of length numberOfClasses
 * @return The predicted class
 *
 */
uint32_t arm_gaussian_naive_bayes_predict_f32(const arm_gaussian_naive_bayes_instance_f32 *S, 
   const float32_t * in, 
   float32_t *pBuffer)
{
    int nbClass;
    int nbDim;
    const float32_t *pPrior = S->classPriors;
    const float32_t *pTheta = S->theta;
    const float32_t *pSigma = S->sigma;
    float32_t *buffer = pBuffer;
    const float32_t *pIn=in;
    float32_t result;
    float32_t sigma;
    float32_t tmp;
    float32_t acc1,acc2;
    uint32_t index;

    pTheta=S->theta;
    pSigma=S->sigma;

    for(nbClass = 0; nbClass < S->numberOfClasses; nbClass++)
    {

        
        pIn = in;

        tmp = log(*pPrior);
        acc1 = 0;
        acc2 = 0;
        for(nbDim = 0; nbDim < S->vectorDimension; nbDim++)
        {
           sigma = *pSigma + S->epsilon;
           acc1 += log(2.0 * PI_F * sigma);
           acc2 += (*pIn - *pTheta) * (*pIn - *pTheta) / sigma;

           pIn++;
           pTheta++;
           pSigma++;
        }

        tmp = -0.5 * acc1;
        tmp -= 0.5 * acc2;


        *buffer = tmp + log(*pPrior++);
        buffer++;
    }

    arm_max_f32(pBuffer,S->numberOfClasses,&result,&index);

    return(index);
}

#endif
/**
 * @} end of groupBayes group
 */
