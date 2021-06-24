/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_logsumexp_f16.c
 * Description:  LogSumExp
 *
 * $Date:        23 April 2021
 * $Revision:    V1.9.0
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

#include "dsp/statistics_functions_f16.h"

#if defined(ARM_FLOAT16_SUPPORTED)

#include <limits.h>
#include <math.h>

/**
  @ingroup groupStats
 */

/**
  @defgroup LogSumExp LogSumExp

  LogSumExp optimizations to compute sum of probabilities with Gaussian distributions

 */

/**
 * @addtogroup LogSumExp
 * @{
 */


/**
 * @brief Dot product with log arithmetic
 *
 * Vectors are containing the log of the samples
 *
 * @param[in]       *pSrcA points to the first input vector
 * @param[in]       *pSrcB points to the second input vector
 * @param[in]       blockSize number of samples in each vector
 * @param[in]       *pTmpBuffer temporary buffer of length blockSize
 * @return The log of the dot product.
 *
 */


float16_t arm_logsumexp_dot_prod_f16(const float16_t * pSrcA,
  const float16_t * pSrcB,
  uint32_t blockSize,
  float16_t *pTmpBuffer)
{
    float16_t result;
    arm_add_f16((float16_t*)pSrcA, (float16_t*)pSrcB, pTmpBuffer, blockSize);

    result = arm_logsumexp_f16(pTmpBuffer, blockSize);
    return(result);
}

/**
 * @} end of LogSumExp group
 */

#endif /* #if defined(ARM_FLOAT16_SUPPORTED) */ 

