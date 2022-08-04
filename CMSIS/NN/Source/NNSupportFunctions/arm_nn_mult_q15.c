/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_mult_q15.c
 * Description:  Q15 vector multiplication with variable output shifts
 *
 * $Date:        4 Aug 2022
 * $Revision:    V.1.1.3
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup NNBasicMath
 * @{
 */

/*
 * Q7 vector multiplication with variable output shifts
 * Refer function header for details
 *
 */

void arm_nn_mult_q15(q15_t *pSrcA, q15_t *pSrcB, q15_t *pDst, const uint16_t out_shift, uint32_t blockSize)
{
    uint32_t blkCnt = blockSize; /* loop counters */

    while (blkCnt > 0U)
    {
        /* C = A * B */
        /* Multiply the inputs and store the result in the destination buffer */
        *pDst++ = (q15_t)__SSAT(((q31_t)((q31_t)(*pSrcA++) * (*pSrcB++) + NN_ROUND(out_shift)) >> out_shift), 16);

        /* Decrement the blockSize loop counter */
        blkCnt--;
    }
}

/**
 * @} end of NNBasicMath group
 */
