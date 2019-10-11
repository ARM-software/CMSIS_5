/*
 * Copyright (C) 2010-2019 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_softmax_s8.c
 * Description:  S8 softmax function
 *
 * $Date:        October 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

#define ACCUM_BITS 12

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Softmax
 * @{
 */
void arm_softmax_s8(const int8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int8_t diff_min,
                    int8_t *output)
{
    const int32_t mask = (1 << shift);

    uint16_t row = 0;
    uint16_t col = 0;
    for(row = 0; row < num_rows; ++row)
    {
        const int32_t row_idx = row * row_size;

        int8_t max = input[row_idx];
        for (col = 1; col < row_size; ++col)
        {
            max = MAX(max, input[row_idx + col]);
        }

        int32_t sum = 0;
        for (col = 0; col < row_size; ++col)
        {
            const int8_t diff = input[row_idx + col] - max;
            if (diff >= diff_min)
            {
                sum += DIV_POW2(EXP_ON_NEG(MUL_SAT(diff * mask, mult)), ACCUM_BITS);
            }
        }

        const int32_t headroom = __CLZ(sum);
        const int32_t bits_over_unit = ACCUM_BITS - headroom;
        const int32_t shifted_scale = ONE_OVER1((sum << headroom) - (1 << 31));
        for (col = 0; col < row_size; ++col)
        {
            const int8_t diff = input[row_idx + col] - max;
            if (diff >= diff_min)
            {
                const int32_t out_val = DIV_POW2(MUL_SAT(shifted_scale, EXP_ON_NEG(MUL_SAT(diff * mask, mult))), bits_over_unit + 23) - 128;
                output[row_idx + col] = (int8_t) CLAMP(out_val, (int32_t)127, (int32_t)-128);
            }
            else
            {
                output[row_idx + col] = -128;
            }
        }
    }
}