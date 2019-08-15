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
 * Title:        arm_avgpool_s8.c
 * Description:  Pooling function implementations
 *
 * $Date:        29. July 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M and Cortex-A cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"


static void buffer_scale_back_q15_to_q7(q15_t * buffer, q7_t * target, uint16_t length, uint16_t scale)
{
    int       i;

    for (i = 0; i < length; i++)
    {

        target[i] = (q7_t) (buffer[i] / scale);
    }
}

#if defined (ARM_MATH_DSP)

static void buffer_scale_back_q15_to_q7_and_clamp(q15_t * buffer, q7_t * target, uint16_t length, uint16_t count,const int act_min,
  const int act_max)
{
    int       i;
    int sum;

    for (i = 0; i < length; i++)
    {
        sum = buffer[i] > 0 ? (buffer[i] + count / 2) / count : (buffer[i] - count / 2) / count;

        sum = MAX(sum, act_min);
        sum = MIN(sum, act_max);

        target[i] = (q7_t) (sum);
    }
}
#endif

    /**
   * @brief s8 average pooling function
   *
   * @details Refer to header file for details.
   *
   */
void arm_avgpool_s8(const int dim_src_height,
                    const int dim_src_width,
                    const int dim_dst_height,
                    const int dim_dst_width,
                    const int stride_height,
                    const int stride_width,
                    const int dim_kernel_height,
                    const int dim_kernel_width,
                    const int padding_height,
                    const int padding_width,
                    const int act_min,
                    const int act_max,
                    const int ch_src,
                    int8_t *src,
                    int16_t *bufferA,
                    int8_t *dst)
{

#if defined(ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP)

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    q15_t    *buffer = (q15_t *) bufferA;
    int16_t   i_x, i_y;
    int16_t   count = 0;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < dim_src_height; i_y++)
    {

        for (i_x = 0; i_x < dim_dst_width; i_x++)
        {
            /* for each output sample */
            q7_t     *target = src + (i_y * dim_src_width + i_x) * ch_src;
            q7_t     *win_start;
            q7_t     *win_stop;
            if (i_x * stride_width - padding_width < 0)
            {
                win_start = target;
            } else
            {
                win_start = src + (i_y * dim_src_width + i_x * stride_width - padding_width) * ch_src;
            }

            if (i_x * stride_width - padding_width + dim_kernel_width >= dim_src_width)
            {
                win_stop = src + (i_y * dim_src_width + dim_src_width) * ch_src;
            } else
            {
                win_stop = src + (i_y * dim_src_width + i_x * stride_width - padding_width + dim_kernel_width) * ch_src;
            }
            /* first step is to copy over initial data */
            arm_q7_to_q15_no_shift(win_start, buffer, ch_src);
            count = 1;

            /* start the average operation from the second part */
            win_start += ch_src;
            for (; win_start < win_stop; win_start += ch_src)
            {
                arm_nn_accumulate_q7_to_q15(buffer, win_start, ch_src);
                count++;
            }
            buffer_scale_back_q15_to_q7(buffer, target, ch_src, count);
        }
    }


    /* then does the pooling along y axis */
    for (i_y = 0; i_y < dim_dst_height; i_y++)
    {
        /* for each output row */
        q7_t     *target = dst + i_y * dim_dst_width * ch_src;
        q7_t     *row_start;
        q7_t     *row_end;
        /* setting the starting row */
        if (i_y * stride_height - padding_height < 0)
        {
            row_start = src;
        } else
        {
            row_start = src + (i_y * stride_height - padding_height) * dim_src_width * ch_src;
        }
        /* setting the stopping row */
        if (i_y * stride_height - padding_height + dim_kernel_height >= dim_src_height)
        {
            row_end = src + dim_src_height * dim_src_width * ch_src;
        } else
        {
            row_end = src + (i_y * stride_height - padding_height + dim_kernel_height) * dim_src_width * ch_src;
        }

        /* copy over the first row */
        arm_q7_to_q15_no_shift(row_start, buffer, dim_dst_width * ch_src);
        count = 1;

        /* move over to next row */
        row_start += ch_src * dim_src_width;

        for (; row_start < row_end; row_start += dim_src_width * ch_src)
        {
            arm_nn_accumulate_q7_to_q15(buffer, row_start, dim_dst_width * ch_src);

            count++;
        }
        buffer_scale_back_q15_to_q7_and_clamp(buffer, target, dim_dst_width * ch_src, count,act_min,act_max);
    }

#else

/* Reference C code adapted from CMSIS-NN arm_avepool_q7_HWC.
 */
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;


    for (i_y = 0; i_y < dim_dst_height; i_y++)
    {
        for (i_x = 0; i_x < dim_dst_width; i_x++)
        {
            for (i_ch_in = 0; i_ch_in < ch_src; i_ch_in++)
            {
                int       sum = 0;
                int       count = 0;
                for (k_y = i_y * stride_height - padding_height; k_y < i_y * stride_height - padding_height + dim_kernel_height; k_y++)
                {
                    for (k_x = i_x * stride_width - padding_width; k_x < i_x * stride_width - padding_width + dim_kernel_width; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_src_height && k_x < dim_src_width)
                        {
                            sum += src[i_ch_in + ch_src * (k_x + k_y * dim_src_width)];
                            count++;
                        }
                    }
                }
                sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                sum = MAX(sum, act_min);
                sum = MIN(sum, act_max);

                dst[i_ch_in + ch_src * (i_x + i_y * dim_dst_width)] = sum;
            }
        }
    }
#endif
}

/**
 * @} end of Pooling group
 */
