/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_pool_q7_HWC.c
 * Description:  Pooling function implementations
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"


  /**
   * @brief Q7 average pooling function
   * @param[in]       dim_im_in_height   input tensor dimention
   * @param[in]       dim_im_in_width    input tensor dimention
   * @param[in]       dim_im_out_height  output tensor dimension
   * @param[in]       dim_im_out_width   output tensor dimension
   * @param[in]       stride_height      stride
   * @param[in]       stride_width       stride
   * @param[in]       dim_kernel_height  filter kernel size
   * @param[in]       dim_kernel_width   filter kernel size
   * @param[in]       padding_height     padding sizes
   * @param[in]       padding_width      padding sizes
   * @param[in]       act_min            Min clamping
   * @param[in]       act_max            Max clamping
   * @param[in]       ch_im_in           number of input tensor channels
   * @param[in,out]   Im_in              pointer to input tensor
   * @param[in,out]   bufferA            temp buffer
   * @param[in,out]   Im_out             pointer to output tensor
   * @return none.
   *
   * @details
   *
   *
   */
static void accumulate_q7_to_q15(q15_t * base, q7_t * target, const uint16_t length)
{
    q15_t    *pCnt = base;
    q7_t     *pV = target;
    q31_t     v1, v2, vo1, vo2;
    uint16_t  cnt = length >> 2;
    q31_t     in;

    while (cnt > 0u)
    {
        q31_t     value = *__SIMD32(pV)++;
        v1 = __SXTB16(__ROR(value, 8));
        v2 = __SXTB16(value);
#ifndef ARM_MATH_BIG_ENDIAN

        vo2 = __PKHTB(v1, v2, 16);
        vo1 = __PKHBT(v2, v1, 16);

#else

        vo1 = __PKHTB(v1, v2, 16);
        vo2 = __PKHBT(v2, v1, 16);

#endif

        in = *__SIMD32(pCnt);
        *__SIMD32(pCnt)++ = __QADD16(vo1, in);

        in = *__SIMD32(pCnt);
        *__SIMD32(pCnt)++ = __QADD16(vo2, in);

        cnt--;
    }
    cnt = length & 0x3;
    while (cnt > 0u)
    {
        *pCnt++ += *pV++;
        cnt--;
    }
}

static void buffer_scale_back_q15_to_q7(q15_t * buffer, q7_t * target, uint16_t length, uint16_t scale)
{
    int       i;

    for (i = 0; i < length; i++)
    {
        
        target[i] = (q7_t) (buffer[i] / scale);
    }
}

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

void
arm_avgpool_s8( const int dim_im_in_height,
  const int dim_im_in_width,
  const int dim_im_out_height,
  const int dim_im_out_width,
  const int stride_height,
  const int stride_width,
  const int dim_kernel_height,
  const int dim_kernel_width,
  const int padding_height,
  const int padding_width,
  const int act_min,
  const int act_max,
  const int ch_im_in,
  int8_t *Im_in,
  int16_t *bufferA,
  int8_t *Im_out)
{

#if defined (ARM_MATH_DSP)

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    q15_t    *buffer = (q15_t *) bufferA;
    int16_t   i_x, i_y;
    int16_t   count = 0;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < dim_im_in_height; i_y++)
    {

        for (i_x = 0; i_x < dim_im_out_width; i_x++)
        {
            /* for each output pixel */
            q7_t     *target = Im_in + (i_y * dim_im_in_width + i_x) * ch_im_in;
            q7_t     *win_start;
            q7_t     *win_stop;
            if (i_x * stride_width - padding_width < 0)
            {
                win_start = target;
            } else
            {
                win_start = Im_in + (i_y * dim_im_in_width + i_x * stride_width - padding_width) * ch_im_in;
            }

            if (i_x * stride_width - padding_width + dim_kernel_width >= dim_im_in_width)
            {
                win_stop = Im_in + (i_y * dim_im_in_width + dim_im_in_width) * ch_im_in;
            } else
            {
                win_stop = Im_in + (i_y * dim_im_in_width + i_x * stride_width - padding_width + dim_kernel_width) * ch_im_in;
            }
            /* first step is to copy over initial data */
            arm_q7_to_q15_no_shift(win_start, buffer, ch_im_in);
            count = 1;

            /* start the max operation from the second part */
            win_start += ch_im_in;
            for (; win_start < win_stop; win_start += ch_im_in)
            { 
                accumulate_q7_to_q15(buffer, win_start, ch_im_in);
                count++;
            }
            buffer_scale_back_q15_to_q7(buffer, target, ch_im_in, count);
        }
    }


    /* then does the pooling along y axis */
    for (i_y = 0; i_y < dim_im_out_height; i_y++)
    {
        /* for each output row */
        q7_t     *target = Im_out + i_y * dim_im_out_width * ch_im_in;
        q7_t     *row_start;
        q7_t     *row_end;
        /* setting the starting row */
        if (i_y * stride_height - padding_height < 0)
        {
            row_start = Im_in;
        } else
        {
            row_start = Im_in + (i_y * stride_height - padding_height) * dim_im_in_width * ch_im_in;
        }
        /* setting the stopping row */
        if (i_y * stride_height - padding_height + dim_kernel_height >= dim_im_in_height)
        {
            row_end = Im_in + dim_im_in_height * dim_im_in_width * ch_im_in;
        } else
        {
            row_end = Im_in + (i_y * stride_height - padding_height + dim_kernel_height) * dim_im_in_width * ch_im_in;
        }

        /* copy over the first row */
        arm_q7_to_q15_no_shift(row_start, buffer, dim_im_out_width * ch_im_in);
        count = 1;
        //("sum %d\n",buffer[0]);

        /* move over to next row */
        row_start += ch_im_in * dim_im_in_width;

        for (; row_start < row_end; row_start += dim_im_in_width * ch_im_in)
        {
            accumulate_q7_to_q15(buffer, row_start, dim_im_out_width * ch_im_in);

            count++;
        }
        buffer_scale_back_q15_to_q7_and_clamp(buffer, target, dim_im_out_width * ch_im_in, count,act_min,act_max);
    }

#else

/* Reference C code adapted from CMSIS-NN arm_avepool_q7_HWC.
 */
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    
    for (i_y = 0; i_y < dim_im_out_height; i_y++)
    {
        for (i_x = 0; i_x < dim_im_out_width; i_x++)
        {
            for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
            {
                int       sum = 0;
                int       count = 0;
                for (k_y = i_y * stride_height - padding_height; k_y < i_y * stride_height - padding_height + dim_kernel_height; k_y++)
                {
                    for (k_x = i_x * stride_width - padding_width; k_x < i_x * stride_width - padding_width + dim_kernel_width; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_height && k_x < dim_im_in_width)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_width)];
                            count++;
                        }
                    }
                }
                // Round to the closest integer value.
                sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                sum = MAX(sum, act_min);
                sum = MIN(sum, act_max);

                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_width)] = sum;
            }
        }
    }
#endif
}

/**
 * @} end of Pooling group
 */
