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

#include "ref_functions.h"

void arm_avepool_q7_HWC_ref(const q7_t * Im_in, // input image
                            const uint16_t dim_im_in,   // input image dimension
                            const uint16_t ch_im_in,    // number of input image channels
                            const uint16_t dim_kernel,  // window kernel size
                            const uint16_t padding, // padding sizes
                            const uint16_t stride,  // stride
                            const uint16_t dim_im_out,  // output image dimension
                            q7_t * bufferA, // a buffer for local storage
                            q7_t * Im_out)
{
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int       sum = 0;
                int       count = 0;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            count++;
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = sum / count;
            }
        }
    }
}

void arm_maxpool_q7_HWC_ref(const q7_t * Im_in, // input image
                            const uint16_t dim_im_in,   // input image dimension
                            const uint16_t ch_im_in,    // number of input image channels
                            const uint16_t dim_kernel,  // window kernel size
                            const uint16_t padding, // padding sizes
                            const uint16_t stride,  // stride
                            const uint16_t dim_im_out,  // output image dimension
                            q7_t * bufferA, // a buffer for local storage
                            q7_t * Im_out)
{
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int       max = -129;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = max;
            }
        }
    }
}

void
arm_avepool_q7_HWC_nonsquare_ref(q7_t * Im_in,
                   const uint16_t dim_im_in_x,
                   const uint16_t dim_im_in_y,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel_x,
                   const uint16_t dim_kernel_y,
                   const uint16_t padding_x,
                   const uint16_t padding_y,
                   const uint16_t stride_x,
                   const uint16_t stride_y,
                   const uint16_t dim_im_out_x,
                   const uint16_t dim_im_out_y,
                   q7_t * bufferA, 
                   q7_t * Im_out) {
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int sum = 0;
                int count = 0;
                int16_t y_start = i_y * stride_y - padding_y;
                int16_t x_start = i_x * stride_x - padding_x;
                for (k_y = y_start; k_y < y_start + dim_kernel_y; k_y++)
                {
                    for (k_x = x_start; k_x < x_start + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            count++
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum/count;
            }
        }
    }
}


void
arm_maxpool_q7_HWC_nonsquare_ref(q7_t * Im_in,
                   const uint16_t dim_im_in_x,
                   const uint16_t dim_im_in_y,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel_x,
                   const uint16_t dim_kernel_y,
                   const uint16_t padding_x,
                   const uint16_t padding_y,
                   const uint16_t stride_x,
                   const uint16_t stride_y,
                   const uint16_t dim_im_out_x,
                   const uint16_t dim_im_out_y,
                   q7_t * bufferA, 
                   q7_t * Im_out) {
   /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int       max = -129;
                int16_t y_start = i_y * stride_y - padding_y;
                int16_t x_start = i_x * stride_x - padding_x;
                for (k_y = y_start; k_y < y_start + dim_kernel_y; k_y++)
                {
                    for (k_x = x_start; k_x < x_start + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }
}
