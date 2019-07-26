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
   * @param[in,out]   Im_out             pointer to output tensor
   * @return none.
   *
   * @details
   *
   *
   */

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
  const int8_t *Im_in,
  int8_t *Im_out)
{

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
}

/**
 * @} end of Pooling group
 */
