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
 * Title:        arm_convolve_1_x_n_s8.c
 * Description:  s8 version of 1xN convolution using symmetric quantization.
 *
 * $Date:        20 June 2022
 * $Revision:    V.3.1.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * 1xN s8 convolution function.
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_1_x_n_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const q7_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const q7_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          q7_t *output_data)
{
    (void)bias_dims;
    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;
    /* The wrapper API is the ultimate reference for argument check */
    if ((input_dims->h != 1) || (output_dims->w % 4 != 0) || conv_params->dilation.w != 1)
    {
        status = ARM_CMSIS_NN_ARG_ERROR;
        goto out;
    }

#if defined(ARM_MATH_MVEI)
    (void)ctx;

    const uint16_t input_x = input_dims->w;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_ch = output_dims->c;
    const uint16_t input_ch = input_dims->c;
    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t stride_x = conv_params->stride.w;

    int i_batch;
    for (i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        for (int i_out_x = 0; i_out_x <= (output_x - 4); i_out_x += 4)
        {
            int32_t input_begin_idx[4];
            int32_t ker_begin_idx[4];
            int32_t ker_end_idx[4];

            for (int i = 0; i < 4; i++)
            {
                const int32_t est_input_x_idx = stride_x * (i_out_x + i) - pad_x;
                input_begin_idx[i] = MAX(0, est_input_x_idx);
                ker_begin_idx[i] = MAX(0, -est_input_x_idx);
                ker_end_idx[i] = MIN(kernel_x, input_x - est_input_x_idx);
            }

            if ((ker_begin_idx[0] != 0) || (ker_end_idx[3] != kernel_x))
            {
                for (int i = 0; i < 4; i++)
                {
                    const int32_t actual_kernel_len = ker_end_idx[i] - ker_begin_idx[i];
                    arm_nn_mat_mul_core_1x_s8(actual_kernel_len * input_ch,
                                              (kernel_x - actual_kernel_len) * input_ch,
                                              input_data + input_begin_idx[i] * input_ch,
                                              filter_data + (ker_begin_idx[i] * input_ch),
                                              output_ch,
                                              conv_params,
                                              quant_params,
                                              bias_data,
                                              output_data);
                    output_data += output_ch;
                }
            }
            else
            {
                output_data = arm_nn_mat_mul_core_4x_s8(kernel_x * input_ch,
                                                        stride_x * input_ch,
                                                        input_data + input_begin_idx[0] * input_ch,
                                                        filter_data,
                                                        output_ch,
                                                        conv_params,
                                                        quant_params,
                                                        bias_data,
                                                        output_data);
            }
        }
        /* Advance to the next batch */
        input_data += (input_x * input_ch);
    }

#else
    status = arm_convolve_s8(ctx,
                             conv_params,
                             quant_params,
                             input_dims,
                             input_data,
                             filter_dims,
                             filter_data,
                             bias_dims,
                             bias_data,
                             output_dims,
                             output_data);
#endif

out:
    /* Return to application */
    return status;
}

int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if !defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
#else
    (void)input_dims;
    (void)filter_dims;
    return 0;
#endif
}

/**
 * @} end of NNConv group
 */
