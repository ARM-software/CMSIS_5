/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_fully_connected_s8
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        April 2, 2020
 * $Revision:    V.1.5.0
 *
 * Target Processor:  Cortex-M and Cortex-A cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup FC
 * @{
 */

/*
   * S8 basic fully-connected and matrix multiplication layer function for TensorFlow Lite
   *
   * Refer header file for details.
   *
   */

arm_status
arm_fully_connected_s8(const int8_t *input,
                       const int8_t *kernel,
                       const uint16_t col_dim,
                       const uint16_t row_dim,
                       const uint16_t nb_batches,
                       const int32_t input_offset,
                       const int32_t filter_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t output_offset,
                       const int32_t *bias,
                       int8_t *output,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       q15_t *vec_buffer)
{

    (void)vec_buffer;

    int32_t batch_cnt = nb_batches;

    while (batch_cnt)
    {
        arm_nn_vec_mat_mult_t_s8(input,
                                 kernel,
                                 bias,
                                 output,
                                 input_offset,
                                 filter_offset,
                                 output_offset,
                                 out_mult,
                                 out_shift,
                                 col_dim,
                                 row_dim,
                                 output_activation_min,
                                 output_activation_max);
        input += col_dim;
        output += row_dim;
        batch_cnt--;
    }
    return (ARM_MATH_SUCCESS);
}


int32_t arm_fully_connected_s8_get_buffer_size(const uint16_t col_dim)
{
    (void)col_dim;
    return 0;
}

/**
 * @} end of FC group
 */
