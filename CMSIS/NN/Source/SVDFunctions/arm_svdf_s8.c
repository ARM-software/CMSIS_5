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
 * Title:        arm_svdf_s8.c
 * Description:  S8 basic SVDF layer function
 *
 * $Date:        17. August 2020
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M processors
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nn_types.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupNN
 */

/**
 * @addtogroup SVDF
 * @{
 */

/*
   * S8 SVDF layer function for TensorFlow Lite
   *
   * Refer to header file for details.
   *
   */

arm_status
arm_svdf_s8(const cmsis_nn_context *input_ctx,
            const cmsis_nn_context *output_ctx,
            const cmsis_nn_svdf_params *svdf_params,
            const cmsis_nn_per_tensor_quant_params *input_quant_params,
            const cmsis_nn_per_tensor_quant_params *output_quant_params,
            const cmsis_nn_dims *input_dims,
            const q7_t *input_data,
            const cmsis_nn_dims *state_dims,
            q15_t *state_data,
            const cmsis_nn_dims *weights_feature_dims,
            const q7_t *weights_feature_data,
            const cmsis_nn_dims *weights_time_dims,
            const q15_t *weights_time_data,
            const cmsis_nn_dims *bias_dims,
            const q31_t *bias_data,
            const cmsis_nn_dims *output_dims,
            q7_t *output_data)
{
  (void)bias_dims;
  (void)state_dims;
  (void)output_dims;

  const q31_t multiplier_in = input_quant_params->multiplier;
  const q31_t shift_in = input_quant_params->shift;
  const q31_t multiplier_out = output_quant_params->multiplier;
  const q31_t shift_2 = output_quant_params->shift;
  const int32_t zp_in = svdf_params->input_offset;
  const int32_t zp_out = svdf_params->output_offset;
  const int32_t in_activation_min = svdf_params->input_activation.min;
  const int32_t in_activation_max = svdf_params->input_activation.max;
  const int32_t out_activation_min = svdf_params->output_activation.min;
  const int32_t out_activation_max = svdf_params->output_activation.max;
  const int16_t rank = svdf_params->rank;

  int32_t zp_32 = (-zp_in & 0xffff) |
                 ((-zp_in & 0xffff) << 16);

  const int32_t input_batches = input_dims->n;
  const int32_t input_height = input_dims->h;
  const int32_t feature_batches = weights_feature_dims->n;
  const int32_t time_batches = weights_time_dims->h;
  const int32_t unit_count = feature_batches / rank;

  q31_t *buffer_a = (q31_t *)input_ctx->buf;
  q31_t *buffer_b = (q31_t *)output_ctx->buf;

  memmove((q15_t *)state_data, (q15_t *)state_data + 1,
          (size_t)(input_batches * feature_batches * time_batches * (int32_t)sizeof(int16_t)));

  q15_t *res_ptr = state_data + (time_batches - 1);
  for (int i_batch = 0; i_batch < input_batches; i_batch++)
  {
    const q7_t *buffer_1 = weights_feature_data;
    for (int r = 0; r < feature_batches; r++)
    {
      q31_t dot_prod = 0;

      const q7_t *buffer_2 = input_data + i_batch * input_height;

#if defined(ARM_MATH_DSP)
      int c = 0;
      int32_t block_count = input_height >> 2;
      for (int i = 0; i < block_count; i++)
      {
        c += 4;

        q31_t r1 = arm_nn_read_q7x4_ia(&buffer_1);
        q31_t r1_a = __SXTB16(r1);
        q31_t r1_b = __SXTB16(__ROR((uint32_t)r1, 8));

        q31_t r2 = arm_nn_read_q7x4_ia(&buffer_2);
        q31_t r2_a = __SXTAB16(zp_32, r2);
        q31_t r2_b = __SXTAB16(zp_32, __ROR((uint32_t)r2, 8));

        dot_prod = __SMLAD(r1_a, r2_a, dot_prod);
        dot_prod = __SMLAD(r1_b, r2_b, dot_prod);
      }

      for (; c < input_height; c++)
      {
        dot_prod += *buffer_1 * (*buffer_2 - zp_in);
        buffer_1++;
        buffer_2++;
      }
#else
      for (int c = 0; c < input_height; c++)
      {
        dot_prod += *buffer_1 * (*buffer_2 - zp_in);
        buffer_1++;
        buffer_2++;
      }
#endif

      dot_prod = arm_nn_requantize(dot_prod,
                                   multiplier_in,
                                   shift_in);
      dot_prod = CLAMP(dot_prod, in_activation_max, in_activation_min);
      *res_ptr = dot_prod;
      res_ptr += time_batches;
    }
  }

  for (int i_batch = 0; i_batch < input_batches; i_batch++)
  {
    q31_t *ptr_a = buffer_a + i_batch * feature_batches;

    const q15_t *v1 = weights_time_data;
    const q15_t *v2 = state_data + i_batch * time_batches * feature_batches;
    for (int i_feature_batch = 0; i_feature_batch < feature_batches; i_feature_batch++)
    {
      *ptr_a = 0;

      int32_t sum = 0;
#if defined(ARM_MATH_DSP)
      int j = 0;
      int32_t block_count = time_batches >> 1;
      for (int i = 0; i < block_count; i++)
      {
        j += 2;
        q31_t r1 = arm_nn_read_q15x2_ia(&v1);
        q31_t r2 = arm_nn_read_q15x2_ia(&v2);

        sum = __SMLAD(r1, r2, sum);
      }

      // Process the remaining data
      for (; j < time_batches; j++)
      {
        sum += *v1 * *v2;
        v1++;
        v2++;
      }
#else
      for (int j = 0; j < time_batches; j++)
      {
        sum += *v1 * *v2;
        v1++;
        v2++;
      }
#endif

      *ptr_a = sum;
      ptr_a++;
    }
  }

  for (int i_batch = 0; i_batch < input_batches; i_batch++)
  {
    q31_t *output_data_temp = buffer_b + i_batch * unit_count;
    q31_t *ptr_a = buffer_a + i_batch * feature_batches;

    for (int i = 0; i < unit_count; i++)
    {
      output_data_temp[i] = bias_data[i];
      for (int j = 0; j < rank; j++)
      {
        output_data_temp[i] += *ptr_a;
        ptr_a++;
      }
    }
  }

  for (int i = 0; i < input_batches * unit_count; i++)
  {
    output_data[i] = (q7_t)CLAMP(arm_nn_requantize(buffer_b[i], multiplier_out, shift_2) + zp_out,
                          out_activation_max, out_activation_min);
  }

  return (ARM_MATH_SUCCESS);
}

/**
 * @} end of SVDF group
 */
