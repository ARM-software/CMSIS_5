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
 * Title:        arm_avgpool_s8.c
 * Description:  Pooling function implementations
 *
 * $Date:        7 July 2022
 * $Revision:    V.3.0.2
 *
 * Target Processor:  Cortex-M CPUs
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
static void scale_q31_to_q7_and_clamp(const q31_t *buffer,
                                      q7_t *target,
                                      int32_t length,
                                      const int32_t count,
                                      const int act_min,
                                      const int act_max)
{
    const int half_count = count / 2;

    for (int i = 0; i < length; i++)
    {
        int32_t sum = buffer[i] > 0 ? (buffer[i] + half_count) : (buffer[i] - half_count);
        sum = sum / count;
        sum = MAX(sum, act_min);
        sum = MIN(sum, act_max);

        target[i] = (q7_t)sum;
    }
}
#endif

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Pooling
 * @{
 */

/*
 * s8 average pooling function
 *
 * Refer to header file for details.
 *
 */

#if defined(ARM_MATH_MVEI)

arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx,
                                   const cmsis_nn_pool_params *pool_params,
                                   const cmsis_nn_dims *input_dims,
                                   const q7_t *src,
                                   const cmsis_nn_dims *filter_dims,
                                   const cmsis_nn_dims *output_dims,
                                   q7_t *dst)
{
    (void)ctx;
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const int32_t act_min = pool_params->activation.min;
    const int32_t act_max = pool_params->activation.max;
    const int32_t ch_src = input_dims->c;

    for (int i_y = 0; i_y < output_y; i_y++)
    {
        for (int i_x = 0; i_x < output_x; i_x++)
        {
            const int32_t k_y_start = MAX(0, i_y * stride_y - pad_y);
            const int32_t k_y_end = MIN(i_y * stride_y - pad_y + kernel_y, input_y);

            const int32_t k_x_start = MAX(0, i_x * stride_x - pad_x);
            const int32_t k_x_end = MIN(i_x * stride_x - pad_x + kernel_x, input_x);

            const int8_t *src_base = src;
            int8_t *out = &dst[ch_src * (i_x + i_y * output_x)];

            int32_t ch_count = (ch_src + 15) / 16;
            int32_t channels = ch_src;

            while (ch_count > 0)
            {
                int8x16_t temp;
                int16x8_t temp_lo, temp_hi;
                int32x4_t temp_lo_lo, temp_lo_hi, temp_hi_lo, temp_hi_hi;
                int32_t count = 0;

                int32x4_t sum_1 = vdupq_n_s32(0);
                int32x4_t sum_2 = vdupq_n_s32(0);
                int32x4_t sum_3 = vdupq_n_s32(0);
                int32x4_t sum_4 = vdupq_n_s32(0);
                // Load store tail predicate
                const mve_pred16_t ld_st_p = vctp8q(channels);
                channels -= 16;

                for (int k_y = k_y_start; k_y < k_y_end; k_y++)
                {
                    for (int k_x = k_x_start; k_x < k_x_end; k_x++)
                    {
                        const int8_t *src_inner = src_base + (ch_src * (k_x + k_y * input_x));
                        temp = vldrbq_z_s8(src_inner, ld_st_p);

                        temp_lo = vmovlbq_s8(temp);
                        temp_hi = vmovltq_s8(temp);

                        temp_lo_lo = vmovlbq_s16(temp_lo);
                        temp_lo_hi = vmovltq_s16(temp_lo);

                        temp_hi_lo = vmovlbq_s16(temp_hi);
                        temp_hi_hi = vmovltq_s16(temp_hi);

                        sum_1 = vaddq_s32(sum_1, temp_lo_lo);
                        sum_2 = vaddq_s32(sum_2, temp_lo_hi);
                        sum_3 = vaddq_s32(sum_3, temp_hi_lo);
                        sum_4 = vaddq_s32(sum_4, temp_hi_hi);

                        count++;
                    }
                }

                // Prevent static code issue DIVIDE_BY_ZERO.
                if (count == 0)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }

                // Perform the following operation
                // sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                const int32_t half_count = count / 2;
                // Predicate for 'sum > 0' operation
                mve_pred16_t p = vcmpgtq_n_s32(sum_1, 0);
                sum_1 = vaddq_m_n_s32(sum_1, sum_1, half_count, p);
                sum_1 = vsubq_m_n_s32(sum_1, sum_1, half_count, ~p);

                p = vcmpgtq_n_s32(sum_2, 0);
                sum_2 = vaddq_m_n_s32(sum_2, sum_2, half_count, p);
                sum_2 = vsubq_m_n_s32(sum_2, sum_2, half_count, ~p);

                p = vcmpgtq_n_s32(sum_3, 0);
                sum_3 = vaddq_m_n_s32(sum_3, sum_3, half_count, p);
                sum_3 = vsubq_m_n_s32(sum_3, sum_3, half_count, ~p);

                p = vcmpgtq_n_s32(sum_4, 0);
                sum_4 = vaddq_m_n_s32(sum_4, sum_4, half_count, p);
                sum_4 = vsubq_m_n_s32(sum_4, sum_4, half_count, ~p);

                for (int i = 0; i < 4; i++)
                {
                    sum_1[i] = sum_1[i] / count;
                    sum_2[i] = sum_2[i] / count;
                    sum_3[i] = sum_3[i] / count;
                    sum_4[i] = sum_4[i] / count;
                }

                sum_1 = vmaxq_s32(sum_1, vdupq_n_s32(act_min));
                sum_1 = vminq_s32(sum_1, vdupq_n_s32(act_max));

                sum_2 = vmaxq_s32(sum_2, vdupq_n_s32(act_min));
                sum_2 = vminq_s32(sum_2, vdupq_n_s32(act_max));

                sum_3 = vmaxq_s32(sum_3, vdupq_n_s32(act_min));
                sum_3 = vminq_s32(sum_3, vdupq_n_s32(act_max));

                sum_4 = vmaxq_s32(sum_4, vdupq_n_s32(act_min));
                sum_4 = vminq_s32(sum_4, vdupq_n_s32(act_max));

                temp_lo = vmovnbq_s32(temp_lo, sum_1);
                temp_lo = vmovntq_s32(temp_lo, sum_2);

                temp_hi = vmovnbq_s32(temp_hi, sum_3);
                temp_hi = vmovntq_s32(temp_hi, sum_4);

                temp = vmovnbq_s16(temp, temp_lo);
                temp = vmovntq_s16(temp, temp_hi);

                vstrbq_p_s8(out, temp, ld_st_p);
                out += 16;

                ch_count--;
                src_base += 16;
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#else
arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx,
                                   const cmsis_nn_pool_params *pool_params,
                                   const cmsis_nn_dims *input_dims,
                                   const q7_t *src,
                                   const cmsis_nn_dims *filter_dims,
                                   const cmsis_nn_dims *output_dims,
                                   q7_t *dst)
{
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const int32_t act_min = pool_params->activation.min;
    const int32_t act_max = pool_params->activation.max;
    const int32_t ch_src = input_dims->c;

    if (ctx->buf == NULL && arm_avgpool_s8_get_buffer_size(output_dims->w, input_dims->c))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    q31_t *buffer = (q31_t *)ctx->buf;

#if defined(ARM_MATH_DSP)

    /* Run the following code for CPU's with DSP extension
     */
    for (int i_y = 0, idx_y = -pad_y; i_y < output_y; idx_y += stride_y, i_y++)
    {
        for (int i_x = 0, idx_x = -pad_x; i_x < output_x; idx_x += stride_x, i_x++)
        {
            /* Condition for kernel start dimension:
                      (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
            const int32_t kernel_y_start = MAX(0, -idx_y);
            const int32_t kernel_x_start = MAX(0, -idx_x);

            /* Condition for kernel end dimension:
                   (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
            const int32_t kernel_y_end = MIN(kernel_y, input_y - idx_y);
            const int32_t kernel_x_end = MIN(kernel_x, input_x - idx_x);

            int count = 0;

            for (int k_y = kernel_y_start; k_y < kernel_y_end; k_y++)
            {
                for (int k_x = kernel_x_start; k_x < kernel_x_end; k_x++)
                {
                    const q7_t *start = src + ch_src * (k_x + idx_x + (k_y + idx_y) * input_x);

                    if (count == 0)
                    {
                        for (int i = 0; i < ch_src; i++)
                        {
                            buffer[i] = start[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < ch_src; i++)
                        {
                            buffer[i] = __QADD(start[i], buffer[i]);
                        }
                    }
                    count++;
                }
            }

            // Prevent static code issue DIVIDE_BY_ZERO.
            if (count == 0)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }

            scale_q31_to_q7_and_clamp(buffer, dst, ch_src, count, act_min, act_max);
            dst += ch_src;
        }
    }
#else

    /* Reference C code adapted from CMSIS-NN arm_avepool_q7_HWC.
     */
    (void)buffer;

    for (int i_y = 0; i_y < output_y; i_y++)
    {
        for (int i_x = 0; i_x < output_x; i_x++)
        {
            for (int i_ch_in = 0; i_ch_in < ch_src; i_ch_in++)
            {
                int sum = 0;
                int count = 0;
                for (int k_y = i_y * stride_y - pad_y; k_y < i_y * stride_y - pad_y + kernel_y; k_y++)
                {
                    for (int k_x = i_x * stride_x - pad_x; k_x < i_x * stride_x - pad_x + kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < input_y && k_x < input_x)
                        {
                            sum += src[i_ch_in + ch_src * (k_x + k_y * input_x)];
                            count++;
                        }
                    }
                }

                // Prevent static code issue DIVIDE_BY_ZERO.
                if (count == 0)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }

                sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                sum = MAX(sum, act_min);
                sum = MIN(sum, act_max);

                dst[i_ch_in + ch_src * (i_x + i_y * output_x)] = sum;
            }
        }
    }

#endif
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_MATH_MVEI */

int32_t arm_avgpool_s8_get_buffer_size(const int output_x, const int ch_src)
{
    (void)output_x;

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return (ch_src * sizeof(int32_t));
#else
    (void)ch_src;
    return 0;
#endif
}
/**
 * @} end of Pooling group
 */
