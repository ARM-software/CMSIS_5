/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_goertzel_init_f32.c
 * Description:  Floating-point Goertzel DFT initialization function
 *
 * $Date:        2019
 * $Revision:    V1.6.0
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2019 ARM Limited or its affiliates. All rights reserved.
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

/**
  @ingroup groupFilters
 */

/**
  @addtogroup Goertzel_DFTs
  @{
 */

/**
   * @brief  Initialization function for the floating-point Goertzel DFT.
   * @param[in,out] S        points to an instance of the floating-point Goertzel DFT structure.
   * @param[in]     w0       frequency to be analysed [rad/samples].
   * @param[in]     scaling  scaling factor of the output
   */

#include "arm_math.h"

void arm_goertzel_init_f32(
    arm_goertzel_instance_f32 *S,
    float32_t w0,
    float32_t * coeffs,
    float32_t scaling)
{
    S->cosine = arm_cos_f32(w0);
    S->sine   = arm_sin_f32(w0);

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
    float32_t c = 2*(S->cosine);
    coeffs[0] = 0;
    coeffs[1] = 0;
    coeffs[2] = 0;
    coeffs[3] = 1;
    coeffs[4] = c;
    coeffs[5] = c*c-1;
    coeffs[6] = c*c*c-2*c;
    coeffs[7] = c*c*c*c-3*c*c+1;
    S->coeffs = coeffs;

    /*
     * +---+---+---+---+---+-------+--------+------------+
     * | 0 | 0 | 0 | 1 | c | c^2-1 | c^3-2c | c^4-3c^2+1 |
     * +---+---+---+---+---+-------+--------+------------+
     * |   |   |   |   |             coeff1              |
     * |   |   |   |   <--------------------------------->
     * |   |   |   |    coeff3 = -coeff2    |
     * |   |   |   <------------------------>
     * |   |   |      coeff4       |
     * |   |   <------------------->
     * |   |     coeff5    |
     * |   <--------------->
     * |     coeff6    |
     * <--------------->
     */
#else
    /* To avoid warning */
    (void)coeffs;
#endif

    S->scaling = 1/(float32_t)scaling;
}

/**
  @} end of Goertzel_DFTs group
 */

