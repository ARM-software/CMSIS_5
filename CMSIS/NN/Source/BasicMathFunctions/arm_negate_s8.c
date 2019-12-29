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
 * Title:        arm_negate_s8
 * Description:  Negate
 *
 * $Date:        December 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup BasicMath
 * @{
 */

/**
 * @brief s8 element-wise negation of a vector
 *
 * @note   Refer header file for details.
 *
 */

void arm_negate_s8(int8_t *vect, uint16_t size) {
#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
  uint16_t simd_loops = size >> 2;
  uint32_t input;
  uint32_t output_packed;

  while (simd_loops > 0) {
    input = *__SIMD32(vect);
    output_packed = __QSUB8(0x00, input);

    write_q7x4_ia(&vect, output_packed);

    simd_loops--;
  }

  uint16_t rem_loops = size & 0x3U;

#else
  // TODO(anybody): add MVEI version

  uint16_t rem_loops = size;

#endif

  while (rem_loops > 0) {
    *vect = 0x00 - *vect;
    vect++;

    rem_loops--;
  }
}

/**
 * @} end of BasicMath group
 */
