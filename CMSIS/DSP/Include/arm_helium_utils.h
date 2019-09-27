/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_helium_utils.h
 * Description:  Utility functions for Helium development
 *
 * $Date:        09. September 2019
 * $Revision:    V.1.5.1
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

#ifndef _ARM_UTILS_HELIUM_H_
#define _ARM_UTILS_HELIUM_H_

/***************************************

Definitions available for MVEF and MVEI

***************************************/
#if defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEF) || defined(ARM_MATH_MVEI)

#define nbLanes(sz)             (128/sz)

#define VEC_LANES_F32       nbLanes(32)
#define VEC_LANES_F16       nbLanes(16)
#define VEC_LANES_Q63       nbLanes(64)
#define VEC_LANES_Q31       nbLanes(32)
#define VEC_LANES_Q15       nbLanes(16)
#define VEC_LANES_Q7        nbLanes(8)

#define nb_vec_lanes(ptr) _Generic((ptr), \
               uint32_t *: VEC_LANES_Q31, \
               uint16_t *: VEC_LANES_Q15, \
                uint8_t *: VEC_LANES_Q7,  \
                  q31_t *: VEC_LANES_Q31, \
                  q15_t *: VEC_LANES_Q15, \
                   q7_t *: VEC_LANES_Q7,  \
               float32_t*: VEC_LANES_F32, \
               float16_t*: VEC_LANES_F16, \
            const q31_t *: VEC_LANES_Q31, \
            const q15_t *: VEC_LANES_Q15, \
             const q7_t *: VEC_LANES_Q7,  \
         const float32_t*: VEC_LANES_F32, \
         const float16_t*: VEC_LANES_F16, \
                  default: "err")



#define post_incr_vec_size(ptr)         ptr += nb_vec_lanes(ptr)

#endif /* defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEF) || defined(ARM_MATH_MVEI) */

/***************************************

Definitions available for MVEF only

***************************************/
#if defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEF)
__STATIC_FORCEINLINE float32_t vecAddAcrossF32Mve(float32x4_t in)
{
    float32_t acc;

    acc = vgetq_lane(in, 0) + vgetq_lane(in, 1) +
          vgetq_lane(in, 2) + vgetq_lane(in, 3);

    return acc;
}
#endif /* defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEF) */

/***************************************

Definitions available for MVEI only

***************************************/
#if defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEI)
#endif /* defined (ARM_MATH_HELIUM) || defined(ARM_MATH_MVEI) */

#endif