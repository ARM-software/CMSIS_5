/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_quaternion_product_f32.c
 * Description:  Floating-point quaternion product
 *
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
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

#include "dsp/quaternion_math_functions.h"
#include <math.h>

/**
  @ingroup groupQuaternionMath
 */

/**
  @defgroup QuatProd Quaternion Product

  Compute the product of quaternions.
 */

/**
  @addtogroup QuatProd
  @{
 */

/**
  @brief         Floating-point elementwise product two quaternions.
  @param[in]     qa                  first array of quaternions
  @param[in]     qb                  second array of quaternions
  @param[out]    r                   elementwise product of quaternions
  @param[in]     nbQuaternions       number of quaternions in the array
  @return        none
 */
void arm_quaternion_product_f32(const float32_t *qa, 
    const float32_t *qb, 
    float32_t *r,
    uint32_t nbQuaternions)
{
   for(uint32_t i=0; i < nbQuaternions; i++)
   {
     arm_quaternion_product_single_f32(qa, qb, r);

     qa += 4;
     qb += 4;
     r += 4;
   }
}

/**
  @} end of QuatProd group
 */
