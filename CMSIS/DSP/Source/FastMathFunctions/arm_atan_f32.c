/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_atan_f32.c
 * Description:  Fast arctan calculation for floating-point values
 *
 * $Date:        18. March 2021
 * $Revision:    V1.6.0
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

#include "dsp/fast_math_functions.h"
#include "arm_common_tables.h"

/**
  @ingroup groupFastMath
 */

/**
  @defgroup atan Arctan

  Computes the trigonometric inverse tangent function using a combination of table lookup
  and linear interpolation. Currently, there is only function for floating-point data type.
  The input to the floating-point version can be any real number.

  The implementation takes advantage of the following identities:
   -# atan(-x) = -atan(x)
   -# atan(x) = pi/2 - atan(1/x)

  Thus, the algorithm needs to compute values in [0,1].

  The implementation is based on table lookup using 512 values together with linear interpolation.
  The steps used are:
   -# Calculation of the nearest integer table index
   -# Compute the fractional portion (fract) of the table index.
   -# The final result equals <code>(1.0f-fract)*a + fract*b;</code>

  where
  <pre>
     b = Table[index];
     c = Table[index+1];
  </pre>
 */

/**
  @addtogroup atan
  @{
 */

/**
  @brief         Fast approximation to the trigonometric arctan function for floating-point data.
  @param[in]     x  input value in radians.
  @return        atan(x)
 */

float32_t arm_atan_f32(
  float32_t x)
{
  float32_t atanVal, fract, in;                   /* Temporary input, output variables */
  uint16_t index;                                /* Index variable */
  float32_t a, b;                                /* Two nearest output values */
  int32_t n;
  float32_t findex;
  int32_t is_negative = 0, is_bigger_one = 0;

  in = x;

  if (in < 0)
  {
    in = -in;
    is_negative = 1;
  }

  if (in > 1)
  {
    in = 1.0f / in;
    is_bigger_one = 1;
  }

  /* Calculation of index of the table */
  findex = (float32_t)FAST_MATH_TABLE_SIZE * in;
  index = (uint16_t)findex;

  /* fractional value calculation */
  fract = findex - (float32_t) index;

  /* Read two nearest values of input value from the atan table */
  a = atanTable_f32[index];
  b = atanTable_f32[index+1];

  /* Linear interpolation process */
  atanVal = (1.0f - fract) * a + fract * b;

  /* Return output value */
  if (is_bigger_one)
  {
    atanVal = 1.570796326794895f - atanVal;
  }

  if (is_negative)
  {
    atanVal = -atanVal;
  }
  
  return (atanVal);
}

/**
  @} end of atan group
 */
