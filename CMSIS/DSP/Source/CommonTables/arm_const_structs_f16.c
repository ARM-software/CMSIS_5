/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_const_structs_f16.c
 * Description:  Constant structs that are initialized for user convenience.
 *               For example, some can be given as arguments to the arm_cfft_f32() or arm_rfft_f32() functions.
 *
 * $Date:        27. January 2017
 * $Revision:    V.1.5.1
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2017 ARM Limited or its affiliates. All rights reserved.
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

#include "arm_math_f16.h"

#if defined(ARM_FLOAT16_SUPPORTED)

#include "arm_const_structs_f16.h"


/*
ALLOW TABLE is true when config table is enabled and the Tramsform folder is included 
for compilation.
*/
#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)


/* Floating-point structs */
#if !defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE)


/* 

Those structures cannot be used to initialize the MVE version of the FFT F32 instances.
So they are not compiled when MVE is defined.

For the MVE version, the new arm_cfft_init_f32 must be used.


*/

#if !defined(__CC_ARM)
const arm_cfft_instance_f16 arm_cfft_sR_f16_len16 = {
  16, twiddleCoefF16_16, armBitRevIndexTable_fixed_16, ARMBITREVINDEXTABLE_FIXED_16_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len32 = {
  32, twiddleCoefF16_32, armBitRevIndexTable_fixed_32, ARMBITREVINDEXTABLE_FIXED_32_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len64 = {
  64, twiddleCoefF16_64, armBitRevIndexTable_fixed_64, ARMBITREVINDEXTABLE_FIXED_64_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len128 = {
  128, twiddleCoefF16_128, armBitRevIndexTable_fixed_128, ARMBITREVINDEXTABLE_FIXED_128_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len256 = {
  256, twiddleCoefF16_256, armBitRevIndexTable_fixed_256, ARMBITREVINDEXTABLE_FIXED_256_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len512 = {
  512, twiddleCoefF16_512, armBitRevIndexTable_fixed_512, ARMBITREVINDEXTABLE_FIXED_512_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len1024 = {
  1024, twiddleCoefF16_1024, armBitRevIndexTable_fixed_1024, ARMBITREVINDEXTABLE_FIXED_1024_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len2048 = {
  2048, twiddleCoefF16_2048, armBitRevIndexTable_fixed_2048, ARMBITREVINDEXTABLE_FIXED_2048_TABLE_LENGTH
};

const arm_cfft_instance_f16 arm_cfft_sR_f16_len4096 = {
  4096, twiddleCoefF16_4096, armBitRevIndexTable_fixed_4096, ARMBITREVINDEXTABLE_FIXED_4096_TABLE_LENGTH
};
#endif 

#endif /* !defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE) */


#endif

#endif
