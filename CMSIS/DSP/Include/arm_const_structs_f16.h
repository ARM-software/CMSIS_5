/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_const_structs_f16.h
 * Description:  Constant structs that are initialized for user convenience.
 *               For example, some can be given as arguments to the arm_cfft_f16() function.
 *
 * $Date:        20. April 2020
 * $Revision:    V.1.5.1
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2020 ARM Limited or its affiliates. All rights reserved.
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

#ifndef _ARM_CONST_STRUCTS_F16_H
#define _ARM_CONST_STRUCTS_F16_H

#include "arm_math_f16.h"
#include "arm_common_tables_f16.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if !defined(__CC_ARM) && defined(ARM_FLOAT16_SUPPORTED)
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len16;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len32;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len64;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len128;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len256;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len512;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len1024;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len2048;
   extern const arm_cfft_instance_f16 arm_cfft_sR_f16_len4096;
#endif

#ifdef   __cplusplus
}
#endif

#endif