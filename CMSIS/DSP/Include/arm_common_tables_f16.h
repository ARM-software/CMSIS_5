/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_common_tables_f16.h
 * Description:  Extern declaration for common tables
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

#ifndef _ARM_COMMON_TABLES_F16_H
#define _ARM_COMMON_TABLES_F16_H

#include "arm_math_f16.h"
#include "arm_common_tables.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)

  /* F16 */
  #if !defined(__CC_ARM) && defined(ARM_FLOAT16_SUPPORTED)
  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_16)
    extern const float16_t twiddleCoefF16_16[32];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_32)
    extern const float16_t twiddleCoefF16_32[64];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_64)
    extern const float16_t twiddleCoefF16_64[128];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_128)
    extern const float16_t twiddleCoefF16_128[256];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_256)
    extern const float16_t twiddleCoefF16_256[512];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_512)
    extern const float16_t twiddleCoefF16_512[1024];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_1024)
    extern const float16_t twiddleCoefF16_1024[2048];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_2048)
    extern const float16_t twiddleCoefF16_2048[4096];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */

  #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_4096)
    extern const float16_t twiddleCoefF16_4096[8192];
  #endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) */
  #endif /* ARMAC5 */
    
#endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES) */


#ifdef   __cplusplus
}
#endif

#endif /*  _ARM_COMMON_TABLES_F16_H */

  
