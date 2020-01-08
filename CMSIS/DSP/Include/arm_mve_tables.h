/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mve_tables.h
 * Description:  common tables like fft twiddle factors, Bitreverse, reciprocal etc
 *               used for MVE implementation only
 *
 * $Date:        08. January 2020
 * $Revision:    V1.7.0
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

 #ifndef _ARM_MVE_TABLES_H
 #define _ARM_MVE_TABLES_H

 #include "arm_math.h"

 #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)

 #if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)


 
#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F32_16) || defined(ARM_TABLE_TWIDDLECOEF_F32_32)

extern uint32_t rearranged_twiddle_tab_stride1_arr_16[2];
extern uint32_t rearranged_twiddle_tab_stride2_arr_16[2];
extern uint32_t rearranged_twiddle_tab_stride3_arr_16[2];
extern float32_t rearranged_twiddle_stride1_16[8];
extern float32_t rearranged_twiddle_stride2_16[8];
extern float32_t rearranged_twiddle_stride3_16[8];
#endif

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F32_64) || defined(ARM_TABLE_TWIDDLECOEF_F32_128)

extern uint32_t rearranged_twiddle_tab_stride1_arr_64[3];
extern uint32_t rearranged_twiddle_tab_stride2_arr_64[3];
extern uint32_t rearranged_twiddle_tab_stride3_arr_64[3];
extern float32_t rearranged_twiddle_stride1_64[40];
extern float32_t rearranged_twiddle_stride2_64[40];
extern float32_t rearranged_twiddle_stride3_64[40];
#endif

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F32_256) || defined(ARM_TABLE_TWIDDLECOEF_F32_512)

extern uint32_t rearranged_twiddle_tab_stride1_arr_256[4];
extern uint32_t rearranged_twiddle_tab_stride2_arr_256[4];
extern uint32_t rearranged_twiddle_tab_stride3_arr_256[4];
extern float32_t rearranged_twiddle_stride1_256[168];
extern float32_t rearranged_twiddle_stride2_256[168];
extern float32_t rearranged_twiddle_stride3_256[168];
#endif

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F32_1024) || defined(ARM_TABLE_TWIDDLECOEF_F32_2048)

extern uint32_t rearranged_twiddle_tab_stride1_arr_1024[5];
extern uint32_t rearranged_twiddle_tab_stride2_arr_1024[5];
extern uint32_t rearranged_twiddle_tab_stride3_arr_1024[5];
extern float32_t rearranged_twiddle_stride1_1024[680];
extern float32_t rearranged_twiddle_stride2_1024[680];
extern float32_t rearranged_twiddle_stride3_1024[680];
#endif

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F32_4096) || defined(ARM_TABLE_TWIDDLECOEF_F32_8192)

extern uint32_t rearranged_twiddle_tab_stride1_arr_4096[6];
extern uint32_t rearranged_twiddle_tab_stride2_arr_4096[6];
extern uint32_t rearranged_twiddle_tab_stride3_arr_4096[6];
extern float32_t rearranged_twiddle_stride1_4096[2728];
extern float32_t rearranged_twiddle_stride2_4096[2728];
extern float32_t rearranged_twiddle_stride3_4096[2728];
#endif


#endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES) */

#endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */

#endif /*_ARM_MVE_TABLES_H*/

