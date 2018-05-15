/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_cfft_init_f32.c
 * Description:  Split Radix Decimation in Frequency CFFT Floating point processing function
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

#include "arm_math.h"
#include "arm_common_tables.h"

/**
 * @ingroup groupTransforms
 */

/**
 * @addtogroup RealFFT
 * @{
 */


/**
* @brief  Initialization function for the 32pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_32_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 16U;
  S->fftLenRFFT = 32U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_16_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable16;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_16;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_32;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 64pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_64_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 32U;
  S->fftLenRFFT = 64U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_32_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable32;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_32;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_64;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 128pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_128_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 64U;
  S->fftLenRFFT = 128U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_64_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable64;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_64;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_128;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 256pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_256_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 128U;
  S->fftLenRFFT = 256U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_128_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable128;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_128;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_256;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 512pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_512_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 256U;
  S->fftLenRFFT = 512U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_256_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable256;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_256;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_512;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 1024pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_1024_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 512U;
  S->fftLenRFFT = 1024U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_512_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable512;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_512;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_1024;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 2048pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_2048_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 1024U;
  S->fftLenRFFT = 2048U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_1024_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable1024;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_1024;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_2048;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the 4096pt floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if an error is detected.
*/
arm_status arm_rfft_4096_fast_init_f32( arm_rfft_fast_instance_f32 * S ) {

  arm_cfft_instance_f32 * Sint;

  if( !S ) return ARM_MATH_ARGUMENT_ERROR;

  Sint = &(S->Sint);
  Sint->fftLen = 2048U;
  S->fftLenRFFT = 4096U;

  Sint->bitRevLength = ARMBITREVINDEXTABLE_2048_TABLE_LENGTH;
  Sint->pBitRevTable = (uint16_t *)armBitRevIndexTable2048;
  Sint->pTwiddle     = (float32_t *) twiddleCoef_2048;
  S->pTwiddleRFFT    = (float32_t *) twiddleCoef_rfft_4096;

  return ARM_MATH_SUCCESS;

}

/**
* @brief  Initialization function for the floating-point real FFT.
* @param[in,out] *S             points to an arm_rfft_fast_instance_f32 structure.
* @param[in]     fftLen         length of the Real Sequence.
* @return        The function returns ARM_MATH_SUCCESS if initialization is successful or ARM_MATH_ARGUMENT_ERROR if <code>fftLen</code> is not a supported value.
*
* \par Description:
* \par
* The parameter <code>fftLen</code>	Specifies length of RFFT/CIFFT process. Supported FFT Lengths are 32, 64, 128, 256, 512, 1024, 2048, 4096.
* \par
* This Function also initializes Twiddle factor table pointer and Bit reversal table pointer.
*/
arm_status arm_rfft_fast_init_f32(
  arm_rfft_fast_instance_f32 * S,
  uint16_t fftLen)
{
  typedef arm_status(*fft_init_ptr)( arm_rfft_fast_instance_f32 *);
  fft_init_ptr fptr = 0x0;

  switch (fftLen)
  {
  case 4096U:
    fptr = arm_rfft_4096_fast_init_f32;
    break;
  case 2048U:
    fptr = arm_rfft_2048_fast_init_f32;
    break;
  case 1024U:
    fptr = arm_rfft_1024_fast_init_f32;
    break;
  case 512U:
    fptr = arm_rfft_512_fast_init_f32;
    break;
  case 256U:
    fptr = arm_rfft_256_fast_init_f32;
    break;
  case 128U:
    fptr = arm_rfft_128_fast_init_f32;
    break;
  case 64U:
    fptr = arm_rfft_64_fast_init_f32;
    break;
  case 32U:
    fptr = arm_rfft_32_fast_init_f32;
    break;
  }

  if( ! fptr ) return ARM_MATH_ARGUMENT_ERROR;
  return fptr( S );

}

/**
 * @} end of RealFFT group
 */
