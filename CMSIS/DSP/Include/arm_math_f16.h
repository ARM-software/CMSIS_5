/******************************************************************************
 * @file     arm_math_f16.h
 * @brief    Public header file for f16 function of the CMSIS DSP Library
 * @version  V1.8.1
 * @date     20. April 2020
 ******************************************************************************/
/*
 * Copyright (c) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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

#ifndef _ARM_MATH_F16_H
#define _ARM_MATH_F16_H

#include "arm_math.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#if !defined( __CC_ARM )

/**
 * @brief 16-bit floating-point type definition.
 * This is already defined in arm_mve.h
 *
 * This is not fully supported on ARM AC5.
 */

/*

Check if the type __fp16 is available.
If it is not available, f16 version of the kernels
won't be built.

*/
#if !(__ARM_FEATURE_MVE & 2) && !(__ARM_NEON)
  #if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE)
  typedef __fp16 float16_t;
  #define ARM_FLOAT16_SUPPORTED
  #endif
#else
  #define ARM_FLOAT16_SUPPORTED
#endif

#if defined(ARM_MATH_NEON) || defined(ARM_MATH_MVEF) /* floating point vector*/
  
#if defined(ARM_MATH_MVE_FLOAT16) || defined(ARM_MATH_NEON_FLOAT16)
  /**
   * @brief 16-bit floating-point 128-bit vector data type
   */
  typedef __ALIGNED(2) float16x8_t f16x8_t;

  /**
   * @brief 16-bit floating-point 128-bit vector pair data type
   */
  typedef float16x8x2_t f16x8x2_t;

  /**
   * @brief 16-bit floating-point 128-bit vector quadruplet data type
   */
  typedef float16x8x4_t f16x8x4_t;

  /**
   * @brief 16-bit ubiquitous 128-bit vector data type
   */
  typedef union _any16x8_t
  {
      float16x8_t     f;
      int16x8_t       i;
  } any16x8_t;
#endif

#endif

#if defined(ARM_MATH_NEON)
 

#if defined(ARM_MATH_NEON_FLOAT16)
  /**
   * @brief 16-bit float 64-bit vector data type.
   */
  typedef  __ALIGNED(2) float16x4_t f16x4_t;

  /**
   * @brief 16-bit floating-point 128-bit vector triplet data type
   */
  typedef float16x8x3_t f16x8x3_t;

  /**
   * @brief 16-bit floating-point 64-bit vector pair data type
   */
  typedef float16x4x2_t f16x4x2_t;

  /**
   * @brief 16-bit floating-point 64-bit vector triplet data type
   */
  typedef float16x4x3_t f16x4x3_t;

  /**
   * @brief 16-bit floating-point 64-bit vector quadruplet data type
   */
  typedef float16x4x4_t f16x4x4_t;

  /**
   * @brief 16-bit ubiquitous 64-bit vector data type
   */
  typedef union _any16x4_t
  {
      float16x4_t     f;
      int16x4_t       i;
  } any16x4_t;
#endif 

#endif



#if defined(ARM_FLOAT16_SUPPORTED)
#define F16_MAX   ((float16_t)FLT_MAX)
#define F16_MIN   (-(float16_t)FLT_MAX)

#define F16_ABSMAX   ((float16_t)FLT_MAX)
#define F16_ABSMIN   ((float16_t)0.0)

  /**
   * @brief Floating-point vector addition.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_add_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Floating-point vector subtraction.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_sub_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t blockSize);

    /**
   * @brief Multiplies a floating-point vector by a scalar.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  scale      scale factor to be applied
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_scale_f16(
  const float16_t * pSrc,
        float16_t scale,
        float16_t * pDst,
        uint32_t blockSize);

    /**
   * @brief Floating-point vector absolute value.
   * @param[in]  pSrc       points to the input buffer
   * @param[out] pDst       points to the output buffer
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_abs_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t blockSize);


  /**
   * @brief  Adds a constant offset to a floating-point vector.
   * @param[in]  pSrc       points to the input vector
   * @param[in]  offset     is the offset to be added
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_offset_f16(
  const float16_t * pSrc,
        float16_t offset,
        float16_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Dot product of floating-point vectors.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[in]  blockSize  number of samples in each vector
   * @param[out] result     output result returned here
   */
  void arm_dot_prod_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        uint32_t blockSize,
        float16_t * result);

  /**
   * @brief Floating-point vector multiplication.
   * @param[in]  pSrcA      points to the first input vector
   * @param[in]  pSrcB      points to the second input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in each vector
   */
  void arm_mult_f16(
  const float16_t * pSrcA,
  const float16_t * pSrcB,
        float16_t * pDst,
        uint32_t blockSize);

  /**
   * @brief  Negates the elements of a floating-point vector.
   * @param[in]  pSrc       points to the input vector
   * @param[out] pDst       points to the output vector
   * @param[in]  blockSize  number of samples in the vector
   */
  void arm_negate_f16(
  const float16_t * pSrc,
        float16_t * pDst,
        uint32_t blockSize);

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
          uint8_t ifftFlag;                  /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;            /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const float16_t *pTwiddle;               /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;            /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;         /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;             /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
          float16_t onebyfftLen;             /**< value of 1/fftLen. */
  } arm_cfft_radix2_instance_f16;

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
          uint8_t ifftFlag;                  /**< flag that selects forward (ifftFlag=0) or inverse (ifftFlag=1) transform. */
          uint8_t bitReverseFlag;            /**< flag that enables (bitReverseFlag=1) or disables (bitReverseFlag=0) bit reversal of output. */
    const float16_t *pTwiddle;               /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;            /**< points to the bit reversal table. */
          uint16_t twidCoefModifier;         /**< twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table. */
          uint16_t bitRevFactor;             /**< bit reversal modifier that supports different size FFTs with the same bit reversal table. */
          float16_t onebyfftLen;             /**< value of 1/fftLen. */
  } arm_cfft_radix4_instance_f16;

  /**
   * @brief Instance structure for the floating-point CFFT/CIFFT function.
   */
  typedef struct
  {
          uint16_t fftLen;                   /**< length of the FFT. */
    const float16_t *pTwiddle;         /**< points to the Twiddle factor table. */
    const uint16_t *pBitRevTable;      /**< points to the bit reversal table. */
          uint16_t bitRevLength;             /**< bit reversal table length. */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
   const uint32_t *rearranged_twiddle_tab_stride1_arr;        /**< Per stage reordered twiddle pointer (offset 1) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride2_arr;        /**< Per stage reordered twiddle pointer (offset 2) */                                                       \
   const uint32_t *rearranged_twiddle_tab_stride3_arr;        /**< Per stage reordered twiddle pointer (offset 3) */                                                       \
   const float16_t *rearranged_twiddle_stride1; /**< reordered twiddle offset 1 storage */                                                                   \
   const float16_t *rearranged_twiddle_stride2; /**< reordered twiddle offset 2 storage */                                                                   \
   const float16_t *rearranged_twiddle_stride3;
#endif
  } arm_cfft_instance_f16;


  arm_status arm_cfft_init_f16(
  arm_cfft_instance_f16 * S,
  uint16_t fftLen);

  void arm_cfft_f16(
  const arm_cfft_instance_f16 * S,
        float16_t * p1,
        uint8_t ifftFlag,
        uint8_t bitReverseFlag);
  
#endif /* ARM_FLOAT16_SUPPORTED*/
#endif /* !defined( __CC_ARM ) */

#ifdef   __cplusplus
}
#endif

#endif /* _ARM_MATH_F16_H */


