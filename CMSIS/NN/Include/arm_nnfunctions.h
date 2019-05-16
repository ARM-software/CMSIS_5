/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_nnfunctions.h
 * Description:  Public header file for CMSIS NN Library
 *
 * $Date:        13. July 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

/**
   \mainpage CMSIS NN Software Library
   *
   * Introduction
   * ------------
   *
   * This user manual describes the CMSIS NN software library,
   * a collection of efficient neural network kernels developed to maximize the
   * performance and minimize the memory footprint of neural networks on Cortex-M processor cores.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - Neural Network Convolution Functions
   * - Neural Network Activation Functions
   * - Fully-connected Layer Functions
   * - Neural Network Pooling Functions
   * - Softmax Functions
   * - Neural Network Support Functions
   *
   * The library has separate functions for operating on different weight and activation data
   * types including 8-bit integers (q7_t) and 16-bit integers (q15_t). The descrition of the
   * kernels are included in the function description. The implementation details are also
   * described in this paper [1].
   *
   * Block Diagram
   * --------
   * \image html CMSIS-NN-OVERVIEW.PNG
   *
   * Examples
   * --------
   *
   * The library ships with a number of examples which demonstrate how to use the library functions.
   *
   * Pre-processor Macros
   * ------------
   *
   * Each library project have differant pre-processor macros.
   *
   * - ARM_MATH_DSP:
   *
   * Define macro ARM_MATH_DSP, If the silicon supports DSP instructions.
   *
   * - ARM_MATH_BIG_ENDIAN:
   *
   * Define macro ARM_MATH_BIG_ENDIAN to build the library for big endian targets. By default library builds for little endian targets.
   *
   * - ARM_NN_TRUNCATE:
   *
   * Define macro ARM_NN_TRUNCATE to use floor instead of round-to-the-nearest-int for the computation.
   *
   * Copyright Notice
   * ------------
   *
   * Copyright (C) 2010-2018 Arm Limited. All rights reserved.
   *
   * [1] CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs https://arxiv.org/abs/1801.06601
   */

/**
 * @defgroup groupNN Neural Network Functions
 * These functions perform basic operations for neural network layers.
 */

#ifndef _ARM_NNFUNCTIONS_H
#define _ARM_NNFUNCTIONS_H

#include "arm_nnsupportfunctions.h"
#include "arm_nn_tables.h"

#define USE_INTRINSIC

//#define ARM_NN_TRUNCATE /* This config the rounding model to floor or round to the nearest int */

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup NNConv Neural Network Convolution Functions
 *
 * Perform convolution layer
 *
 * The convolution is implemented in 2 steps: im2col and GEMM
 *
 * im2col is a process of converting each patch of image data into
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 *
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated and
 * computed with GEMM kernels similar to CMSIS-DSP arm_mat_mult functions.
 *
 */

/**
   * @brief Basic s8 convolution function
   * @param[in]       input      pointer to input tensor. Range: int8, format: [H,W,in_ch]
   * @param[in]       input_x    input tensor width
   * @param[in]       input_y    input tensor height
   * @param[in]       input_ch   number of input tensor channels
   * @param[in]       kernel     pointer to kernel weights. Range: int8, format: [out_ch, H, W, in_ch]
   * @param[in]       output_ch  number of filters, i.e., output tensor channels
   * @param[in]       kernel_x   filter/kernel width
   * @param[in]       kernel_y   filter/kernel height
   * @param[in]       pad_x      padding along width
   * @param[in]       pad_y      padding along height
   * @param[in]       stride_x   convolution stride x
   * @param[in]       stride_y   convolution stride y
   * @param[in]       bias       pointer to per output channel bias. Range: int8
   * @param[in,out]   output     pointer to output tensor. format: [H, W, out_ch]
   * @param[in]       output_shift    pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult     pointer to per output channel requantization multiplier parameter.
   * @param[in]       out_offset      output tensor offset.
   * @param[in]       input_offset    input tensor offset.
   * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_x    output tensor width
   * @param[in]       output_y    output tensor height
   * @param[in]       buffer_a    pointer to buffer space used for input optimization(partial im2col) and is necessary
   *                              when ARM_MATH_LOOPUNROLL and ARM_MATH_DSP is defined.
   *                              Required space: (2 * input_ch * kernel_x * kernel_x) * sizeof(q15_t) bytes
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   * @details
   *    1. Supported framework: TensorFlow Lite micro
   *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
   *    3. Additional memory is required for optimization. Refer to argument 'buffer_a' for details.
   *
   */
    arm_status arm_convolve_s8(const q7_t *input,
                               const uint16_t input_x,
                               const uint16_t input_y,
                               const uint16_t input_ch,
                               const q7_t *kernel,
                               const uint16_t output_ch,
                               const uint16_t kernel_x,
                               const uint16_t kernel_y,
                               const uint16_t pad_x,
                               const uint16_t pad_y,
                               const uint16_t stride_x,
                               const uint16_t stride_y,
                               const q7_t *bias,
                               q7_t *output,
                               const int32_t *output_shift,
                               const int32_t *output_mult,
                               const int32_t out_offset,
                               const int32_t input_offset,
                               const int32_t output_activation_min,
                               const int32_t output_activation_max,
                               const uint16_t output_x,
                               const uint16_t output_y,
                               q15_t *buffer_a);

  /**
   * @brief Basic Q7 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */
    arm_status arm_convolve_HWC_q7_basic(const q7_t * Im_in,
                                         const uint16_t dim_im_in,
                                         const uint16_t ch_im_in,
                                         const q7_t * wt,
                                         const uint16_t ch_im_out,
                                         const uint16_t dim_kernel,
                                         const uint16_t padding,
                                         const uint16_t stride,
                                         const q7_t * bias,
                                         const uint16_t bias_shift,
                                         const uint16_t out_shift,
                                         q7_t * Im_out,
                                         const uint16_t dim_im_out,
                                         q15_t * bufferA,
                                         q7_t * bufferB);

  /**
   * @brief Basic Q7 convolution function (non-square shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   */
    arm_status arm_convolve_HWC_q7_basic_nonsquare(const q7_t * Im_in,
                                                  const uint16_t dim_im_in_x,
                                                  const uint16_t dim_im_in_y,
                                                  const uint16_t ch_im_in,
                                                  const q7_t * wt,
                                                  const uint16_t ch_im_out,
                                                  const uint16_t dim_kernel_x,
                                                  const uint16_t dim_kernel_y,
                                                  const uint16_t padding_x,
                                                  const uint16_t padding_y,
                                                  const uint16_t stride_x,
                                                  const uint16_t stride_y,
                                                  const q7_t * bias,
                                                  const uint16_t bias_shift,
                                                  const uint16_t out_shift,
                                                  q7_t * Im_out,
                                                  const uint16_t dim_im_out_x,
                                                  const uint16_t dim_im_out_y,
                                                  q15_t * bufferA,
                                                  q7_t * bufferB);

  /**
   * @brief Basic Q15 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */
    arm_status arm_convolve_HWC_q15_basic(const q15_t * Im_in,
                                          const uint16_t dim_im_in,
                                          const uint16_t ch_im_in,
                                          const q15_t * wt,
                                          const uint16_t ch_im_out,
                                          const uint16_t dim_kernel,
                                          const uint16_t padding,
                                          const uint16_t stride,
                                          const q15_t * bias,
                                          const uint16_t bias_shift,
                                          const uint16_t out_shift,
                                          q15_t * Im_out,
                                          const uint16_t dim_im_out,
                                          q15_t * bufferA,
                                          q7_t * bufferB);

  /**
   * @brief Fast Q7 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 4
   *   ch_im_out is multiple of 2
   */
    arm_status arm_convolve_HWC_q7_fast(const q7_t * Im_in,
                                        const uint16_t dim_im_in,
                                        const uint16_t ch_im_in,
                                        const q7_t * wt,
                                        const uint16_t ch_im_out,
                                        const uint16_t dim_kernel,
                                        const uint16_t padding,
                                        const uint16_t stride,
                                        const q7_t * bias,
                                        const uint16_t bias_shift,
                                        const uint16_t out_shift,
                                        q7_t * Im_out,
                                        const uint16_t dim_im_out,
                                        q15_t * bufferA,
                                        q7_t * bufferB);

  /**
   * @brief Fast Q7 convolution function (non-sqaure shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 4
   *   ch_im_out is multiple of 2
   */

    arm_status arm_convolve_HWC_q7_fast_nonsquare(const q7_t * Im_in,
                                                  const uint16_t dim_im_in_x,
                                                  const uint16_t dim_im_in_y,
                                                  const uint16_t ch_im_in,
                                                  const q7_t * wt,
                                                  const uint16_t ch_im_out,
                                                  const uint16_t dim_kernel_x,
                                                  const uint16_t dim_kernel_y,
                                                  const uint16_t padding_x,
                                                  const uint16_t padding_y,
                                                  const uint16_t stride_x,
                                                  const uint16_t stride_y,
                                                  const q7_t * bias,
                                                  const uint16_t bias_shift,
                                                  const uint16_t out_shift,
                                                  q7_t * Im_out,
                                                  const uint16_t dim_im_out_x,
                                                  const uint16_t dim_im_out_y,
                                                  q15_t * bufferA,
                                                  q7_t * bufferB);

  /**
   * @brief Fast Q7 version of 1x1 convolution (non-sqaure shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function implement convolution with 1x1 kernel size (i.e., dim_kernel_x=1
   * and dim_kernel_y=1). It can be used for
   * second half of MobileNets after depthwise separable convolution.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 4
   *   ch_im_out is multiple of 2
   */
    arm_status arm_convolve_1x1_HWC_q7_fast_nonsquare(const q7_t * Im_in,
                                                      const uint16_t dim_im_in_x,
                                                      const uint16_t dim_im_in_y,
                                                      const uint16_t ch_im_in,
                                                      const q7_t * wt,
                                                      const uint16_t ch_im_out,
                                                      const uint16_t dim_kernel_x,
                                                      const uint16_t dim_kernel_y,
                                                      const uint16_t padding_x,
                                                      const uint16_t padding_y,
                                                      const uint16_t stride_x,
                                                      const uint16_t stride_y,
                                                      const q7_t * bias,
                                                      const uint16_t bias_shift,
                                                      const uint16_t out_shift,
                                                      q7_t * Im_out,
                                                      const uint16_t dim_im_out_x,
                                                      const uint16_t dim_im_out_y,
                                                      q15_t * bufferA,
                                                      q7_t * bufferB);

  /**
   * @brief Q7 version of convolution for RGB image
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This kernel is written exclusively for convolution with ch_im_in
   * equals 3. This applies on the first layer of CNNs which has input
   * image with RGB format.
   */

    arm_status arm_convolve_HWC_q7_RGB(const q7_t * Im_in,
                                       const uint16_t dim_im_in,
                                       const uint16_t ch_im_in,
                                       const q7_t * wt,
                                       const uint16_t ch_im_out,
                                       const uint16_t dim_kernel,
                                       const uint16_t padding,
                                       const uint16_t stride,
                                       const q7_t * bias,
                                       const uint16_t bias_shift,
                                       const uint16_t out_shift,
                                       q7_t * Im_out,
                                       const uint16_t dim_im_out,
                                       q15_t * bufferA,
                                       q7_t * bufferB);

  /**
   * @brief Fast Q15 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 2
   *   ch_im_out is multiple of 2
   */

    arm_status arm_convolve_HWC_q15_fast(const q15_t * Im_in,
                                         const uint16_t dim_im_in,
                                         const uint16_t ch_im_in,
                                         const q15_t * wt,
                                         const uint16_t ch_im_out,
                                         const uint16_t dim_kernel,
                                         const uint16_t padding,
                                         const uint16_t stride,
                                         const q15_t * bias,
                                         const uint16_t bias_shift,
                                         const uint16_t out_shift,
                                         q15_t * Im_out,
                                         const uint16_t dim_im_out,
                                         q15_t * bufferA,
                                         q7_t * bufferB);

  /**
   * @brief Fast Q15 convolution function (non-sqaure shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
   *
   * bufferB size: 0
   *
   * <b>Input dimension constraints:</b>
   *
   * ch_im_in is multiple of 2
   *
   * ch_im_out is multipe of 2
   *
   */

    arm_status
    arm_convolve_HWC_q15_fast_nonsquare(const q15_t * Im_in,
                              const uint16_t dim_im_in_x,
                              const uint16_t dim_im_in_y,
                              const uint16_t ch_im_in,
                              const q15_t * wt,
                              const uint16_t ch_im_out,
                              const uint16_t dim_kernel_x,
                              const uint16_t dim_kernel_y,
                              const uint16_t padding_x,
                              const uint16_t padding_y,
                              const uint16_t stride_x,
                              const uint16_t stride_y,
                              const q15_t * bias,
                              const uint16_t bias_shift,
                              const uint16_t out_shift,
                              q15_t * Im_out,
                              const uint16_t dim_im_out_x,
                              const uint16_t dim_im_out_y,
                              q15_t * bufferA,
                              q7_t * bufferB);

  /**
   * @brief Q7 depthwise separable convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 2
   *   ch_im_out is multiple of 2
   */

    arm_status arm_depthwise_separable_conv_HWC_q7(const q7_t * Im_in,
                                                   const uint16_t dim_im_in,
                                                   const uint16_t ch_im_in,
                                                   const q7_t * wt,
                                                   const uint16_t ch_im_out,
                                                   const uint16_t dim_kernel,
                                                   const uint16_t padding,
                                                   const uint16_t stride,
                                                   const q7_t * bias,
                                                   const uint16_t bias_shift,
                                                   const uint16_t out_shift,
                                                   q7_t * Im_out,
                                                   const uint16_t dim_im_out,
                                                   q15_t * bufferA,
                                                   q7_t * bufferB);

  /**
   * @brief Q7 depthwise separable convolution function (non-square shape)
   * @param[in]       Im_in         pointer to input tensor
   * @param[in]       dim_im_in_x   input tensor dimension x
   * @param[in]       dim_im_in_y   input tensor dimension y
   * @param[in]       ch_im_in      number of input tensor channels
   * @param[in]       wt            pointer to kernel weights
   * @param[in]       ch_im_out     number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x  filter kernel size x
   * @param[in]       dim_kernel_y  filter kernel size y
   * @param[in]       padding_x     padding sizes x
   * @param[in]       padding_y     padding sizes y
   * @param[in]       stride_x      convolution stride x
   * @param[in]       stride_y      convolution stride y
   * @param[in]       bias          pointer to bias
   * @param[in]       bias_shift    amount of left-shift for bias
   * @param[in]       out_shift     amount of right-shift for output
   * @param[in,out]   Im_out        pointer to output tensor
   * @param[in]       dim_im_out_x  output tensor dimension x
   * @param[in]       dim_im_out_y  output tensor dimension y
   * @param[in,out]   bufferA       pointer to buffer space for input
   * @param[in,out]   bufferB       pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 2
   *   ch_im_out is multiple of 2
   */
    arm_status arm_depthwise_separable_conv_HWC_q7_nonsquare(const q7_t * Im_in,
                                                             const uint16_t dim_im_in_x,
                                                             const uint16_t dim_im_in_y,
                                                             const uint16_t ch_im_in,
                                                             const q7_t * wt,
                                                             const uint16_t ch_im_out,
                                                             const uint16_t dim_kernel_x,
                                                             const uint16_t dim_kernel_y,
                                                             const uint16_t padding_x,
                                                             const uint16_t padding_y,
                                                             const uint16_t stride_x,
                                                             const uint16_t stride_y,
                                                             const q7_t * bias,
                                                             const uint16_t bias_shift,
                                                             const uint16_t out_shift,
                                                             q7_t * Im_out,
                                                             const uint16_t dim_im_out_x,
                                                             const uint16_t dim_im_out_y,
                                                             q15_t * bufferA,
                                                             q7_t * bufferB);

/**
   * @brief Basic s8 depthwise convolution function
   * @param[in]       input      pointer to input tensor. Range: int8, format: [H,W,in_ch]
   * @param[in]       input_x    input tensor width
   * @param[in]       input_y    input tensor height
   * @param[in]       input_ch   number of input tensor channels
   * @param[in]       kernel     pointer to kernel weights. Range: int8, format: [in_ch, H, W, out_ch]
   * @param[in]       output_ch  Number of output channels.
   * @param[in]       ch_mult    channel multiplier. optput_ch = ch_mult * input_ch
   * @param[in]       kernel_x   filter/kernel width
   * @param[in]       kernel_y   filter/kernel height
   * @param[in]       pad_x      padding along width
   * @param[in]       pad_y      padding along height
   * @param[in]       stride_x   convolution stride along width
   * @param[in]       stride_y   convolution stride along height
   * @param[in]       bias       pointer to per output channel bias. Range: int8
   * @param[in,out]   output     pointer to output tensor. Format: [H, W, out_ch]
   * @param[in]       output_shift pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult  pointer to per output channel requantization multiplier parameter.
   * @param[in]       output_x     output tensor width
   * @param[in]       output_y     output tensor height
   * @param[in]       output_offset   offset to elements of output tensor
   * @param[in]       input_offset    offset to elements of input tensor
   * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]       dilation_x   dilation along x. Not used. Dilation factor of 1 is used.
   * @param[in]       dilation_y   dilation along y. Not used. Dilation factor of 1 is used.
   * @param[in]       buffer_a     pointer to buffer space used for optimization. Size information will be updated along
   *                               with the optimized version.
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   * @details
   *    1. Supported framework: TensorFlow Lite
   *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
   *    3. The optimized version to follow can have limitations on the ranges of certain inputs.
   *
   */

    arm_status arm_depthwise_conv_s8(const q7_t *input,
                                     const uint16_t input_x,
                                     const uint16_t input_y,
                                     const uint16_t input_ch,
                                     const q7_t *kernel,
                                     const uint16_t output_ch,
                                     const uint16_t ch_mult,
                                     const uint16_t kernel_x,
                                     const uint16_t kernel_y,
                                     const uint16_t pad_x,
                                     const uint16_t pad_y,
                                     const uint16_t stride_x,
                                     const uint16_t stride_y,
                                     const q7_t *bias,
                                     q7_t *output,
                                     const int32_t *output_shift,
                                     const int32_t *output_mult,
                                     const uint16_t output_x,
                                     const uint16_t output_y,
                                     const int32_t output_offset,
                                     const int32_t input_offset,
                                     const int32_t output_activation_min,
                                     const int32_t output_activation_max,
                                     const uint16_t dilation_x,
                                     const uint16_t dilation_y,
                                     q15_t *buffer_a);


/**
 * @defgroup FC Fully-connected Layer Functions
 *
 * Perform fully-connected layer
 *
 * Fully-connected layer is basically a matrix-vector multiplication
 * with bias. The matrix is the weights and the input/output vectors
 * are the activation values. Supported {weight, activation} precisions
 * include {8-bit, 8-bit}, {16-bit, 16-bit}, and {8-bit, 16-bit}.
 *
 * Here we have two types of kernel functions. The basic function
 * implements the function using regular GEMV approach. The opt functions
 * operates with weights in interleaved formats.
 *
 */

  /**
   * @brief Q7 basic fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q7(const q7_t * pV,
                                      const q7_t * pM,
                                      const uint16_t dim_vec,
                                      const uint16_t num_of_rows,
                                      const uint16_t bias_shift,
                                      const uint16_t out_shift,
                                      const q7_t * bias,
                                      q7_t * pOut,
                                      q15_t * vec_buffer);

  /**
   * @brief S8 basic fully-connected layer function for TF Lite
   * @param[in]       pInput                       pointer to pInput vector
   * @param[in]       pWeight                      pointer to matrix weights
   * @param[in]       col_dim                      dimension of the input vector
   * @param[in]       row_dim                      dimension of the output vector
   * @param[in]       nb_batches                   number of batches
   * @param[in]       input_offset
   * @param[in]       filter_offset
   * @param[in]       out_mult                     requantization parameter
   * @param[in]       out_shift                    requantization parameter
   * @param[in]       output_offset
   * @param[in]       pBias                        pointer to bias
   * @param[out]      pOut                         pointer to output vector
   * @param[in]       output_activation_min        for clamping
   * @param[in]       output_activation_max        for clamping
   * @param[in,out]   vec_buffer                   pointer to buffer space for pInput
   * @return          The function returns         ARM_MATH_SUCCESS
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * vec_buffer size: col_dim of word16.
   *
   * This basic function is designed to work with regular pWeight
   * matrix without interleaving.
   *
   */
  arm_status
  arm_fully_connected_s8(const int8_t   *pInput,
                         const int8_t   *weight,
                         const uint16_t input_length,
                         const uint16_t num_rows,
                         const uint16_t nb_batches,
                         const int32_t  input_offset,
                         const int32_t  filter_offset,
                         const int32_t  out_mult,
                         const int32_t  out_shift,
                         const int32_t  output_offset,
                         const int8_t   *bias,
                         int8_t         *pOut,
                         const int32_t  output_activation_min,
                         const int32_t  output_activation_max,
                         q15_t          *vec_buffer);

  /**
   * @brief Q7 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q7_opt(const q7_t * pV,
                                          const q7_t * pM,
                                          const uint16_t dim_vec,
                                          const uint16_t num_of_rows,
                                          const uint16_t bias_shift,
                                          const uint16_t out_shift,
                                          const q7_t * bias,
                                          q7_t * pOut,
                                          q15_t * vec_buffer);

  /**
   * @brief Q15 basic fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q15(const q15_t * pV,
                                       const q15_t * pM,
                                       const uint16_t dim_vec,
                                       const uint16_t num_of_rows,
                                       const uint16_t bias_shift,
                                       const uint16_t out_shift,
                                       const q15_t * bias,
                                       q15_t * pOut,
                                       q15_t * vec_buffer);

  /**
   * @brief Q15 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q15_opt(const q15_t * pV,
                                           const q15_t * pM,
                                           const uint16_t dim_vec,
                                           const uint16_t num_of_rows,
                                           const uint16_t bias_shift,
                                           const uint16_t out_shift,
                                           const q15_t * bias,
                                           q15_t * pOut,
                                           q15_t * vec_buffer);

  /**
   * @brief Mixed Q15-Q7 fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_mat_q7_vec_q15(const q15_t * pV,
                                                  const q7_t * pM,
                                                  const uint16_t dim_vec,
                                                  const uint16_t num_of_rows,
                                                  const uint16_t bias_shift,
                                                  const uint16_t out_shift,
                                                  const q7_t * bias,
                                                  q15_t * pOut,
                                                  q15_t * vec_buffer);

  /**
   * @brief Mixed Q15-Q7 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_mat_q7_vec_q15_opt(const q15_t * pV,
                                                      const q7_t * pM,
                                                      const uint16_t dim_vec,
                                                      const uint16_t num_of_rows,
                                                      const uint16_t bias_shift,
                                                      const uint16_t out_shift,
                                                      const q7_t * bias,
                                                      q15_t * pOut,
                                                      q15_t * vec_buffer);

/**
 * @brief Matrix-Multiplication Kernels for Convolution
 *
 * These functions are used within convolution layer functions for
 * matrix multiplication.
 *
 * The implementation is similar to CMSIS-DSP arm_mat_mult functions
 * with one Q7 and one Q15 operands. The Q15 operand is the im2col
 * output which is always with 2 columns.
 *
 */

  /**
   * @brief Matrix-multiplication function for convolution
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always conssists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   */

    q7_t     *arm_nn_mat_mult_kernel_q7_q15(const q7_t * pA,
                                            const q15_t * pInBuffer,
                                            const uint16_t ch_im_out,
                                            const uint16_t numCol_A,
                                            const uint16_t bias_shift,
                                            const uint16_t out_shift,
                                            const q7_t * bias,
                                            q7_t * pOut);
/**
   * @brief Matrix-multiplication function for convolution with per-channel requantization.
   * @param[in]       input_a     pointer to operand A
   * @param[in]       input_b     pointer to operand B, always consists of 2 vectors.
   * @param[in]       output_ch   number of rows of A
   * @param[in]       output_shift  pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult   pointer to per output channel requantization multiplier parameter.
   * @param[in]       out_offset      output tensor offset.
   * @param[in]       activation_min   minimum value to clamp the output to. Range : int8
   * @param[in]       activation_max   maximum value to clamp the output to. Range : int8
   * @param[in]       num_col_a   number of columns of A
   * @param[in]       output_bias per output channel bias
   * @param[in,out]   out_0       pointer to output
   * @return     The function returns one of the two
   *              1. The incremented output pointer for a successful operation or
   *              2. NULL if implementation is not available.
   *
   * @details   This function does the matrix multiplication of weight matrix for all output channels
   *            with 2 columns from im2col producing an output of two elements/output_channel.
   */
    q7_t *arm_nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                        const q15_t *input_b,
                                        const uint16_t output_ch,
                                        const int32_t *out_shift,
                                        const int32_t *out_mult,
                                        const int32_t out_offset,
                                        const int16_t activation_min,
                                        const int16_t activation_max,
                                        const uint16_t num_col_a,
                                        const q7_t *const output_bias,
                                        q7_t *out_0);

  /**
   * @brief Matrix-multiplication function for convolution with reordered columns
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always conssists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   */

    q7_t     *arm_nn_mat_mult_kernel_q7_q15_reordered(const q7_t * pA,
                                                      const q15_t * pInBuffer,
                                                      const uint16_t ch_im_out,
                                                      const uint16_t numCol_A,
                                                      const uint16_t bias_shift,
                                                      const uint16_t out_shift,
                                                      const q7_t * bias,
                                                      q7_t * pOut);

#ifdef __cplusplus
}
#endif

/*
 *  Other functions
 *  These layers are typically not timing critical
 *  Basic implementation is supported here
 */

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup Acti Neural Network Activation Functions
 *
 * Perform activation layers, including ReLU (Rectified Linear Unit),
 * sigmoid and tanh
 *
 */

  /**
   * @brief Q7 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   */

    void      arm_relu_q7(q7_t * data, uint16_t size);

  /**
   * @brief Q15 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   */

    void      arm_relu_q15(q15_t * data, uint16_t size);

  /**
   * @brief Q7 neural network activation function using direct table look-up
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
   * @param[in]       type        type of activation functions
   * @return none.
   */

    void      arm_nn_activations_direct_q7(q7_t * data, uint16_t size, uint16_t int_width,
                                           arm_nn_activation_type type);

  /**
   * @brief Q15 neural network activation function using direct table look-up
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
   * @param[in]       type        type of activation functions
   * @return none.
   */

    void      arm_nn_activations_direct_q15(q15_t * data, uint16_t size, uint16_t int_width,
                                            arm_nn_activation_type type);

/**
 * @defgroup Pooling Neural Network Pooling Functions
 *
 * Perform pooling functions, including max pooling and average pooling
 *
 */

  /**
   * @brief Q7 max pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      arm_maxpool_q7_HWC(q7_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 q7_t * bufferA,
                                 q7_t * Im_out);

  /**
   * @brief Q7 average pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      arm_avepool_q7_HWC(q7_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 q7_t * bufferA,
                                 q7_t * Im_out);

/**
 * @defgroup Softmax Softmax Functions
 *
 * EXP(2) based softmax function
 *
 */

  /**
   * @brief Q7 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   */

    void      arm_softmax_q7(const q7_t * vec_in, const uint16_t dim_vec, q7_t * p_out);

  /**
   * @brief Q15 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   */

    void      arm_softmax_q15(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out);

  /**
   * @brief uint8 depthwise convolution function with asymmetric quantization for even number of channel multiplier
   *        and input channels. Unless specified otherwise, arguments are mandatory.
   *
   * @param[in]     input     Pointer to input tensor
   * @param[in]     input_x   Width of input tensor
   * @param[in]     input_y   Height of input tensor
   * @param[in]     input_ch  Channels in input tensor
   * @param[in]     kernel    Pointer to kernel weights
   * @param[in]     kernel_x  Width of kernel
   * @param[in]     kernel_y  Height of kernel
   * @param[in]     ch_mult   Number of channel multiplier
   * @param[in]     pad_x     Padding sizes x
   * @param[in]     pad_y     Padding sizes y
   * @param[in]     stride_x  Convolution stride along the width
   * @param[in]     stride_y  Convolution stride along the height
   * @param[in]     dilation_x Dilation along width. Not used and intended for future enhancement.
   * @param[in]     dilation_y Dilation along height. Not used and intended for future enhancement.
   * @param[in]     bias       Pointer to optional bias values. If no bias is
   *                           availble, NULL is expected
   * @param[in]     input_offset  Input tensor zero offset
   * @param[in]     filter_offset Kernel tensor zero offset
   * @param[in]     output_offset Output tensor zero offset
   * @param[in,out] output        Pointer to output tensor
   * @param[in]     output_x  Width of output tensor
   * @param[in]     output_y  Height of output tensor
   * @param[in]     output_activation_min   Minimum value to clamp the output to. Range : {0, 255}
   * @param[in]     output_activation_max   Minimum value to clamp the output to. Range : {0, 255}
   * @param[in]     out_shift  Amount of right-shift for output
   * @param[in]     out_mult   Output multiplier for requantization
   * @return        The function returns one of the following
   *                <code>ARM_MATH_SIZE_MISMATCH</code> - Not supported dimension of tensors
   *                <code>ARM_MATH_SUCCESS</code> - Successful operation
   *                <code>ARM_MATH_ARGUMENT_ERROR</code> - Implementation not available
   *
   * <b> Input constraints</b>
   * ch_mult  is multiple of 2
   * kernel_x is multiple of 2
   *
   */
    arm_status arm_depthwise_conv_u8_basic_ver1(const uint8_t *input,
                                                const uint16_t input_x,
                                                const uint16_t input_y,
                                                const uint16_t input_ch,
                                                const uint8_t *kernel,
                                                const uint16_t kernel_x,
                                                const uint16_t kernel_y,
                                                const int16_t ch_mult,
                                                const int16_t pad_x,
                                                const int16_t pad_y,
                                                const int16_t stride_x,
                                                const int16_t stride_y,
                                                const int16_t dilation_x,
                                                const int16_t dilation_y,
                                                const int32_t *bias,
                                                const int32_t input_offset,
                                                const int32_t filter_offset,
                                                const int32_t output_offset,
                                                uint8_t *output,
                                                const uint16_t output_x,
                                                const uint16_t output_y,
                                                const int32_t output_activation_min,
                                                const int32_t output_activation_max,
                                                const int32_t out_shift,
                                                const int32_t out_mult);
#ifdef __cplusplus
}
#endif

#endif
