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

/* ----------------------------------------------------------------------
 * Project:      CMSIS-NN
 * Title:        arm_tanh.c
 * Description:  Tanh function implementations with Q7 and Q15
 *
 * Target Processor: Cortex-M4 and Cortex-M7 cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_common_tables.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */

  /**
   * @brief Q7 tanh function using direct table look-up
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
   * @return none.
   *
   * @details
   * 
   * This is the direct table look-up approach.
   *
   * Assume here the integer part of the fixed-point is <= 3.
   * More than 3 just not making much sense, makes no difference with
   * saturation followed by any of these activation functions. 
   */

void arm_tanh_direct_q7(
        q7_t * data,         
        uint16_t size,       
        uint16_t int_width   
) {
  uint16_t i = size >> 2;
  q7_t* pIn = data;
  q7_t* pOut = data;
  union arm_nnword in;
  union arm_nnword out;
  uint16_t shift_size = 3 - int_width;
  while (i) {
    in.word = *__SIMD32(pIn)++;

    out.bytes[0] = tanhTable_q7[(uint8_t)(in.bytes[0]>>shift_size)];
    out.bytes[1] = tanhTable_q7[(uint8_t)(in.bytes[1]>>shift_size)];
    out.bytes[2] = tanhTable_q7[(uint8_t)(in.bytes[2]>>shift_size)];
    out.bytes[3] = tanhTable_q7[(uint8_t)(in.bytes[3]>>shift_size)];

    *__SIMD32(pOut)++ = out.word;
    i--;
  }

  i = size & 0x3;
  while (i) {
    q7_t buf = *pIn ++;
    *pOut ++ = tanhTable_q7[(uint8_t)buf];
    i--;
  }

}

  /**
   * @brief Q15 tanh function using direct table look-up
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
   * @return none.
   *
   * @details
   * 
   * This is the direct table look-up approach.
   *
   * Assume here the integer part of the fixed-point is <= 3.
   * More than 3 just not making much sense, makes no difference with
   * saturation followed by any of these activation functions. 
   */

void arm_tanh_direct_q15(
        q15_t * data,        
        uint16_t size,       
        uint16_t int_width   
) {
  uint16_t i = size;
  q15_t* pIn = data;
  q15_t* pOut = data;
  uint16_t shift_size = 8 + 3 - int_width;
  uint32_t bit_mask = 0x7FF >> int_width;
  uint32_t full_frac = bit_mask + 1;
  while (i) { 
    q15_t in = *pIn++;
    q15_t out;

    q15_t frac = (uint32_t)in & bit_mask; 

    q15_t value  = tanhTable_q15[(uint8_t)__SSAT(in>>shift_size, 8)];
    q15_t value2  = tanhTable_q15[(uint8_t)__SSAT(1+(in>>shift_size), 8)];

    out = ((q31_t)(full_frac - frac)*value + (q31_t)value2 * frac) >> shift_size;

    *pOut++ = out;
    i--;
  }

}


/**
 * @} end of Acti group
 */

