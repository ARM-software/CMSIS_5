/*
 * Copyright (C) 2010-2019 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_fully_connected_s8
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        10. July 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M and Cortex-A cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup FC
 * @{
 */

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
                       const int8_t   *pWeight,           
                       const uint16_t col_dim,   
                       const uint16_t row_dim,    
                       const uint16_t nb_batches,      
                       const int32_t  input_offset,    
                       const int32_t  filter_offset,   
                       const int32_t  out_mult,       
                       const int32_t  out_shift,     
                       const int32_t  output_offset,     
                       const int8_t   *pBias,             
                       int8_t         *pOut,                   
                       const int32_t  output_activation_min,
                       const int32_t  output_activation_max,
                       q15_t          *vec_buffer)            
{
#if defined(ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP)
    q31_t acc;
    
    uint16_t batchCnt = nb_batches;

    /* CMSIS-DSP and NN are generally using q7 and q15 types.
       Here we are computing with s8 and not q7.
       So, q7_t is not really the right type to use but 
       it is kept for consistency with some function APIs
       which are used in this implementation.

    */
    const int8_t   *pBiasTmp = pBias;
    const q7_t *pB = pWeight;
    const q7_t *pB2;
    q7_t       *pO = pOut;
    q15_t      *pA;
    q31_t      ioffset;
    q31_t      foffset;

    ioffset = ((input_offset & 0x0FFFF) << 16) | (input_offset & 0x0FFFF);
    foffset = ((filter_offset & 0x0FFFF) << 16) | (filter_offset & 0x0FFFF);

while(batchCnt)
{
    pBiasTmp = pBias;
    pB = pWeight;
    arm_q7_to_q15_reordered_no_shift(pInput, vec_buffer, col_dim);
    uint16_t  rowCnt = row_dim >> 1;
    /* Unroll on the rows */
    while (rowCnt)
    {
        q31_t     sum =  (q31_t)(*pBiasTmp++);
        q31_t     sum2 = (q31_t)(*pBiasTmp++);
        uint16_t  colCnt = col_dim >> 2;

        pA = vec_buffer;
        pB2 = pB + col_dim;

        /* Vectorize on the columns */
        while (colCnt)
        {
            q31_t     inV, inM11, inM12, inM21, inM22;
            pB =  read_and_pad_reordered_with_offset((q7_t *)pB, &inM11, &inM12,foffset);
            pB2 = read_and_pad_reordered_with_offset((q7_t *)pB2, &inM21, &inM22,foffset);

            inV = read_q15x2_ia(&pA);
            inV = __QADD16(inV,ioffset);

            sum =  __SMLAD(inV, inM11, sum);
            sum2 = __SMLAD(inV, inM21, sum2);

            inV = read_q15x2_ia(&pA);
            inV = __QADD16(inV,ioffset);

            sum =  __SMLAD(inV, inM12, sum);
            sum2 = __SMLAD(inV, inM22, sum2);

            colCnt--;
        }

        /* Column vector tail */
        colCnt = col_dim & 0x3;
        while (colCnt)
        {
            q15_t    inV = *pA++;
            q7_t     inM = *pB++;
            q7_t     inM2 = *pB2++;

            sum += (inV + input_offset) * (inM + filter_offset);
            sum2 += (inV + input_offset)  * (inM2 + filter_offset);
            colCnt--;
        }                      

       acc = arm_nn_sat_doubling_high_mult(sum * (1 << LEFT_SHIFT(out_shift)), out_mult);
       acc = arm_nn_divide_by_power_of_two(acc,RIGHT_SHIFT(out_shift));
       acc += output_offset;
       acc = MAX(acc, output_activation_min);
       acc = MIN(acc, output_activation_max);

       *pO++ = (q7_t) (acc);

       acc = arm_nn_sat_doubling_high_mult(sum2 * (1 << LEFT_SHIFT(out_shift)), out_mult);
       acc = arm_nn_divide_by_power_of_two(acc,RIGHT_SHIFT(out_shift));
       acc += output_offset;
       acc = MAX(acc, output_activation_min);
       acc = MIN(acc, output_activation_max);
       *pO++ = (q7_t) (acc);

        
        pB += col_dim;
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = row_dim & 0x1;

    while (rowCnt)
    {
        uint16_t  colCnt = col_dim >> 2;
        q31_t     sum = (q31_t)(*pBiasTmp++);

        pA = vec_buffer;

        /* Vectorize on the columns */
        while (colCnt)
        {
            q31_t  inV, inM11, inM12;

            pB = read_and_pad_reordered_with_offset((q7_t *)pB, &inM11, &inM12, foffset);

            inV = read_q15x2_ia(&pA);
            inV = __QADD16(inV,ioffset);

            sum = __SMLAD(inV, inM11, sum);

            inV = read_q15x2_ia(&pA);
            inV = __QADD16(inV,ioffset);

            sum = __SMLAD(inV, inM12, sum);

            colCnt--;
        }

        /* Column vector tail */
        colCnt = col_dim & 0x3;

        while (colCnt)
        {
            q15_t    inV = *pA++;
            q7_t     inM = *pB++;
            sum += (inV + input_offset) * (inM + filter_offset);
            colCnt--;
        }

        acc = arm_nn_sat_doubling_high_mult(sum * (1 << LEFT_SHIFT(out_shift)), out_mult);
        acc = arm_nn_divide_by_power_of_two(acc,RIGHT_SHIFT(out_shift));
        acc += output_offset;
        acc = MAX(acc, output_activation_min);
        acc = MIN(acc, output_activation_max);
        *pO++ = (q7_t) (acc);

        rowCnt--;
    }
    pInput += col_dim;
    batchCnt--;
  }
    return(ARM_MATH_SUCCESS);

#else
    const int8_t *pInputA;
    const int8_t   *pBiasTmp = pBias;
    const int8_t   *pWeightTmp = pWeight;
     uint16_t batchCnt = nb_batches;

while(batchCnt)
{
    pBiasTmp = pBias;
    pWeightTmp = pWeight;
    for (int out_c = 0; out_c < row_dim; out_c++) 
    {

      int32_t acc = *pBiasTmp++;
      pInputA = pInput;
      for (int d = 0; d < col_dim; d++) 
      {

        int32_t input_val = *pInputA++;

        int32_t filter_val = *pWeightTmp++;

        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }

      
      acc = arm_nn_sat_doubling_high_mult(acc * (1 << LEFT_SHIFT(out_shift)), out_mult);
      acc = arm_nn_divide_by_power_of_two(acc,RIGHT_SHIFT(out_shift));

      acc += output_offset;

      acc = MAX(acc, output_activation_min);
      acc = MIN(acc, output_activation_max);


      *pOut++ = (uint8_t)(acc);

    }
    pInput += col_dim;
    batchCnt--;
}
  return(ARM_MATH_SUCCESS);
#endif  /*  defined(ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP) */
}
/**
 * @} end of FC group
 */
