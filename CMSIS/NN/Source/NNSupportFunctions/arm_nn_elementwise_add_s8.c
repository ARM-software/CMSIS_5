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
 * Title:        arm_nn_elementwise_add_s8
 * Description:  Element wise add
 *
 * $Date:        7. August 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M and Cortex-A cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 * @ingroup groupSupport
 */

  /**
   * @brief S8 element wise add
   * @param[in]       pInput1                      pointer to pInput1 vector
   * @param[in]       pInput2                      pointer to pInput2 vector
   * @param[in]       input1_offset                input1 offset
   * @param[in]       input1_mult                  input1 multiplier
   * @param[in]       input1_shift                 input1 shift
   * @param[in]       input2_offset                input2 offset
   * @param[in]       input2_mult                  input2 multiplier
   * @param[in]       input2_shift                 input2 shift
   * @param[in]       left_shift                    
   * @param[out]      pOut                         pointer to output vector
   * @param[in]       output_offset                output offset
   * @param[in]       output_mult                  output multiplier
   * @param[in]       output_shift                 output shift
   * @param[in]       output_activation_min        for clamping
   * @param[in]       output_activation_max        for clamping
   * @param[in]       blockSize                    number of samples
   * @return          The function returns         ARM_MATH_SUCCESS
   *
   * @details
   *
   *
   */

#ifndef ARM_MATH_BIG_ENDIAN
#define READQ7TOQ15(PIN,IN,IN1,IN2,OUT1,OUT2)                            \
        IN = arm_nn_read_q7x4_ia(&PIN);                                  \
                                                                         \
        /* rotatate in by 8 and extend two q7_t values to q15_t values */\
        IN1 = __SXTB16(__ROR(IN, 8));                                    \
                                                                         \
        /* extend remaining two q7_t values to q15_t values */           \
        IN2 = __SXTB16(IN);                                              \
                                                                         \
        OUT2 = __PKHTB(IN1, IN2, 16);                                    \
        OUT1 = __PKHBT(IN2, IN1, 16);

#else
#define READQ7TOQ15(PIN,IN,IN1,IN2,OUT1,OUT2)                            \
        IN = arm_nn_read_q7x4_ia(&PIN);                                  \
                                                                         \
        /* rotatate in by 8 and extend two q7_t values to q15_t values */\
        IN1 = __SXTB16(__ROR(IN, 8));                                    \
                                                                         \
        /* extend remaining two q7_t values to q15_t values */           \
        IN2 = __SXTB16(IN);                                              \
                                                                         \
        OUT1 = __PKHTB(IN1, IN2, 16);                                    \
        OUT2 = __PKHBT(IN2, IN1, 16);
#endif 

#define SATOUTPUT(SUM,MULT,SHIFT,OFFSET)                \
        SUM = arm_nn_sat_doubling_high_mult(SUM,MULT);  \
        SUM = arm_nn_divide_by_power_of_two(SUM,-SHIFT);\
        SUM += OFFSET;                                  \
                                                        \
        SUM = MAX(SUM, output_activation_min);          \
        SUM = MIN(SUM, output_activation_max);

#define SATINPUT(TMP,MULT,SHIFT)                     \
     TMP = arm_nn_sat_doubling_high_mult(TMP,MULT);  \
     TMP = arm_nn_divide_by_power_of_two(TMP,-SHIFT);


arm_status
arm_nn_elementwise_add_s8(const int8_t   *pInput1,             
                          const int8_t   *pInput2,           
                          const int32_t  input1_offset,   
                          const int32_t  input1_mult, 
                          const int32_t  input1_shift, 
                          const int32_t  input2_offset,
                          const int32_t  input2_mult, 
                          const int32_t  input2_shift,
                          const int32_t  left_shift,
                          int8_t         *pOutput,
                          const int32_t  output_offset,             
                          const int32_t  out_mult,       
                          const int32_t  out_shift,     
                          const int32_t  output_activation_min,
                          const int32_t  output_activation_max,
                          const uint32_t blockSize
                          )            
{

  uint32_t blkCnt;                               /* Loop counter */
  int16_t tmp1,tmp2;
  int16_t sum;


#if defined(ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP)

  int32_t inV,in1V,in2V,out1AV,out1BV,out2AV,out2BV;

  int32_t offset1V,offset2V;

  int8_t r1,r2,r3,r4;

  /* 

  It is the reason why the type of input offsets is still int32_t although the
  dynamic of the value is int_8.

  With int32_t we know we have the sign extension on 16 bits so we can just do a mask.

  */

  offset1V = (input1_offset << 16) | (input1_offset & 0x0FFFF);
  offset2V = (input2_offset << 16) | (input2_offset & 0x0FFFF);

      blkCnt = blockSize >> 2;

      while (blkCnt > 0U)
      {

        READQ7TOQ15(pInput1,inV,in1V,in2V,out1AV,out1BV);
        READQ7TOQ15(pInput1,inV,in1V,in2V,out2AV,out2BV);

        /*

        tmp1 = (*pInput1++ + input1_offset) << left_shift;
        tmp2 = (*pInput2++ + input2_offset) << left_shift;

        */
        out1AV = __SADD16(out1AV,offset1V);
        out1BV = __SADD16(out1BV,offset1V);

        out2AV = __SADD16(out2AV,offset2V);
        out2BV = __SADD16(out2BV,offset2V);

        /* SUM 1 */
        tmp1 = (out1AV & 0x0FFFF) << left_shift;
        SATINPUT(tmp1,input1_mult,input1_shift);
 
        tmp2 = (out2AV & 0x0FFFF) << left_shift;
        SATINPUT(tmp2,input2_mult,input2_shift);
     
        sum = tmp1 + tmp2;

        SATOUTPUT(sum,out_mult,out_shift,output_offset);
        r1 = (q7_t)sum;

        /* SUM 2 */
        tmp1 = ((out1AV >> 16) & 0x0FFFF) << left_shift;
        SATINPUT(tmp1,input1_mult,input1_shift);
       
        tmp2 = ((out2AV >> 16) & 0x0FFFF) << left_shift;
        SATINPUT(tmp2,input2_mult,input2_shift);
     
        sum = tmp1 + tmp2;

        SATOUTPUT(sum,out_mult,out_shift,output_offset);
        r2 = (q7_t)sum;

        /* SUM 3 */
        tmp1 = (out1BV & 0x0FFFF) << left_shift;
        SATINPUT(tmp1,input1_mult,input1_shift);
 
        tmp2 = (out2BV & 0x0FFFF) << left_shift;
        SATINPUT(tmp2,input2_mult,input2_shift);
     
        sum = tmp1 + tmp2;

        SATOUTPUT(sum,out_mult,out_shift,output_offset);
        r3 = (q7_t)sum;

        /* SUM 4 */
        tmp1 = ((out1BV >> 16) & 0x0FFFF) << left_shift;
        SATINPUT(tmp1,input1_mult,input1_shift);
 
        tmp2 = ((out2BV >> 16) & 0x0FFFF) << left_shift;
        SATINPUT(tmp2,input2_mult,input2_shift);
     
        sum = tmp1 + tmp2;

        SATOUTPUT(sum,out_mult,out_shift,output_offset);
        r4 = (q7_t)sum;

        write_q7x4_ia (&pOutput, __PACKq7(r1,r2,r3,r4));

        /* Decrement loop counter */
        blkCnt--;
      }

      blkCnt = blockSize & 0x3;
#else
      blkCnt = blockSize;
#endif

  while (blkCnt > 0U)
  {
    /* C = A + B */

    tmp1 = (*pInput1++ + input1_offset) << left_shift;
    tmp2 = (*pInput2++ + input2_offset) << left_shift;

    tmp1 = arm_nn_sat_doubling_high_mult(tmp1,input1_mult);
    tmp1 = arm_nn_divide_by_power_of_two(tmp1,-input1_shift);

    tmp2 = arm_nn_sat_doubling_high_mult(tmp2,input2_mult);
    tmp2 = arm_nn_divide_by_power_of_two(tmp2,-input2_shift);
    
    sum = tmp1 + tmp2;

    sum = arm_nn_sat_doubling_high_mult(sum,out_mult);
    sum = arm_nn_divide_by_power_of_two(sum,-out_shift);
    sum += output_offset;

    sum = MAX(sum, output_activation_min);
    sum = MIN(sum, output_activation_max);

    *pOutput++ = (q7_t) sum;

    /* Decrement loop counter */
    blkCnt--;
  }

  return(ARM_MATH_SUCCESS);
}
/**
 * @} end of FC group
 */
