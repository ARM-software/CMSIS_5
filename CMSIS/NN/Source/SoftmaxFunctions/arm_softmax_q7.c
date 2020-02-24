/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_softmax_q7.c
 * Description:  Q7 softmax function
 *
 * $Date:        February 27, 2020
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Softmax
 * @{
 */

  /**
   * @brief Q7 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimention
   * @param[out]      p_out       pointer to output vector
   *
   * @details
   *
   *  Here, instead of typical natural logarithm e based softmax, we use
   *  2-based softmax here, i.e.,:
   *
   *  y_i = 2^(x_i) / sum(2^x_j)
   *
   *  The relative output will be different here.
   *  But mathematically, the gradient will be the same
   *  with a log(2) scaling factor.
   *
   *  If we compare the position of the max value in output of this
   *  function with a reference float32 softmax (and thus using exp)
   *  we see that the position of the max value is sometimes different.
   *
   *  If we do statistics on lot of input vectors we can compute
   *  an average error rate in percent. It is the percent of time
   *  that the max will be at a position different from the one
   *  computed with a reference float32 implementation.
   *
   *  This average error rate is dependent on the vector size.
   *  We have:
   *
   *  Average error rate in percent = -0.555548 + 0.246918 dim_vec
   *  Variance of the error rate = -0.0112281 + 0.0382476 dim_vec
   *
   *
   */

#define Q7BITS 8
#define LOG2Q7BITS 3

void arm_softmax_q7(const q7_t * vec_in, const uint16_t dim_vec, q7_t * p_out )
{
#if defined (ARM_MATH_DSP)
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q15_t     base;
    uint16_t blkCnt;

    q31_t in,in1,in2;
    q31_t out1, out2;

    q31_t baseV;
    q31_t shiftV;
    const q31_t pad=0x0d0d0d0d;
    const q7_t *pIn=vec_in;

    base = -128;


    /* We first search for the maximum */

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }


    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - Q7BITS;
    baseV = ((base & 0x0FF) << 24) | ((base & 0x0FF) << 16) | ((base & 0x0FF) << 8) | ((base & 0x0FF));

    sum = 0;

    blkCnt = dim_vec >> 2;

    while(blkCnt)
    {
       in=arm_nn_read_q7x4_ia(&pIn);
       in=__SSUB8(in,baseV);

        in1 = __SXTB16(__ROR(in, 8));

        /* extend remaining two q7_t values to q15_t values */
        in2 = __SXTB16(in);

#ifndef ARM_MATH_BIG_ENDIAN
        out2 = __PKHTB(in1, in2, 16);
        out1 = __PKHBT(in2, in1, 16);
#else
        out1 = __PKHTB(in1, in2, 16);
        out2 = __PKHBT(in2, in1, 16);
#endif


       shiftV = __USAT16(out1,LOG2Q7BITS);
       sum += 0x1 << (shiftV & 0x0FF);
       sum += 0x1 << ((shiftV >> 16) & 0x0FF);

       shiftV = __USAT16(out2,LOG2Q7BITS);
       sum += 0x1 << (shiftV & 0x0FF);
       sum += 0x1 << ((shiftV >> 16) & 0x0FF);

       blkCnt--;
    }

    blkCnt = dim_vec & 3;

    while(blkCnt)
    {
       shift = (uint8_t)__USAT(*pIn++ - base, LOG2Q7BITS);
       sum += 0x1 << shift;
       blkCnt--;
    }


    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;


    pIn=vec_in;

    blkCnt = dim_vec >> 2;
    while(blkCnt)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        in=arm_nn_read_q7x4_ia(&pIn);
        in=__SSUB8(pad,in);
        in=__SADD8(in,baseV);

        in1 = __SXTB16(__ROR(in, 8));

        /* extend remaining two q7_t values to q15_t values */
        in2 = __SXTB16(in);

#ifndef ARM_MATH_BIG_ENDIAN
        out2 = __PKHTB(in1, in2, 16);
        out1 = __PKHBT(in2, in1, 16);
#else
        out1 = __PKHTB(in1, in2, 16);
        out2 = __PKHBT(in2, in1, 16);
#endif

        shiftV = __USAT16(out1,5);
        *p_out++ = (q7_t) __SSAT((output_base >> (shiftV & 0x0FF)), 8);
        *p_out++ = (q7_t) __SSAT((output_base >> ((shiftV >> 16) & 0x0FF)), 8);

        shiftV = __USAT16(out2,5);
        *p_out++ = (q7_t) __SSAT((output_base >> (shiftV & 0x0FF)), 8);
        *p_out++ = (q7_t) __SSAT((output_base >> ((shiftV >> 16) & 0x0FF)), 8);

        blkCnt --;
    }


    blkCnt = dim_vec & 3;
    while(blkCnt)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__USAT(13 + base - *pIn++, 5);
        *p_out++ = (q7_t) __SSAT((output_base >> shift), 8);

        blkCnt --;
    }
#else
    q31_t     sum;
    int16_t   i;
    uint8_t   shift;
    q15_t     base;

    base = -128;

    /* We first search for the maximum */

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }


    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - Q7BITS;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        shift = (uint8_t)__USAT(vec_in[i] - base, LOG2Q7BITS);
        sum += 0x1 << shift;
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;


    for (i = 0; i < dim_vec; i++)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__USAT(13 + base - vec_in[i], 5);
        p_out[i] = (q7_t) __SSAT((output_base >> shift), 8);

    }
#endif
}
/**
 * @} end of Softmax group
 */
