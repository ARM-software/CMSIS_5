/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_cfft_radix2_f16.c
 * Description:  Radix-2 Decimation in Frequency CFFT & CIFFT Floating point processing function
 *
 * $Date:        23 April 2021
 * $Revision:    V1.9.0
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
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

#include "dsp/transform_functions_f16.h"

#if defined(ARM_FLOAT16_SUPPORTED)

void arm_radix2_butterfly_f16(
        float16_t * pSrc,
        uint32_t fftLen,
  const float16_t * pCoef,
        uint16_t twidCoefModifier);

void arm_radix2_butterfly_inverse_f16(
        float16_t * pSrc,
        uint32_t fftLen,
  const float16_t * pCoef,
        uint16_t twidCoefModifier,
        float16_t onebyfftLen);

extern void arm_bitreversal_f16(
        float16_t * pSrc,
        uint16_t fftSize,
        uint16_t bitRevFactor,
  const uint16_t * pBitRevTab);

/**
  @ingroup groupTransforms
 */

/**
  @addtogroup ComplexFFT
  @{
 */

/**
  @brief         Radix-2 CFFT/CIFFT.
  @deprecated    Do not use this function. It has been superseded by \ref arm_cfft_f16 and will be removed in the future
  @param[in]     S    points to an instance of the floating-point Radix-2 CFFT/CIFFT structure
  @param[in,out] pSrc points to the complex data buffer of size <code>2*fftLen</code>. Processing occurs in-place
  @return        none
 */

void arm_cfft_radix2_f16(
const arm_cfft_radix2_instance_f16 * S,
      float16_t * pSrc)
{

   if (S->ifftFlag == 1U)
   {
      /* Complex IFFT radix-2 */
      arm_radix2_butterfly_inverse_f16(pSrc, S->fftLen, S->pTwiddle,
      S->twidCoefModifier, S->onebyfftLen);
   }
   else
   {
      /* Complex FFT radix-2 */
      arm_radix2_butterfly_f16(pSrc, S->fftLen, S->pTwiddle,
      S->twidCoefModifier);
   }

   if (S->bitReverseFlag == 1U)
   {
      /* Bit Reversal */
      arm_bitreversal_f16(pSrc, S->fftLen, S->bitRevFactor, S->pBitRevTable);
   }

}


/**
  @} end of ComplexFFT group
 */



/* ----------------------------------------------------------------------
** Internal helper function used by the FFTs
** ------------------------------------------------------------------- */

/*
* @brief  Core function for the floating-point CFFT butterfly process.
* @param[in, out] *pSrc            points to the in-place buffer of floating-point data type.
* @param[in]      fftLen           length of the FFT.
* @param[in]      *pCoef           points to the twiddle coefficient buffer.
* @param[in]      twidCoefModifier twiddle coefficient modifier that supports different size FFTs with the same twiddle factor table.
* @return none.
*/

void arm_radix2_butterfly_f16(
float16_t * pSrc,
uint32_t fftLen,
const float16_t * pCoef,
uint16_t twidCoefModifier)
{

   uint32_t i, j, k, l;
   uint32_t n1, n2, ia;
   float16_t xt, yt, cosVal, sinVal;
   float16_t p0, p1, p2, p3;
   float16_t a0, a1;

#if defined (ARM_MATH_DSP)

   /*  Initializations for the first stage */
   n2 = fftLen >> 1;
   ia = 0;
   i = 0;

   // loop for groups
   for (k = n2; k > 0; k--)
   {
      cosVal = pCoef[ia * 2];
      sinVal = pCoef[(ia * 2) + 1];

      /*  Twiddle coefficients index modifier */
      ia += twidCoefModifier;

      /*  index calculation for the input as, */
      /*  pSrc[i + 0], pSrc[i + fftLen/1] */
      l = i + n2;

      /*  Butterfly implementation */
      a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
      xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

      yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];
      a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];

      p0 = (_Float16)xt * (_Float16)cosVal;
      p1 = (_Float16)yt * (_Float16)sinVal;
      p2 = (_Float16)yt * (_Float16)cosVal;
      p3 = (_Float16)xt * (_Float16)sinVal;

      pSrc[2 * i]     = a0;
      pSrc[2 * i + 1] = a1;

      pSrc[2 * l]     = (_Float16)p0 + (_Float16)p1;
      pSrc[2 * l + 1] = (_Float16)p2 - (_Float16)p3;

      i++;
   }                             // groups loop end

   twidCoefModifier <<= 1U;

   // loop for stage
   for (k = n2; k > 2; k = k >> 1)
   {
      n1 = n2;
      n2 = n2 >> 1;
      ia = 0;

      // loop for groups
      j = 0;
      do
      {
         cosVal = pCoef[ia * 2];
         sinVal = pCoef[(ia * 2) + 1];
         ia += twidCoefModifier;

         // loop for butterfly
         i = j;
         do
         {
            l = i + n2;
            a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
            xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

            yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];
            a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];

            p0 = (_Float16)xt * (_Float16)cosVal;
            p1 = (_Float16)yt * (_Float16)sinVal;
            p2 = (_Float16)yt * (_Float16)cosVal;
            p3 = (_Float16)xt * (_Float16)sinVal;

            pSrc[2 * i] = a0;
            pSrc[2 * i + 1] = a1;

            pSrc[2 * l]     = (_Float16)p0 + (_Float16)p1;
            pSrc[2 * l + 1] = (_Float16)p2 - (_Float16)p3;

            i += n1;
         } while ( i < fftLen );                        // butterfly loop end
         j++;
      } while ( j < n2);                          // groups loop end
      twidCoefModifier <<= 1U;
   }                             // stages loop end

   // loop for butterfly
   for (i = 0; i < fftLen; i += 2)
   {
      a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * i + 2];
      xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * i + 2];

      yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * i + 3];
      a1 = (_Float16)pSrc[2 * i + 3] + (_Float16)pSrc[2 * i + 1];

      pSrc[2 * i] = a0;
      pSrc[2 * i + 1] = a1;
      pSrc[2 * i + 2] = xt;
      pSrc[2 * i + 3] = yt;
   }                             // groups loop end

#else

   n2 = fftLen;

   // loop for stage
   for (k = fftLen; k > 1; k = k >> 1)
   {
      n1 = n2;
      n2 = n2 >> 1;
      ia = 0;

      // loop for groups
      j = 0;
      do
      {
         cosVal = pCoef[ia * 2];
         sinVal = pCoef[(ia * 2) + 1];
         ia += twidCoefModifier;

         // loop for butterfly
         i = j;
         do
         {
            l = i + n2;
            a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
            xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

            yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];
            a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];

            p0 = (_Float16)xt * (_Float16)cosVal;
            p1 = (_Float16)yt * (_Float16)sinVal;
            p2 = (_Float16)yt * (_Float16)cosVal;
            p3 = (_Float16)xt * (_Float16)sinVal;

            pSrc[2 * i] = a0;
            pSrc[2 * i + 1] = a1;

            pSrc[2 * l]     = (_Float16)p0 + (_Float16)p1;
            pSrc[2 * l + 1] = (_Float16)p2 - (_Float16)p3;

            i += n1;
         } while (i < fftLen);
         j++;
      } while (j < n2);
      twidCoefModifier <<= 1U;
   }

#endif //    #if defined (ARM_MATH_DSP)

}


void arm_radix2_butterfly_inverse_f16(
float16_t * pSrc,
uint32_t fftLen,
const float16_t * pCoef,
uint16_t twidCoefModifier,
float16_t onebyfftLen)
{

   uint32_t i, j, k, l;
   uint32_t n1, n2, ia;
   float16_t xt, yt, cosVal, sinVal;
   float16_t p0, p1, p2, p3;
   float16_t a0, a1;

#if defined (ARM_MATH_DSP)

   n2 = fftLen >> 1;
   ia = 0;

   // loop for groups
   for (i = 0; i < n2; i++)
   {
      cosVal = pCoef[ia * 2];
      sinVal = pCoef[(ia * 2) + 1];
      ia += twidCoefModifier;

      l = i + n2;
      a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
      xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

      yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];
      a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];

      p0 = (_Float16)xt * (_Float16)cosVal;
      p1 = (_Float16)yt * (_Float16)sinVal;
      p2 = (_Float16)yt * (_Float16)cosVal;
      p3 = (_Float16)xt * (_Float16)sinVal;

      pSrc[2 * i] = a0;
      pSrc[2 * i + 1] = a1;

      pSrc[2 * l]     = (_Float16)p0 - (_Float16)p1;
      pSrc[2 * l + 1] = (_Float16)p2 + (_Float16)p3;
   }                             // groups loop end

   twidCoefModifier <<= 1U;

   // loop for stage
   for (k = fftLen / 2; k > 2; k = k >> 1)
   {
      n1 = n2;
      n2 = n2 >> 1;
      ia = 0;

      // loop for groups
      j = 0;
      do
      {
         cosVal = pCoef[ia * 2];
         sinVal = pCoef[(ia * 2) + 1];
         ia += twidCoefModifier;

         // loop for butterfly
         i = j;
         do
         {
            l = i + n2;
            a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
            xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

            yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];
            a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];

            p0 = (_Float16)xt * (_Float16)cosVal;
            p1 = (_Float16)yt * (_Float16)sinVal;
            p2 = (_Float16)yt * (_Float16)cosVal;
            p3 = (_Float16)xt * (_Float16)sinVal;

            pSrc[2 * i] = a0;
            pSrc[2 * i + 1] = a1;

            pSrc[2 * l]     = (_Float16)p0 - (_Float16)p1;
            pSrc[2 * l + 1] = (_Float16)p2 + (_Float16)p3;

            i += n1;
         } while ( i < fftLen );                 // butterfly loop end
         j++;
      } while (j < n2);                      // groups loop end

      twidCoefModifier <<= 1U;
   }                             // stages loop end

   // loop for butterfly
   for (i = 0; i < fftLen; i += 2)
   {
      a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * i + 2];
      xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * i + 2];

      a1 = (_Float16)pSrc[2 * i + 3] + (_Float16)pSrc[2 * i + 1];
      yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * i + 3];

      p0 = (_Float16)a0 * (_Float16)onebyfftLen;
      p2 = (_Float16)xt * (_Float16)onebyfftLen;
      p1 = (_Float16)a1 * (_Float16)onebyfftLen;
      p3 = (_Float16)yt * (_Float16)onebyfftLen;

      pSrc[2 * i] = p0;
      pSrc[2 * i + 1] = p1;
      pSrc[2 * i + 2] = p2;
      pSrc[2 * i + 3] = p3;
   }                             // butterfly loop end

#else

   n2 = fftLen;

   // loop for stage
   for (k = fftLen; k > 2; k = k >> 1)
   {
      n1 = n2;
      n2 = n2 >> 1;
      ia = 0;

      // loop for groups
      j = 0;
      do
      {
         cosVal = pCoef[ia * 2];
         sinVal = pCoef[(ia * 2) + 1];
         ia = ia + twidCoefModifier;

         // loop for butterfly
         i = j;
         do
         {
            l = i + n2;
            a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
            xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

            yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];
            a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];

            p0 = (_Float16)xt * (_Float16)cosVal;
            p1 = (_Float16)yt * (_Float16)sinVal;
            p2 = (_Float16)yt * (_Float16)cosVal;
            p3 = (_Float16)xt * (_Float16)sinVal;

            pSrc[2 * i] = a0;
            pSrc[2 * i + 1] = a1;

            pSrc[2 * l]     = (_Float16)p0 - (_Float16)p1;
            pSrc[2 * l + 1] = (_Float16)p2 + (_Float16)p3;

            i += n1;
         } while ( i < fftLen );                    // butterfly loop end
         j++;
      } while ( j < n2 );                      // groups loop end

      twidCoefModifier = twidCoefModifier << 1U;
   }                             // stages loop end

   n1 = n2;
   n2 = n2 >> 1;

   // loop for butterfly
   for (i = 0; i < fftLen; i += n1)
   {
      l = i + n2;

      a0 = (_Float16)pSrc[2 * i] + (_Float16)pSrc[2 * l];
      xt = (_Float16)pSrc[2 * i] - (_Float16)pSrc[2 * l];

      a1 = (_Float16)pSrc[2 * l + 1] + (_Float16)pSrc[2 * i + 1];
      yt = (_Float16)pSrc[2 * i + 1] - (_Float16)pSrc[2 * l + 1];

      p0 = (_Float16)a0 * (_Float16)onebyfftLen;
      p2 = (_Float16)xt * (_Float16)onebyfftLen;
      p1 = (_Float16)a1 * (_Float16)onebyfftLen;
      p3 = (_Float16)yt * (_Float16)onebyfftLen;

      pSrc[2 * i] = p0;
      pSrc[2U * l] = p2;

      pSrc[2 * i + 1] = p1;
      pSrc[2U * l + 1U] = p3;
   }                             // butterfly loop end

#endif //      #if defined (ARM_MATH_DSP)

}


#endif /* #if defined(ARM_FLOAT16_SUPPORTED) */
