/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Error.cpp
 * Description:  Error functions
 *
 * $Date:        20. June 2019
 * $Revision:    V1.0.0
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2019 ARM Limited or its affiliates. All rights reserved.
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
#include "Error.h"

namespace Client {

template <> 
void assert_near_equal(unsigned long nb,float32_t pa, float32_t pb, float32_t threshold)
{
    if (fabs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};

template <>
void assert_near_equal(unsigned long nb,AnyPattern<float32_t> &pa, AnyPattern<float32_t> &pb, float32_t threshold)
{
    if (pa.nbSamples() != pb.nbSamples())
    {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
    }

    unsigned long i=0;

    float32_t *ptrA = pa.ptr();
    float32_t *ptrB = pb.ptr();

    for(i=0; i < pa.nbSamples(); i++)
    {
       if (fabs(ptrA[i] - ptrB[i]) > threshold)
       {
         throw (Error(NEAR_EQUAL_ERROR,nb));
       }
    }
};

void assert_relative_error(unsigned long nb,float32_t &a, float32_t &b, float32_t threshold)
{
    float32_t rel,delta,average;

    delta=abs(a-b);
    average = (abs(a) + abs(b)) / 2.0;
    if (average !=0)
    {
        rel = delta / average;
        if (rel > threshold)
        {
            throw (Error(RELATIVE_ERROR,nb));
        }
    }
};

void assert_relative_error(unsigned long nb,AnyPattern<float32_t> &pa, AnyPattern<float32_t> &pb, float32_t threshold)
{
    if (pa.nbSamples() != pb.nbSamples())
    {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
    }

    unsigned long i=0;

    float32_t *ptrA = pa.ptr();
    float32_t *ptrB = pb.ptr();

    for(i=0; i < pa.nbSamples(); i++)
    {
       assert_relative_error(nb,ptrA[i],ptrB[i],threshold);
    }
};

/*

SNR functions below are just computing the error noise.
Signal power needs to be computed.

*/

/**
 * @brief  Caluclation of SNR
 * @param  float*   Pointer to the reference buffer
 * @param  float*   Pointer to the test buffer
 * @param  uint32_t     total number of samples
 * @return float    SNR
 * The function Caluclates signal to noise ratio for the reference output
 * and test output
 */

float arm_snr_f32(float *pRef, float *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
  int temp;
  int *test;

  for (i = 0; i < buffSize; i++)
    {
      /* Checking for a NAN value in pRef array */
      test =   (int *)(&pRef[i]);
      temp =  *test;

      if (temp == 0x7FC00000)
      {
        return(100000.0);
      }

      /* Checking for a NAN value in pTest array */
      test =   (int *)(&pTest[i]);
      temp =  *test;

      if (temp == 0x7FC00000)
      {
        return(100000.0);
      }
      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }

    /* Checking for a NAN value in EnergyError */
    test =   (int *)(&EnergyError);
    temp =  *test;

    if (temp == 0x7FC00000)
    {
        return(100000.0);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

    /* Checking for a NAN value in SNR */
    test =   (int *)(&SNR);
    temp =  *test;

    if (temp == 0x7FC00000)
    {
        return(100000.0);
    }

  return (SNR);

}

float arm_snr_q31(q31_t *pRef, q31_t *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
  int temp;
  float32_t test,ref;

  for (i = 0; i < buffSize; i++)
    {
     
      test = (float32_t)pTest[i]  / 2147483648.0f;
      ref = (float32_t)pRef[i]  / 2147483648.0f;

      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

  return (SNR);

}

float arm_snr_q15(q15_t *pRef, q15_t *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
  int temp;
  float32_t test,ref;

  for (i = 0; i < buffSize; i++)
    {
     
      test = (float32_t)pTest[i]   / 32768.0f;
      ref = (float32_t)pRef[i]  / 32768.0f;

      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

  return (SNR);

}

float arm_snr_q7(q7_t *pRef, q7_t *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
  int temp;
  float32_t test,ref;

  for (i = 0; i < buffSize; i++)
    {
     
      test = (float32_t)pTest[i]   / 128.0f;
      ref = (float32_t)pRef[i]  / 128.0f;

      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

  return (SNR);

}

double arm_snr_f64(double *pRef, double *pTest, uint32_t buffSize)
{
  double EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  double SNR;
  int temp;
  int *test;

  for (i = 0; i < buffSize; i++)
    {
      /* Checking for a NAN value in pRef array */
      test =   (int *)(&pRef[i]);
      temp =  *test;

      if (temp == 0x7FC00000)
      {
        return(100000.0);
      }

      /* Checking for a NAN value in pTest array */
      test =   (int *)(&pTest[i]);
      temp =  *test;

      if (temp == 0x7FC00000)
      {
        return(100000.0);
      }
      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }

    /* Checking for a NAN value in EnergyError */
    test =   (int *)(&EnergyError);
    temp =  *test;

    if (temp == 0x7FC00000)
    {
        return(100000.0);
    }


  SNR = 10 * log10 (EnergySignal / EnergyError);

    /* Checking for a NAN value in SNR */
    test =   (int *)(&SNR);
    temp =  *test;

    if (temp == 0x7FC00000)
    {
        return(10000.0);
    }

  return (SNR);

}

void assert_snr_error(unsigned long nb,AnyPattern<float32_t> &pa,AnyPattern<float32_t> &pb, float32_t threshold)
{
   float32_t snr;

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   float32_t *ptrA = pa.ptr();
   float32_t *ptrB = pb.ptr();

   snr = arm_snr_f32(ptrA, ptrB, pa.nbSamples());


   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }
}

void assert_snr_error(unsigned long nb,AnyPattern<q31_t> &pa,AnyPattern<q31_t> &pb, float32_t threshold)
{
   float32_t snr;

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   q31_t *ptrA = pa.ptr();
   q31_t *ptrB = pb.ptr();

   snr = arm_snr_q31(ptrA, ptrB, pa.nbSamples());


   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,AnyPattern<q15_t> &pa,AnyPattern<q15_t> &pb, float32_t threshold)
{
   float32_t snr;

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   q15_t *ptrA = pa.ptr();
   q15_t *ptrB = pb.ptr();

   snr = arm_snr_q15(ptrA, ptrB, pa.nbSamples());


   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,AnyPattern<q7_t> &pa,AnyPattern<q7_t> &pb, float32_t threshold)
{
   float32_t snr;

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   q7_t *ptrA = pa.ptr();
   q7_t *ptrB = pb.ptr();

   snr = arm_snr_q7(ptrA, ptrB, pa.nbSamples());


   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_true(unsigned long nb,bool cond)
{
   if (!cond)
   {
     throw (Error(BOOL_ERROR,nb));
   }
}

void assert_false(unsigned long nb,bool cond)
{
   if (cond)
   {
      throw (Error(BOOL_ERROR,nb));
   }
}

}