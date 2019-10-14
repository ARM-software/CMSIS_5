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
#include "arm_math.h"
#include "Error.h"
#include <stdlib.h> 

namespace Client {

template <typename T> 
void assert_not_empty_generic(unsigned long nb, AnyPattern<T> &p)
{
   if (p.nbSamples() == 0)                    
   {                                          
        throw (Error(EMPTY_PATTERN_ERROR,nb));
   }                                          
   if (p.ptr() == NULL)                       
   {                                          
        throw (Error(EMPTY_PATTERN_ERROR,nb));
   }
};


template <> 
void assert_near_equal(unsigned long nb,double pa, double pb, double threshold)
{
    if (fabs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};

template <> 
void assert_near_equal(unsigned long nb,float32_t pa, float32_t pb, float32_t threshold)
{
    if (fabs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};


template <> 
void assert_near_equal(unsigned long nb,q63_t pa, q63_t pb, q63_t threshold)
{
    if (abs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};

template <> 
void assert_near_equal(unsigned long nb,q31_t pa, q31_t pb, q31_t threshold)
{
    if (abs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};

template <> 
void assert_near_equal(unsigned long nb,q15_t pa, q15_t pb, q15_t threshold)
{
    if (abs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};

template <> 
void assert_near_equal(unsigned long nb,q7_t pa, q7_t pb, q7_t threshold)
{
    if (abs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};


void assert_not_empty(unsigned long nb, AnyPattern<float32_t> &p)
{
  assert_not_empty_generic(nb,p);
}

void assert_not_empty(unsigned long nb, AnyPattern<q63_t> &p)
{
  assert_not_empty_generic(nb,p);
}

void assert_not_empty(unsigned long nb, AnyPattern<q31_t> &p)
{
  assert_not_empty_generic(nb,p);
}

void assert_not_empty(unsigned long nb, AnyPattern<q15_t> &p)
{
  assert_not_empty_generic(nb,p);
}

void assert_not_empty(unsigned long nb, AnyPattern<q7_t> &p)
{
  assert_not_empty_generic(nb,p);
}

void assert_relative_error(unsigned long nb,float32_t &a, float32_t &b, double threshold)
{
    double rel,delta,average;

    delta=abs(a-b);
    average = (abs(a) + abs(b)) / 2.0f;
    if (average !=0)
    {
        rel = delta / average;
        //printf("%6.9f %6.9f %6.9f\n",a,b,rel);
        if (rel > threshold)
        {
            throw (Error(RELATIVE_ERROR,nb));
        }
    }
};

void assert_relative_error(unsigned long nb,AnyPattern<float32_t> &pa, AnyPattern<float32_t> &pb, double threshold)
{
    ASSERT_NOT_EMPTY(pa);
    ASSERT_NOT_EMPTY(pb);

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



/**
 * @brief  Calculation of SNR
 * @param  float*   Pointer to the reference buffer
 * @param  float*   Pointer to the test buffer
 * @param  uint32_t     total number of samples
 * @return float    SNR
 * The function calculates signal to noise ratio for the reference output
 * and test output
 */

/* If NaN, force SNR to 0.0 to ensure test will fail */
#define IFNANRETURNZERO(val)\
     if (isnan((val)))      \
     {                      \
       return(0.0);         \
     }

#define IFINFINITERETURN(val,def)\
     if (isinf((val)))           \
     {                           \
       if ((val) > 0)            \
       {                         \
          return(def);           \
       }                         \
       else                      \
       {                         \
         return(-def);           \
       }                         \
     }

float arm_snr_f32(float *pRef, float *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
 
  for (i = 0; i < buffSize; i++)
    {
      /* Checking for a NAN value in pRef array */
      IFNANRETURNZERO(pRef[i]);
      
      /* Checking for a NAN value in pTest array */
      IFNANRETURNZERO(pTest[i]);

      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }

    /* Checking for a NAN value in EnergyError */
    IFNANRETURNZERO(EnergyError);


    SNR = 10 * log10f (EnergySignal / EnergyError);

    /* Checking for a NAN value in SNR */
    IFNANRETURNZERO(SNR);
    IFINFINITERETURN(SNR,100000.0);
    

  return (SNR);

}

float arm_snr_q63(q63_t *pRef, q63_t *pTest, uint32_t buffSize)
{
  double EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
 
  double testVal,refVal;

  for (i = 0; i < buffSize; i++)
    {
     
      testVal = ((double)pTest[i])  / 9223372036854775808.0;
      refVal = ((double)pRef[i])  / 9223372036854775808.0;

      EnergySignal += refVal * refVal;
      EnergyError += (refVal - testVal) * (refVal - testVal);

    }


  SNR = 10 * log10 (EnergySignal / EnergyError);


  /* Checking for a NAN value in SNR */
   IFNANRETURNZERO(SNR);
   IFINFINITERETURN(SNR,100000.0);
   
  //printf("SNR = %f\n",SNR);

  return (SNR);

}

float arm_snr_q31(q31_t *pRef, q31_t *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
 
  float32_t testVal,refVal;

  for (i = 0; i < buffSize; i++)
    {
     
      testVal = ((float32_t)pTest[i])  / 2147483648.0f;
      refVal = ((float32_t)pRef[i])  / 2147483648.0f;

      EnergySignal += refVal * refVal;
      EnergyError += (refVal - testVal) * (refVal - testVal);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

  /* Checking for a NAN value in SNR */
   IFNANRETURNZERO(SNR);
   IFINFINITERETURN(SNR,100000.0);
   
   //printf("SNR = %f\n",SNR);

  return (SNR);

}

float arm_snr_q15(q15_t *pRef, q15_t *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR; 

  float32_t testVal,refVal;

  for (i = 0; i < buffSize; i++)
    {
     
      testVal = ((float32_t)pTest[i])   / 32768.0f;
      refVal = ((float32_t)pRef[i])  / 32768.0f;

      EnergySignal += refVal * refVal;
      EnergyError += (refVal - testVal) * (refVal - testVal);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

  /* Checking for a NAN value in SNR */
  IFNANRETURNZERO(SNR);
  IFINFINITERETURN(SNR,100000.0);

  //printf("SNR = %f\n",SNR);

  return (SNR);

}

float arm_snr_q7(q7_t *pRef, q7_t *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
  
  float32_t testVal,refVal;

  for (i = 0; i < buffSize; i++)
    {
     
      testVal = ((float32_t)pTest[i])   / 128.0f;
      refVal = ((float32_t)pRef[i])  / 128.0f;

      EnergySignal += refVal * refVal;
      EnergyError += (refVal - testVal) * (refVal - testVal);
    }


  SNR = 10 * log10f (EnergySignal / EnergyError);

  IFNANRETURNZERO(SNR);
  IFINFINITERETURN(SNR,100000.0);

  return (SNR);

}

double arm_snr_f64(double *pRef, double *pTest, uint32_t buffSize)
{
  double EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  double SNR;
  
  for (i = 0; i < buffSize; i++)
    {
      /* Checking for a NAN value in pRef array */
      IFNANRETURNZERO(pRef[i]);
      

      /* Checking for a NAN value in pTest array */
      IFNANRETURNZERO(pTest[i]);
      
      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }

    /* Checking for a NAN value in EnergyError */
    IFNANRETURNZERO(EnergyError);

    SNR = 10 * log10 (EnergySignal / EnergyError);

    /* Checking for a NAN value in SNR */
    IFNANRETURNZERO(SNR);
    IFINFINITERETURN(SNR,100000.0);

  return (SNR);

}

void assert_snr_error(unsigned long nb,AnyPattern<float32_t> &pa,AnyPattern<float32_t> &pb, float32_t threshold)
{
   float32_t snr;

   ASSERT_NOT_EMPTY(pa);
   ASSERT_NOT_EMPTY(pb);

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   float32_t *ptrA = pa.ptr();
   float32_t *ptrB = pb.ptr();

   snr = arm_snr_f32(ptrA, ptrB, pa.nbSamples());

   //printf("SNR = %f\n",snr);
   
   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }
}

void assert_snr_error(unsigned long nb,float32_t a,float32_t b, float32_t threshold)
{
   float32_t snr;

   snr = arm_snr_f32(&a, &b, 1);

   //printf("SNR = %f, %f %f\n",snr,a,b);
   
   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }
}

void assert_snr_error(unsigned long nb,AnyPattern<q63_t> &pa,AnyPattern<q63_t> &pb, float32_t threshold)
{
   float32_t snr;

   ASSERT_NOT_EMPTY(pa);
   ASSERT_NOT_EMPTY(pb);

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   q63_t *ptrA = pa.ptr();
   q63_t *ptrB = pb.ptr();


   snr = arm_snr_q63(ptrA, ptrB, pa.nbSamples());

   //printf("SNR = %f\n",snr);

   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,q63_t a,q63_t b, float32_t threshold)
{
   float32_t snr;

   snr = arm_snr_q63(&a, &b, 1);

   //printf("SNR = %f\n",snr);

   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,AnyPattern<q31_t> &pa,AnyPattern<q31_t> &pb, float32_t threshold)
{
   float32_t snr;

   ASSERT_NOT_EMPTY(pa);
   ASSERT_NOT_EMPTY(pb);

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

void assert_snr_error(unsigned long nb,q31_t a,q31_t b, float32_t threshold)
{
   float32_t snr;

   snr = arm_snr_q31(&a, &b, 1);


   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,AnyPattern<q15_t> &pa,AnyPattern<q15_t> &pb, float32_t threshold)
{
   float32_t snr;

   ASSERT_NOT_EMPTY(pa);
   ASSERT_NOT_EMPTY(pb);

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   q15_t *ptrA = pa.ptr();
   q15_t *ptrB = pb.ptr();

   snr = arm_snr_q15(ptrA, ptrB, pa.nbSamples());

   //printf("SNR = %f\n",snr);

   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,q15_t a,q15_t b, float32_t threshold)
{
   float32_t snr;

   snr = arm_snr_q15(&a, &b, 1);

   //printf("SNR = %f\n",snr);

   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,AnyPattern<q7_t> &pa,AnyPattern<q7_t> &pb, float32_t threshold)
{
   float32_t snr;

   ASSERT_NOT_EMPTY(pa);
   ASSERT_NOT_EMPTY(pb);

   if (pa.nbSamples() != pb.nbSamples())
   {
        throw (Error(DIFFERENT_LENGTH_ERROR,nb));
   }

   q7_t *ptrA = pa.ptr();
   q7_t *ptrB = pb.ptr();

   snr = arm_snr_q7(ptrA, ptrB, pa.nbSamples());

   //printf("SNR = %f\n",snr);

   if (snr < threshold)
   {
     throw (Error(SNR_ERROR,nb));
   }

}

void assert_snr_error(unsigned long nb,q7_t a,q7_t b, float32_t threshold)
{
   float32_t snr;

   snr = arm_snr_q7(&a, &b, 1);

   //printf("SNR = %f\n",snr);

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
