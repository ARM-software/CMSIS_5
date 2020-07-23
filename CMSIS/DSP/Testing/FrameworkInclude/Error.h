/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Error.h
 * Description:  Error Header
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
#ifndef _ASSERT_H_
#define _ASSERT_H_
#include "arm_math_types.h"
#include "arm_math_types_f16.h"
#include <exception>
#include "Test.h"
#include "Pattern.h"


#define UNKNOWN_ERROR 1
#define EQUAL_ERROR 2
#define NEAR_EQUAL_ERROR 3
#define RELATIVE_ERROR 4
#define SNR_ERROR 5
#define DIFFERENT_LENGTH_ERROR 6
#define BOOL_ERROR 7
#define MEMORY_ALLOCATION_ERROR 8
#define EMPTY_PATTERN_ERROR 9
#define TAIL_NOT_EMPTY_ERROR 10
#define CLOSE_ERROR 11

namespace Client {

// Exception used by tests and runner
// to report errors
class Error: public std::exception
{
  public:
    Error(Testing::errorID_t id,unsigned long nb)
    {
        this->errorID = id;
        this->lineNumber = nb;
        this->details[0]='\0';
    };

    Error(Testing::errorID_t id,unsigned long nb, const char *details)
    {
        this->errorID = id;
        this->lineNumber = nb;
        strcpy(this->details,details);
    };

    Testing::errorID_t errorID;
    unsigned long lineNumber;
    char details[200];
};

/*

Several test functions to implement tests in the client.
They should not be called directly but through macro
to get the line number.

(SNR functions to finish implementing)

*/
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
extern void assert_relative_error(unsigned long nb,float16_t &a, float16_t &b, double threshold);
extern void assert_relative_error(unsigned long nb,AnyPattern<float16_t> &pa, AnyPattern<float16_t> &pb, double threshold);
#endif

extern void assert_relative_error(unsigned long nb,float32_t &a, float32_t &b, double threshold);
extern void assert_relative_error(unsigned long nb,AnyPattern<float32_t> &pa, AnyPattern<float32_t> &pb, double threshold);

extern void assert_relative_error(unsigned long nb,float64_t &a, float64_t &b, double threshold);
extern void assert_relative_error(unsigned long nb,AnyPattern<float64_t> &pa, AnyPattern<float64_t> &pb, double threshold);

/* Similar to numpy isclose */
extern void assert_close_error(unsigned long nb,float64_t &ref, float64_t &val, double absthreshold, double relthreshold);
extern void assert_close_error(unsigned long nb,AnyPattern<float64_t> &pref, AnyPattern<float64_t> &pval, double absthreshold, double relthreshold);

extern void assert_close_error(unsigned long nb,float32_t &ref, float32_t &val, double absthreshold, double relthreshold);
extern void assert_close_error(unsigned long nb,AnyPattern<float32_t> &pref, AnyPattern<float32_t> &pval, double absthreshold, double relthreshold);

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
extern void assert_close_error(unsigned long nb,float16_t &ref, float16_t &val, double absthreshold, double relthreshold);
extern void assert_close_error(unsigned long nb,AnyPattern<float16_t> &pref, AnyPattern<float16_t> &pval, double absthreshold, double relthreshold);
#endif

extern void assert_snr_error(unsigned long nb,AnyPattern<float64_t> &pa,AnyPattern<float64_t> &pb, float64_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<float32_t> &pa,AnyPattern<float32_t> &pb, float32_t threshold);

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
extern void assert_snr_error(unsigned long nb,AnyPattern<float16_t> &pa,AnyPattern<float16_t> &pb, float32_t threshold);
#endif

extern void assert_snr_error(unsigned long nb,AnyPattern<q63_t> &pa,AnyPattern<q63_t> &pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<q31_t> &pa,AnyPattern<q31_t> &pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<q15_t> &pa,AnyPattern<q15_t> &pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<q7_t> &pa,AnyPattern<q7_t> &pb, float32_t threshold);

extern void assert_snr_error(unsigned long nb,float64_t pa,float64_t pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,float32_t pa,float32_t pb, float32_t threshold);

#if !defined (__CC_ARM) && defined(ARM_FLOAT16_SUPPORTED)
extern void assert_snr_error(unsigned long nb,float16_t pa,float16_t pb, float32_t threshold);
#endif 

extern void assert_snr_error(unsigned long nb,q63_t pa,q63_t pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,q31_t pa,q31_t pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,q15_t pa,q15_t pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,q7_t pa,q7_t pb, float32_t threshold);

extern void assert_true(unsigned long nb,bool cond);
extern void assert_false(unsigned long nb,bool cond);

extern void assert_not_empty(unsigned long nb, AnyPattern<float64_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<float32_t> &p);

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
extern void assert_not_empty(unsigned long nb, AnyPattern<float16_t> &p);
#endif 

extern void assert_not_empty(unsigned long nb, AnyPattern<q63_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<q31_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<q15_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<q7_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<uint32_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<uint16_t> &p);
extern void assert_not_empty(unsigned long nb, AnyPattern<uint8_t> &p);

}

/*

Macros to use to implement tests.

*/
#define ASSERT_EQ(A,B) Client::assert_equal(__LINE__,A,B)
#define ASSERT_NEAR_EQ(A,B,THRESH) Client::assert_near_equal(__LINE__,A,B,THRESH)
#define ASSERT_REL_ERROR(A,B,THRESH) Client::assert_relative_error(__LINE__,A,B,THRESH)
#define ASSERT_CLOSE_ERROR(A,B,ABSTHRESH,RELTHRESH) Client::assert_close_error(__LINE__,A,B,ABSTHRESH,RELTHRESH)
#define ASSERT_SNR(A,B,SNR) Client::assert_snr_error(__LINE__,A,B,SNR)
#define ASSERT_TRUE(A) Client::assert_true(__LINE__,A)
#define ASSERT_FALSE(A) Client::assert_false(__LINE__,A)
#define ASSERT_NOT_EMPTY(A) Client::assert_not_empty(__LINE__,A)
#define ASSERT_EMPTY_TAIL(A) if (!A.isTailEmpty()) throw (Client::Error(TAIL_NOT_EMPTY_ERROR,__LINE__))

namespace Client {

using namespace std;

template <typename T> 
void assert_equal(unsigned long nb,T pa, T pb)
{
    if (pa != pb)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
   
};

template <typename T> 
void assert_equal(unsigned long nb,AnyPattern<T> &pa, AnyPattern<T> &pb)
{
    ASSERT_NOT_EMPTY(pa);
    ASSERT_NOT_EMPTY(pb);
    
    if (pa.nbSamples() != pb.nbSamples())
    {
        throw (Error(EQUAL_ERROR,nb));
    }

    unsigned long i=0;
    char id[40];

    T *ptrA = pa.ptr();
    T *ptrB = pb.ptr();

    for(i=0; i < pa.nbSamples(); i++)
    {
       try
       {
          assert_equal(nb,ptrA[i],ptrB[i]);
       }
       catch(Error &err)
       {          
          sprintf(id," (nb=%lu)",i+1);
          strcat(err.details,id);
          throw(err);
       }
    }
};

template <typename T> 
void assert_near_equal(unsigned long nb,T pa, T pb, T threshold)
{
    if (abs(pa - pb) > threshold)
    {
         throw (Error(NEAR_EQUAL_ERROR,nb));
    }
};

template <> 
void assert_near_equal(unsigned long nb,double pa, double pb, double threshold);
template <> 
void assert_near_equal(unsigned long nb,float32_t pa, float32_t pb, float32_t threshold);
template <> 
void assert_near_equal(unsigned long nb,q63_t pa, q63_t pb, q63_t threshold);
template <> 
void assert_near_equal(unsigned long nb,q31_t pa, q31_t pb, q31_t threshold);
template <> 
void assert_near_equal(unsigned long nb,q15_t pa, q15_t pb, q15_t threshold);
template <> 
void assert_near_equal(unsigned long nb,q7_t pa, q7_t pb, q7_t threshold);

template <typename T> 
void assert_near_equal(unsigned long nb,AnyPattern<T> &pa, AnyPattern<T> &pb, T threshold)
{

    ASSERT_NOT_EMPTY(pa);
    ASSERT_NOT_EMPTY(pb);

    if (pa.nbSamples() != pb.nbSamples())
    {
        throw (Error(NEAR_EQUAL_ERROR,nb));
    }

    unsigned long i=0;
    char id[40];

    T *ptrA = pa.ptr();
    T *ptrB = pb.ptr();

    for(i=0; i < pa.nbSamples(); i++)
    {
       
       try
       {
          assert_near_equal(nb,ptrA[i],ptrB[i],threshold);
       }
       catch(Error &err)
       {          
          sprintf(id," (nb=%lu)",i+1);
          strcat(err.details,id);
          throw(err);
       }
    }
};


}
#endif
