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
    };

    Testing::errorID_t errorID;
    unsigned long lineNumber;
};

/*

Several test functions to implement tests in the client.
They should not be called directly but through macro
to get the line number.

(SNR functions to finish implementing)

*/

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
    if (pa.nbSamples() != pb.nbSamples())
    {
        throw (Error(EQUAL_ERROR,nb));
    }

    unsigned long i=0;

    T *ptrA = pa.ptr();
    T *ptrB = pb.ptr();

    for(i=0; i < pa.nbSamples(); i++)
    {
       assert_equal(nb,ptrA[i],ptrB[i]);
    }
};

template <typename T> 
void assert_near_equal(unsigned long nb,T pa, T pb, T threshold)
{
    if (abs(pa - pb) > threshold)
    {
         throw (Error(EQUAL_ERROR,nb));
    }
};

template <> 
void assert_near_equal(unsigned long nb,float32_t pa, float32_t pb, float32_t threshold);

template <typename T> 
void assert_near_equal(unsigned long nb,AnyPattern<T> &pa, AnyPattern<T> &pb, T threshold)
{
    if (pa.nbSamples() != pb.nbSamples())
    {
        throw (Error(NEAR_EQUAL_ERROR,nb));
    }

    unsigned long i=0;

    T *ptrA = pa.ptr();
    T *ptrB = pb.ptr();

    for(i=0; i < pa.nbSamples(); i++)
    {
       if (abs(ptrA[i] - ptrB[i]) > threshold)
       {
         throw (Error(NEAR_EQUAL_ERROR,nb));
       }
    }
};

template <> 
void assert_near_equal(unsigned long nb,AnyPattern<float32_t> &pa, AnyPattern<float32_t> &pb, float32_t threshold);


extern void assert_relative_error(unsigned long nb,float32_t &a, float32_t &b, float32_t threshold);
extern void assert_relative_error(unsigned long nb,AnyPattern<float32_t> &pa, AnyPattern<float32_t> &pb, float32_t threshold);

extern void assert_snr_error(unsigned long nb,AnyPattern<float32_t> &pa,AnyPattern<float32_t> &pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<q31_t> &pa,AnyPattern<q31_t> &pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<q15_t> &pa,AnyPattern<q15_t> &pb, float32_t threshold);
extern void assert_snr_error(unsigned long nb,AnyPattern<q7_t> &pa,AnyPattern<q7_t> &pb, float32_t threshold);

extern void assert_true(unsigned long nb,bool cond);
extern void assert_false(unsigned long nb,bool cond);

}

/*

Macros to use to implement tests.

*/
#define ASSERT_EQ(A,B) Client::assert_equal(__LINE__,A,B)
#define ASSERT_NEAR_EQ(A,B,THRESH) Client::assert_near_equal(__LINE__,A,B,THRESH)
#define ASSERT_REL_ERROR(A,B,THRESH) Client::assert_relative_error(__LINE__,A,B,THRESH)
#define ASSERT_SNR(A,B,SNR) Client::assert_snr_error(__LINE__,A,B,SNR)
#define ASSERT_TRUE(A) Client::assert_true(__LINE__,A)
#define ASSERT_FALSE(A) Client::assert_false(__LINE__,A)
#endif