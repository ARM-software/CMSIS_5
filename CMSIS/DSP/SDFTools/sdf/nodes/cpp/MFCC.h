/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        MFCC.h
 * Description:  Node for CMSIS-DSP MFCC
 *
 * $Date:        06 October 2021
 * $Revision:    V1.10.0
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
#ifndef _MFCC_H_ 
#define _MFCC_H_

#include <vector>

template<typename IN, int inputSize,typename OUT,int outputSize>
class MFCC;

/*

The MFCC configuration data has to be generated with the script DSP/Scripts/GenMFCCDataForCPP.py.
It is using a yaml file to describe the configuration

*/

/*

The CMSIS-DSP MFCC F32

*/
template<int inputSize,int outputSize>
class MFCC<float32_t,inputSize,float32_t,outputSize>: public GenericNode<float32_t,inputSize,float32_t,outputSize>
{
public:
    MFCC(FIFOBase<float32_t> &src,FIFOBase<float32_t> &dst,const arm_mfcc_instance_f32 *config):GenericNode<float32_t,inputSize,float32_t,outputSize>(src,dst){
         mfccConfig = config;
#if defined(ARM_MFCC_CFFT_BASED)
         memory.resize(2*inputSize);
#else
         memory.resize(inputSize + 2);
#endif
    };

    int run(){
        float32_t *a=this->getReadBuffer();
        float32_t *b=this->getWriteBuffer();
        arm_mfcc_f32(mfccConfig,a,b,memory.data());
        return(0);
    };

    const arm_mfcc_instance_f32 *mfccConfig;
    std::vector<float32_t> memory;
};

#if defined(ARM_FLOAT16_SUPPORTED)
/*

The CMSIS-DSP MFCC F16

*/
template<int inputSize,int outputSize>
class MFCC<float16_t,inputSize,float16_t,outputSize>: public GenericNode<float16_t,inputSize,float16_t,outputSize>
{
public:
    MFCC(FIFOBase<float16_t> &src,FIFOBase<float16_t> &dst,const arm_mfcc_instance_f16 *config):GenericNode<float16_t,inputSize,float16_t,outputSize>(src,dst){
         mfccConfig = config;
#if defined(ARM_MFCC_CFFT_BASED)
         memory.resize(2*inputSize);
#else
         memory.resize(inputSize + 2);
#endif
    };

    int run(){
        float16_t *a=this->getReadBuffer();
        float16_t *b=this->getWriteBuffer();
        arm_mfcc_f16(mfccConfig,a,b,memory.data());
        return(0);
    };

    const arm_mfcc_instance_f16 *mfccConfig;
    std::vector<float16_t> memory;
};
#endif 

/*

The CMSIS-DSP MFCC Q31

*/
template<int inputSize,int outputSize>
class MFCC<q31_t,inputSize,q31_t,outputSize>: public GenericNode<q31_t,inputSize,q31_t,outputSize>
{
public:
    MFCC(FIFOBase<q31_t> &src,FIFOBase<q31_t> &dst,const arm_mfcc_instance_q31 *config):GenericNode<q31_t,inputSize,q31_t,outputSize>(src,dst){
         mfccConfig = config;
         memory.resize(2*inputSize);
    };

    int run(){
        q31_t *a=this->getReadBuffer();
        q31_t *b=this->getWriteBuffer();
        arm_mfcc_q31(mfccConfig,a,b,memory.data());
        return(0);
    };

    const arm_mfcc_instance_q31 *mfccConfig;
    std::vector<q31_t> memory;
};

/*

The CMSIS-DSP MFCC Q15

*/
template<int inputSize,int outputSize>
class MFCC<q15_t,inputSize,q15_t,outputSize>: public GenericNode<q15_t,inputSize,q15_t,outputSize>
{
public:
    MFCC(FIFOBase<q15_t> &src,FIFOBase<q15_t> &dst,const arm_mfcc_instance_q15 *config):GenericNode<q15_t,inputSize,q15_t,outputSize>(src,dst){
         mfccConfig = config;
         memory.resize(2*inputSize);
    };

    int run(){
        q15_t *a=this->getReadBuffer();
        q15_t *b=this->getWriteBuffer();
        arm_mfcc_q15(mfccConfig,a,b,memory.data());
        return(0);
    };

    const arm_mfcc_instance_q15 *mfccConfig;
    std::vector<q31_t> memory;
};


#endif