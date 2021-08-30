/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        StereoToMonoQ15.h
 * Description:  Stereo to mno stream in Q15
 *
 * $Date:        06 August 2021
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

#ifndef _STEREOTOMONOQ15_H_
#define _STEREOTOMONOQ15_H_ 


template<typename IN, int inputSize,typename OUT,int outputSize>
class StereoToMonoQ15;

template<int inputSize,int outputSize>
class StereoToMonoQ15<q15_t,inputSize,q15_t,outputSize>: public GenericNode<q15_t,inputSize,q15_t,outputSize>
{
public:
    StereoToMonoQ15(FIFOBase<q15_t> &src,FIFOBase<q15_t> &dst):
    GenericNode<q15_t,inputSize,q15_t,outputSize>(src,dst){};

   
    int run(){
        q15_t *a=this->getReadBuffer();
        q15_t *b=this->getWriteBuffer();
        for(int i = 0; i<outputSize; i++)
        {
           b[i] = (a[2*i]>>1) + (a[2*i+1]>>1);
        }
        return(0);
    };

};

#endif