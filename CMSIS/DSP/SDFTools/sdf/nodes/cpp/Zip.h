/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Zip.h
 * Description:  Node to zip a pair of stream
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
#ifndef _ZIP_H_
#define _ZIP_H_ 


template<typename IN1, int inputSize1,typename IN2,int inputSize2,typename OUT,int outputSize>
class Zip;

template<typename IN, int inputSize,int outputSize>
class Zip<IN,inputSize,IN,inputSize,IN,outputSize>: public GenericNode21<IN,inputSize,IN,inputSize,IN,outputSize>
{
public:
    Zip(FIFOBase<IN> &src1,FIFOBase<IN> &src2,FIFOBase<IN> &dst):
    GenericNode21<IN,inputSize,IN,inputSize,IN,outputSize>(src1,src2,dst){};


    int run(){
        IN *a1=this->getReadBuffer1();
        IN *a2=this->getReadBuffer2();
        IN *b=this->getWriteBuffer1();
        for(int i = 0; i<inputSize; i++)
        {
           b[2*i] = a1[i];
           b[2*i+1] = a2[i];
        }
        return(0);
    };

};

#endif