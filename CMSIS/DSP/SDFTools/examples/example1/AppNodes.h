/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        AppNodes.h
 * Description:  Application nodes for Example 1
 *
 * $Date:        29 July 2021
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
#ifndef _APPNODES_H_
#define _APPNODES_H_

#include <iostream>

template<typename IN, int inputSize>
class Sink: public GenericSink<IN, inputSize>
{
public:
    Sink(FIFOBase<IN> &src):GenericSink<IN,inputSize>(src){};

    int run()
    {
        IN *b=this->getReadBuffer();
        printf("Sink\n");
        for(int i=0;i<inputSize;i++)
        {
            std::cout << (int)b[i] << std::endl;
        }
        return(0);
    };

};

template<typename OUT,int outputSize>
class Source: GenericSource<OUT,outputSize>
{
public:
    Source(FIFOBase<OUT> &dst):GenericSource<OUT,outputSize>(dst),mCounter(0){};

    int run(){
        OUT *b=this->getWriteBuffer();

        printf("Source\n");
        for(int i=0;i<outputSize;i++)
        {
            b[i] = (OUT)mCounter++;
        }
        return(0);
    };

    int mCounter;

};

template<typename IN, int inputSize,typename OUT,int outputSize>
class ProcessingNode: public GenericNode<IN,inputSize,OUT,outputSize>
{
public:
    ProcessingNode(FIFOBase<IN> &src,FIFOBase<OUT> &dst,int,const char*,int):GenericNode<IN,inputSize,OUT,outputSize>(src,dst){};

    int run(){
        printf("ProcessingNode\n");
        IN *a=this->getReadBuffer();
        OUT *b=this->getWriteBuffer();
        b[0] =(OUT)a[3];
        return(0);
    };

};

#endif