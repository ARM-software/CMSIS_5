/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        ToComplex.h
 * Description:  Node to convert real to complex 
 *
 * $Date:        30 July 2021
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
#ifndef _TOCOMPLEX_H_
#define _TOCOMPLEX_H_

/*

Convert a stream of reals a b c d ...
to complexes a 0 b 0 c 0 d 0 ...
*/

template<typename IN, int inputSize,typename OUT,int outputSize>
class ToComplex;

template<typename IN, int inputSize,int outputSize>
class ToComplex<IN,inputSize,IN,outputSize>: public GenericNode<IN,inputSize,IN,outputSize>
{
public:
    ToComplex(FIFOBase<IN> &src,FIFOBase<IN> &dst):GenericNode<IN,inputSize,IN,outputSize>(src,dst){
    };

    int run(){
        IN *a=this->getReadBuffer();
        IN *b=this->getWriteBuffer();
        for(int i=0;i<inputSize;i++)
        {
            b[2*i]=a[i];
            b[2*i+1]=0;
        }
        return(0);
    };


};

#endif 
