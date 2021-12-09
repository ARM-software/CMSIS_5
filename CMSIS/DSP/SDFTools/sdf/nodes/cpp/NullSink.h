/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        NullSink.h
 * Description:  Sink doing nothing for debug
 *
 * $Date:        08 August 2021
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
#ifndef _NULLSINK_H_
#define _NULLSINK_H_

/* Write a list of samples to a file in text form */
template<typename IN, int inputSize>
class NullSink: public GenericSink<IN, inputSize>
{
public:
    NullSink(FIFOBase<IN> &src):GenericSink<IN,inputSize>(src){};

    int run()
    {
        IN *b=this->getReadBuffer();

        

        return(0);
    };

};

#endif