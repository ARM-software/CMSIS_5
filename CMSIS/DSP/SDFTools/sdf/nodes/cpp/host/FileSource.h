/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        FileSource.h
 * Description:  Node for creating File sources
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
#ifndef _FILESOURCE_H_
#define _FILESOURCE_H_

template<typename OUT,int outputSize> class FileSource;

/* 

Real a list of floats from a file and pad with zeros indefinitely when end of
file is reached.

*/
template<int outputSize>
class FileSource<float32_t,outputSize>: public GenericSource<float32_t,outputSize>
{
public:
    FileSource(FIFOBase<float32_t> &dst,std::string name):GenericSource<float32_t,outputSize>(dst),
    input(name)
    {

    };

    int run(){
        string str;
        int i;
        float32_t *b=this->getWriteBuffer();

        if (input.eof())
        {
            for(i=0;i<outputSize;i++)
            {
                b[i] = 0;
            }

        }
        else
        {
             for(i=0;i<outputSize;i++)
             {
                 if (!getline(input, str))
                 {
                     b[i] = 0;
                     break;
                 }
                 b[i] = (float)atof(str.c_str());
             }
     
             for(;i<outputSize;i++)
             {
                 b[i] = 0;
             }
        }
        return(0);
    };


    ifstream input;

};

#endif
