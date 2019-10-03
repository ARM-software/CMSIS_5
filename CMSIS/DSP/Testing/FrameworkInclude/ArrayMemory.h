/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        ArrayMemory.h
 * Description:  Array Memory Header
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

#ifndef _ARRAY_MEMORY_H_
#define _ARRAY_MEMORY_H_

#include "Test.h"

namespace Client{

// Memory is implemented as a big buffer in which
// we reserve blocks.
// Like that we can manage alignment and tails
class ArrayMemory:public Client::Memory
{
   public:
    ArrayMemory(char* ptr, size_t bufferLength,int aligned, bool tail);
    ArrayMemory(char* ptr, size_t bufferLength);
    virtual char *NewBuffer(size_t length);
    virtual void FreeMemory();
    virtual bool HasMemError();
    virtual bool IsTailEmpty(char *, size_t);
   
   private:
     // Pointer to C array used for memory
     char *m_ptr;
     // Size of C array buffer
     size_t m_bufferLength;
     // Alignement required for all buffers
     // (in future may be a setting per bufer)
     int alignSize;
     // True if some padding must be added after buffers
     bool tail=true;
     // Current pointer to the memory 
     // It is where a new buffer will be allocated
     char *m_currentPtr;
     // Error occured
     bool memError=false;

     size_t getTailSize();
};
}

#endif
