/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        ArrayMemory.cpp
 * Description:  Array Memory Manager
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
#include "ArrayMemory.h"
#include <cstdlib>
#include <cstring>
#include <math.h>

namespace Client {
     ArrayMemory::ArrayMemory(char* ptr, size_t bufferLength,int aligned, bool tail)
     {
         this->m_ptr=ptr;
         this->m_currentPtr=ptr;
         this->alignSize = aligned;
         this->tail=tail;
         this->m_bufferLength = bufferLength;
         this->m_generation=0;
         this->memError=false;
         #if !defined(BENCHMARK)
         memset((void*)ptr, 0, bufferLength);
         #endif
     }

     // By default there is alignment and  tail
     ArrayMemory::ArrayMemory(char* ptr, size_t bufferLength)
     {  
         this->m_ptr=ptr;
         this->m_currentPtr=ptr;
         // Align on 64 bits per default
         this->alignSize = 8;
         this->tail=true;
         this->m_bufferLength = bufferLength;
         this->m_generation=0;
         this->memError=false;
         #if !defined(BENCHMARK)
         memset((void*)ptr, 0, bufferLength);
         #endif
        }
     
     bool ArrayMemory::HasMemError()
     {
         return(this->memError);
     }

     size_t ArrayMemory::getTailSize()
     {
        if (this->tail)
        {
           return(16);  
        }
        else
        {
            return(0);
        }
     }

     char *ArrayMemory::NewBuffer(size_t length)
     {
         if (length == 0)
         {
            return(NULL);
         }
         
         size_t tailSize = 0;
         // Add a tail of 16 bytes corresponding to the max number of lanes.
         tailSize = this->getTailSize();  
         
         // Compute some offset to align the new buffer to be allocated
         if (this->alignSize > 0)
         {
            unsigned long offset;
            unsigned long pad;

            offset=(unsigned long)(this->m_currentPtr - this->m_ptr);
            pad = this->alignSize*ceil(1.0*offset / (1.0*this->alignSize)) - offset;
            //printf("new  = %ld, old = %ld\n",pad,offset);
            this->m_currentPtr += pad;
         }

         // Return NULL is no more enough memory in array
         if (this->m_currentPtr + length + tailSize < this->m_ptr + m_bufferLength)
         {
            char *result=this->m_currentPtr;
            this->m_currentPtr += length + tailSize;

            return(result);
         }
         else
        {
            this->memError=true;
            return(NULL);
        }
     }

     bool ArrayMemory::IsTailEmpty(char *ptr, size_t length)
     {
        if ((ptr == NULL) || (length == 0))
        {
           return(true);
        }
        else
        {
            char *p=ptr + length;
            bool isEmpty=true;
    
            for(unsigned long i=0; i < this->getTailSize() ; i++)
            {
                //printf("%d\n",p[i]);
                if (p[i] != 0)
                {
                    isEmpty = false;
                }
            }
            return(isEmpty);
        }
     }

     
    /** Reset memory

        The full C buffer is set to 0
        Current pointer is moved to start of buffer
        Memory generation is incremented (which is
        indirectly unvalidating all patterns.
        If the patterns are not reloaded after this, they'll return NULL
        when trying to access their pointer.
        )

    */
    void ArrayMemory::FreeMemory()
    {
        #if !defined(BENCHMARK)
           /*
            In benchmark mode, memory is not clearer between
            tests. It is faster when running on cycle model or RTL.
            In benchmark mode, we don't tests so having a memory not
            in a clean state is not a problem.
           */
           memset(this->m_ptr, 0, this->m_bufferLength);
        #endif
        this->m_currentPtr=this->m_ptr;
        this->m_generation++;
        this->memError=false;

    }
}
