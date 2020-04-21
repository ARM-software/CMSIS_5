/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        PatternMgr.cpp
 * Description:  Pattern Manager
 *
 *               The link between a pattern and a memory manager.
 *               Allow creation and initialization of patterns
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
#include "Test.h"
#include "arm_math.h"
#include "arm_math_f16.h"

namespace Client
{
PatternMgr::PatternMgr(IO* io, Memory *mem)
{
   m_io = io;
   m_mem = mem;
}

#define LOCAL(TYPE,EXT) \
TYPE *PatternMgr::local_##EXT(Testing::nbSamples_t nbSamples) \
{ \
    return((TYPE*)(m_mem->NewBuffer(sizeof(TYPE)*nbSamples))); \
}

LOCAL(float64_t,f64)
LOCAL(float32_t,f32)
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
LOCAL(float16_t,f16)
#endif
LOCAL(q63_t,q63)
LOCAL(q31_t,q31)
LOCAL(q15_t,q15)
LOCAL(q7_t,q7)
LOCAL(uint32_t,u32)
LOCAL(uint16_t,u16)
LOCAL(uint8_t,u8)

float64_t *PatternMgr::load_f64(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    
    nbSamples=m_io->GetPatternSize(id);
    
    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }
  
    char *b = m_mem->NewBuffer(sizeof(float64_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_f64(id,b,nbSamples);
    }
    return((float64_t*)b);
   
}

float32_t *PatternMgr::load_f32(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);
  
    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(float32_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_f32(id,b,nbSamples);
    }
    return((float32_t*)b);
   
}

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
float16_t *PatternMgr::load_f16(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);
  
    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(float16_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_f16(id,b,nbSamples);
    }
    return((float16_t*)b);
   
}
#endif

q63_t *PatternMgr::load_q63(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(q63_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_q63(id,b,nbSamples);
    }
    return((q63_t*)b);
}


q31_t *PatternMgr::load_q31(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(q31_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_q31(id,b,nbSamples);
    }
    return((q31_t*)b);
}

q15_t *PatternMgr::load_q15(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(q15_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_q15(id,b,nbSamples);
    }
    return((q15_t*)b);
}

q7_t *PatternMgr::load_q7(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(q7_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_q7(id,b,nbSamples);
    }
    return((q7_t*)b);
}

uint32_t *PatternMgr::load_u32(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(uint32_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_u32(id,b,nbSamples);
    }
    return((uint32_t*)b);
}

uint16_t *PatternMgr::load_u16(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(uint16_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_u16(id,b,nbSamples);
    }
    return((uint16_t*)b);
}

uint8_t *PatternMgr::load_u8(Testing::PatternID_t id,Testing::nbSamples_t& nbSamples,Testing::nbSamples_t maxSamples)
{
    nbSamples=m_io->GetPatternSize(id);

    if ((maxSamples != MAX_NB_SAMPLES) && (maxSamples < nbSamples))
    {
        nbSamples = maxSamples;
    }

    char *b = m_mem->NewBuffer(sizeof(uint8_t)*nbSamples);
    if (b != NULL)
    {
       m_io->ImportPattern_u8(id,b,nbSamples);
    }
    return((uint8_t*)b);
}

void PatternMgr::dumpPattern_f64(Testing::outputID_t id,Testing::nbSamples_t nbSamples,float64_t* data)
{
    
    m_io->DumpPattern_f64(id,nbSamples,data);
}

void PatternMgr::dumpPattern_f32(Testing::outputID_t id,Testing::nbSamples_t nbSamples,float32_t* data)
{
   m_io->DumpPattern_f32(id,nbSamples,data);
}

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
void PatternMgr::dumpPattern_f16(Testing::outputID_t id,Testing::nbSamples_t nbSamples,float16_t* data)
{
   m_io->DumpPattern_f16(id,nbSamples,data);
}
#endif 

void PatternMgr::dumpPattern_q63(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q63_t* data)
{
   m_io->DumpPattern_q63(id,nbSamples,data);
}

void PatternMgr::dumpPattern_q31(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q31_t* data)
{
   m_io->DumpPattern_q31(id,nbSamples,data);
}

void PatternMgr::dumpPattern_q15(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q15_t* data)
{
   m_io->DumpPattern_q15(id,nbSamples,data);
}

void PatternMgr::dumpPattern_q7(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q7_t* data)
{
  m_io->DumpPattern_q7(id,nbSamples,data);
}

void PatternMgr::dumpPattern_u32(Testing::outputID_t id,Testing::nbSamples_t nbSamples,uint32_t* data)
{
 m_io->DumpPattern_u32(id,nbSamples,data);
}

void PatternMgr::dumpPattern_u16(Testing::outputID_t id,Testing::nbSamples_t nbSamples,uint16_t* data)
{
 m_io->DumpPattern_u16(id,nbSamples,data);
}

void PatternMgr::dumpPattern_u8(Testing::outputID_t id,Testing::nbSamples_t nbSamples,uint8_t* data)
{
 m_io->DumpPattern_u8(id,nbSamples,data);
}

void PatternMgr::freeAll()
{
    m_mem->FreeMemory();
}

}
