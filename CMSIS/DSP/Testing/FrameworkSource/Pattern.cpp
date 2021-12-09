/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Pattern.cpp
 * Description:  Patterns
 *
 *               Abstraction to manipulate test patterns
 *               and hiding where they come from
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
#include "Pattern.h"
#include "arm_math_types.h"
#include "arm_math_types_f16.h"

namespace Client {

template <> 
float64_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_f64(id,nb,maxSamples));
}

template <> 
float32_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_f32(id,nb,maxSamples));
}

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
template <> 
float16_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_f16(id,nb,maxSamples));
}
#endif

template <> 
q63_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_q63(id,nb,maxSamples));
}

template <> 
q31_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_q31(id,nb,maxSamples));
}

template <> 
q15_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_q15(id,nb,maxSamples));
}

template <> 
q7_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_q7(id,nb,maxSamples));
}

template <> 
uint32_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_u32(id,nb,maxSamples));
}

template <> 
uint16_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_u16(id,nb,maxSamples));
}

template <> 
uint8_t *loadPattern(Testing::PatternID_t id, Client::PatternMgr *mgr,Testing::nbSamples_t &nb, Testing::nbSamples_t maxSamples)
{
    return(mgr->load_u8(id,nb,maxSamples));
}


template <> 
float64_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_f64(id));
}

template <> 
float32_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_f32(id));
}

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
template <> 
float16_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_f16(id));
}
#endif

template <> 
q63_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_q63(id));
}

template <> 
q31_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_q31(id));
}

template <> 
q15_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_q15(id));
}

template <> 
q7_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_q7(id));
}

template <> 
uint32_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_u32(id));
}

template <> 
uint16_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_u16(id));
}

template <> 
uint8_t *localPattern(Testing::PatternID_t id, Client::PatternMgr *mgr)
{
    return(mgr->local_u8(id));
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,float64_t* data,PatternMgr *mgr)
{
   mgr->dumpPattern_f64(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,float32_t* data,PatternMgr *mgr)
{
  mgr->dumpPattern_f32(id,nbSamples,data);
}

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,float16_t* data,PatternMgr *mgr)
{
  mgr->dumpPattern_f16(id,nbSamples,data);
}
#endif

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q63_t* data,PatternMgr *mgr)
{
  mgr->dumpPattern_q63(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q31_t* data,PatternMgr *mgr)
{
  mgr->dumpPattern_q31(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q15_t* data,PatternMgr *mgr)
{
 mgr->dumpPattern_q15(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,q7_t* data,PatternMgr *mgr)
{
 mgr->dumpPattern_q7(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,uint32_t* data,PatternMgr *mgr)
{
  mgr->dumpPattern_u32(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,uint16_t* data,PatternMgr *mgr)
{
 mgr->dumpPattern_u16(id,nbSamples,data);
}

void dumpPattern(Testing::outputID_t id,Testing::nbSamples_t nbSamples,uint8_t* data,PatternMgr *mgr)
{
 mgr->dumpPattern_u8(id,nbSamples,data);
}


}
