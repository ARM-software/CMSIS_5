/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_init_f32.c
 * Description:  IIR filter initialization function
 *
 * $Date:        2019
 * $Revision:    
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

#include "arm_math.h"

void arm_iir_1st_init_f32(arm_iir_instance_f32 * S, arm_iir_type iirType, uint32_t nbCascaded, uint8_t simdFlag, float32_t * pState, float32_t * pCoeffs );
void arm_iir_2nd_init_f32(arm_iir_instance_f32 * S, arm_iir_type iirType, uint32_t nbCascaded, uint8_t simdFlag, float32_t * pState, float32_t * pCoeffs );
void arm_iir_3rd_init_f32(arm_iir_instance_f32 * S, arm_iir_type iirType, uint32_t nbCascaded, uint8_t simdFlag, float32_t * pState, float32_t * pCoeffs );
void arm_iir_generic_init_f32(arm_iir_instance_f32 * S, arm_iir_type iirType, uint32_t nbCascaded, uint8_t simdFlag, uint16_t order, float32_t * pState, float32_t * pCoeffs );

/**
  @ingroup groupFilters
 */

/**
  @addtogroup IIRs
  @{
 */

/**
 * @brief  Initialization function for the floating-point IIR filter.
 * @param[in,out] S            points to an instance of the floating-point IIR filter.
 * @param[in]     iirType      type of IIR filter.
 * @param[in]     order        order of IIR filter.
 * @param[in]     nbCascaded   number of stages
 * @param[in]     simdFlag     flag for simd version
 * @param[in]     debugFlag    flag for debug
 * @param[in]     pState       points to the state values
 * @param[in]     pCoeffs      points to the coefficients (b0 ... bk a1 ... ak); (b0 ... bk a1 ... ak) ...
 */

void arm_iir_init_f32(
    arm_iir_instance_f32 * S,
    arm_iir_type iirType,
    uint16_t order,
    uint32_t nbCascaded,
    uint8_t simdFlag,
    uint8_t debugFlag,
    float32_t * pState,
    float32_t * pCoeffs ) 
{

    switch(order)
    {
        case 1:
        arm_iir_1st_init_f32(S, iirType, nbCascaded, simdFlag, pState, pCoeffs);
        break;
        
        case 2:
        arm_iir_2nd_init_f32(S, iirType, nbCascaded, simdFlag, pState, pCoeffs);
        break;
        
        case 3:
        arm_iir_3rd_init_f32(S, iirType, nbCascaded, simdFlag, pState, pCoeffs);
        break;
        
        default:
        arm_iir_generic_init_f32(S, iirType, nbCascaded, simdFlag, order, pState, pCoeffs);
        break;
    }

    S->iirType   = iirType;
    S->order     = order;
    S->numStages = nbCascaded;
    S->simdFlag  = simdFlag;
    S->debugFlag = debugFlag;
}

/**
  @} end of IIRs group
 */
