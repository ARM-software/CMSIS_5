
/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_boolean_distance.c
 * Description:  Templates for boolean distances
 *
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




/**
 * @defgroup DISTANCEF Distance Functions
 *
 * Computes Distances between vectors. 
 *
 * Distance functions are useful in a lot of algorithms.
 *
 */


/**
 * @addtogroup DISTANCEF
 * @{
 */


/**
 * @brief        Elements of boolean distances
 *
 * Different values which are used to compute boolean distances
 *
 * @param[in]    pA              First vector of packed booleans
 * @param[in]    pB              Second vector of packed booleans
 * @param[in]    numberOfBools   Number of booleans
 * @param[out]   cTT             cTT value
 * @param[out]   cTF             cTF value
 * @param[out]   cFT             cFT value
 * @return None
 *
 */

#define _FUNC(A,B) A##B 

#define FUNC(EXT) _FUNC(arm_boolean_distance, EXT)

#if defined(ARM_MATH_NEON)


void FUNC(EXT)(const uint32_t *pA
       , const uint32_t *pB
       , uint32_t numberOfBools
#ifdef TT
       , uint32_t *cTT
#endif
#ifdef FF
       , uint32_t *cFF
#endif
#ifdef TF
       , uint32_t *cTF
#endif
#ifdef FT
       , uint32_t *cFT
#endif
       )
{
#ifdef TT
    uint32_t _ctt=0;
#endif
#ifdef FF
    uint32_t _cff=0;
#endif
#ifdef TF
    uint32_t _ctf=0;
#endif
#ifdef FT
    uint32_t _cft=0;
#endif
    uint32_t nbBoolBlock;
    uint32_t a,b,ba,bb,cc=1;
    int shift;
    uint32x4_t aV, bV;
#ifdef TT
    uint32x4_t cttV;
#endif
#ifdef FF
    uint32x4_t cffV;
#endif
#ifdef TF
    uint32x4_t ctfV;
#endif
#ifdef FT
    uint32x4_t cftV;
#endif
    uint8x16_t tmp;
    uint16x8_t tmp2;
    uint32x4_t tmp3;
    uint64x2_t tmp4;
#ifdef TT
    uint64x2_t tmp4tt;
#endif
#ifdef FF
    uint64x2_t tmp4ff;
#endif
#ifdef TF
    uint64x2_t tmp4tf;
#endif
#ifdef FT
    uint64x2_t tmp4ft;
#endif

#ifdef TT
    tmp4tt = vdupq_n_u64(0);
#endif
#ifdef FF
    tmp4ff = vdupq_n_u64(0);
#endif
#ifdef TF
    tmp4tf = vdupq_n_u64(0);
#endif
#ifdef FT
    tmp4ft = vdupq_n_u64(0);
#endif

    nbBoolBlock = numberOfBools >> 7;
    while(nbBoolBlock > 0)
    {
       aV = vld1q_u32(pA);
       bV = vld1q_u32(pB);
       pA += 4;
       pB += 4;

#ifdef TT
       cttV = vandq_u32(aV,bV);
#endif
#ifdef FF
       cffV = vandq_u32(vmvnq_u32(aV),vmvnq_u32(bV));
#endif
#ifdef TF
       ctfV = vandq_u32(aV,vmvnq_u32(bV));
#endif
#ifdef FT
       cftV = vandq_u32(vmvnq_u32(aV),bV);
#endif

#ifdef TT
       tmp = vcntq_u8(vreinterpretq_u8_u32(cttV));
       tmp2 = vpaddlq_u8(tmp);
       tmp3 = vpaddlq_u16(tmp2);
       tmp4 = vpaddlq_u32(tmp3);
       tmp4tt = vaddq_u64(tmp4tt, tmp4);
#endif

#ifdef FF
       tmp = vcntq_u8(vreinterpretq_u8_u32(cffV));
       tmp2 = vpaddlq_u8(tmp);
       tmp3 = vpaddlq_u16(tmp2);
       tmp4 = vpaddlq_u32(tmp3);
       tmp4ff = vaddq_u64(tmp4ff, tmp4);
#endif

#ifdef TF
       tmp = vcntq_u8(vreinterpretq_u8_u32(ctfV));
       tmp2 = vpaddlq_u8(tmp);
       tmp3 = vpaddlq_u16(tmp2);
       tmp4 = vpaddlq_u32(tmp3);
       tmp4tf = vaddq_u64(tmp4tf, tmp4);
#endif 

#ifdef FT
       tmp = vcntq_u8(vreinterpretq_u8_u32(cftV));
       tmp2 = vpaddlq_u8(tmp);
       tmp3 = vpaddlq_u16(tmp2);
       tmp4 = vpaddlq_u32(tmp3);
       tmp4ft = vaddq_u64(tmp4ft, tmp4);
#endif


       nbBoolBlock --;
    }

#ifdef TT
    _ctt += tmp4tt[0] + tmp4tt[1];
#endif
#ifdef FF
    _cff += tmp4ff[0] + tmp4ff[1];
#endif
#ifdef TF
    _ctf += tmp4tf[0] + tmp4tf[1];
#endif
#ifdef FT
    _cft += tmp4ft[0] + tmp4ft[1];
#endif

    nbBoolBlock = numberOfBools & 0x7F;
    while(nbBoolBlock >= 32)
    {
       a = *pA++;
       b = *pB++;
       shift = 0;
       while(shift < 32)
       {
          ba = a & 1;
          bb = b & 1;
          a = a >> 1;
          b = b >> 1;

#ifdef TT
          _ctt += (ba && bb);
#endif
#ifdef FF
          _cff += ((1 ^ ba) && (1 ^ bb));
#endif
#ifdef TF
          _ctf += (ba && (1 ^ bb));
#endif
#ifdef FT
          _cft += ((1 ^ ba) && bb);
#endif
          shift ++;
       }

       nbBoolBlock -= 32;
    }

    a = *pA++;
    b = *pB++;

    a = a >> (32 - nbBoolBlock);
    b = b >> (32 - nbBoolBlock);

    while(nbBoolBlock > 0)
    {
          ba = a & 1;
          bb = b & 1;
          a = a >> 1;

          b = b >> 1;
#ifdef TT
          _ctt += (ba && bb);
#endif
#ifdef FF
          _cff += ((1 ^ ba) && (1 ^ bb));
#endif
#ifdef TF
          _ctf += (ba && (1 ^ bb));
#endif
#ifdef FT
          _cft += ((1 ^ ba) && bb);
#endif
          nbBoolBlock --;
    }

#ifdef TT
    *cTT = _ctt;
#endif
#ifdef FF
    *cFF = _cff;
#endif
#ifdef TF
    *cTF = _ctf;
#endif
#ifdef FT
    *cFT = _cft;
#endif
}

#else

void FUNC(EXT)(const uint32_t *pA
       , const uint32_t *pB
       , uint32_t numberOfBools
#ifdef TT
       , uint32_t *cTT
#endif
#ifdef FF
       , uint32_t *cFF
#endif
#ifdef TF
       , uint32_t *cTF
#endif
#ifdef FT
       , uint32_t *cFT
#endif
       )
{
#ifdef TT
    uint32_t _ctt=0;
#endif
#ifdef FF
    uint32_t _cff=0;
#endif
#ifdef TF
    uint32_t _ctf=0;
#endif
#ifdef FT
    uint32_t _cft=0;
#endif
    uint32_t a,b,ba,bb,cc=1;
    int shift;

    while(numberOfBools >= 32)
    {
       a = *pA++;
       b = *pB++;
       shift = 0;
       while(shift < 32)
       {
          ba = a & 1;
          bb = b & 1;
          a = a >> 1;
          b = b >> 1;
#ifdef TT
          _ctt += (ba && bb);
#endif
#ifdef FF
          _cff += ((1 ^ ba) && (1 ^ bb));
#endif
#ifdef TF
          _ctf += (ba && (1 ^ bb));
#endif
#ifdef FT
          _cft += ((1 ^ ba) && bb);
#endif
          shift ++;
       }

       numberOfBools -= 32;
    }

    a = *pA++;
    b = *pB++;

    a = a >> (32 - numberOfBools);
    b = b >> (32 - numberOfBools);

    while(numberOfBools > 0)
    {
          ba = a & 1;
          bb = b & 1;
          a = a >> 1;
          b = b >> 1;

#ifdef TT
          _ctt += (ba && bb);
#endif
#ifdef FF
          _cff += ((1 ^ ba) && (1 ^ bb));
#endif
#ifdef TF
          _ctf += (ba && (1 ^ bb));
#endif
#ifdef FT
          _cft += ((1 ^ ba) && bb);
#endif
          numberOfBools --;
    }

#ifdef TT
    *cTT = _ctt;
#endif
#ifdef FF
    *cFF = _cff;
#endif
#ifdef TF
    *cTF = _ctf;
#endif
#ifdef FT
    *cFT = _cft;
#endif
}
#endif


/**
 * @} end of DISTANCEF group
 */
