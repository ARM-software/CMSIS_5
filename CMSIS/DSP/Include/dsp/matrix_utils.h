/******************************************************************************
 * @file     matrix_utils.h
 * @brief    Public header file for CMSIS DSP Library
 * @version  V1.11.0
 * @date     30 May 2022
 * Target Processor: Cortex-M and Cortex-A cores
 ******************************************************************************/
/*
 * Copyright (c) 2010-2022 Arm Limited or its affiliates. All rights reserved.
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

 
#ifndef _MATRIX_UTILS_H_
#define _MATRIX_UTILS_H_

#include "arm_math_types.h"
#include "arm_math_memory.h"

#include "dsp/none.h"
#include "dsp/utils.h"

#ifdef   __cplusplus
extern "C"
{
#endif

#define ELEM(A,ROW,COL) &((A)->pData[(A)->numCols* (ROW) + (COL)])

#define SCALE_COL_T(T,CAST,A,ROW,v,i)        \
{                                       \
  int32_t w;                            \
  T *data = (A)->pData;                 \
  const int32_t numCols = (A)->numCols; \
  const int32_t nb = (A)->numRows - ROW;\
                                        \
  data += i + numCols * (ROW);          \
                                        \
  for(w=0;w < nb; w++)                  \
  {                                     \
     *data *= CAST v;                   \
     data += numCols;                   \
  }                                     \
}

#if defined(ARM_FLOAT16_SUPPORTED)
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

#define SWAP_ROWS_F16(A,COL,i,j)                  \
  {                                               \
    int cnt = ((A)->numCols)-(COL);               \
    int32_t w;                                   \
    float16_t *data = (A)->pData;                 \
    const int32_t numCols = (A)->numCols;        \
                                                  \
    for(w=(COL);w < numCols; w+=8)                \
    {                                             \
       f16x8_t tmpa,tmpb;                         \
       mve_pred16_t p0 = vctp16q(cnt);            \
                                                  \
       tmpa=vldrhq_z_f16(&data[i*numCols + w],p0);\
       tmpb=vldrhq_z_f16(&data[j*numCols + w],p0);\
                                                  \
       vstrhq_p(&data[i*numCols + w], tmpb, p0);  \
       vstrhq_p(&data[j*numCols + w], tmpa, p0);  \
                                                  \
       cnt -= 8;                                  \
    }                                             \
  }

#define SCALE_ROW_F16(A,COL,v,i)                   \
{                                                   \
  int cnt = ((A)->numCols)-(COL);                   \
  int32_t w;                                       \
  float16_t *data = (A)->pData;                     \
  const int32_t numCols = (A)->numCols;            \
                                                    \
  for(w=(COL);w < numCols; w+=8)                    \
  {                                                 \
       f16x8_t tmpa;                                \
       mve_pred16_t p0 = vctp16q(cnt);              \
       tmpa = vldrhq_z_f16(&data[i*numCols + w],p0);\
       tmpa = vmulq_n_f16(tmpa,(_Float16)v);                  \
       vstrhq_p(&data[i*numCols + w], tmpa, p0);    \
       cnt -= 8;                                    \
  }                                                 \
                                                    \
}

#define MAC_ROW_F16(COL,A,i,v,B,j)                   \
{                                                    \
  int cnt = ((A)->numCols)-(COL);                    \
  int32_t w;                                        \
  float16_t *dataA = (A)->pData;                     \
  float16_t *dataB = (B)->pData;                     \
  const int32_t numCols = (A)->numCols;             \
                                                     \
  for(w=(COL);w < numCols; w+=8)                     \
  {                                                  \
       f16x8_t tmpa,tmpb;                            \
       mve_pred16_t p0 = vctp16q(cnt);               \
       tmpa = vldrhq_z_f16(&dataA[i*numCols + w],p0);\
       tmpb = vldrhq_z_f16(&dataB[j*numCols + w],p0);\
       tmpa = vfmaq_n_f16(tmpa,tmpb,v);              \
       vstrhq_p(&dataA[i*numCols + w], tmpa, p0);    \
       cnt -= 8;                                     \
  }                                                  \
                                                     \
}

#define MAS_ROW_F16(COL,A,i,v,B,j)                   \
{                                                    \
  int cnt = ((A)->numCols)-(COL);                    \
  int32_t w;                                        \
  float16_t *dataA = (A)->pData;                     \
  float16_t *dataB = (B)->pData;                     \
  const int32_t numCols = (A)->numCols;             \
  f16x8_t vec=vdupq_n_f16(v);                        \
                                                     \
  for(w=(COL);w < numCols; w+=8)                     \
  {                                                  \
       f16x8_t tmpa,tmpb;                            \
       mve_pred16_t p0 = vctp16q(cnt);               \
       tmpa = vldrhq_z_f16(&dataA[i*numCols + w],p0);\
       tmpb = vldrhq_z_f16(&dataB[j*numCols + w],p0);\
       tmpa = vfmsq_f16(tmpa,tmpb,vec);              \
       vstrhq_p(&dataA[i*numCols + w], tmpa, p0);    \
       cnt -= 8;                                     \
  }                                                  \
                                                     \
}

#else

#define SWAP_ROWS_F16(A,COL,i,j)       \
{                                      \
  int32_t w;                           \
  float16_t *dataI = (A)->pData;       \
  float16_t *dataJ = (A)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);    \
                                       \
  dataI += i*numCols + (COL);          \
  dataJ += j*numCols + (COL);          \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     float16_t tmp;                    \
     tmp = *dataI;                     \
     *dataI++ = *dataJ;                \
     *dataJ++ = tmp;                   \
  }                                    \
}

#define SCALE_ROW_F16(A,COL,v,i)       \
{                                      \
  int32_t w;                           \
  float16_t *data = (A)->pData;        \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);    \
                                       \
  data += i*numCols + (COL);           \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     *data++ *= (_Float16)v;           \
  }                                    \
}


#define MAC_ROW_F16(COL,A,i,v,B,j)                \
{                                                 \
  int32_t w;                                      \
  float16_t *dataA = (A)->pData;                  \
  float16_t *dataB = (B)->pData;                  \
  const int32_t numCols = (A)->numCols;           \
  const int32_t nb = numCols-(COL);               \
                                                  \
  dataA += i*numCols + (COL);                     \
  dataB += j*numCols + (COL);                     \
                                                  \
  for(w=0;w < nb; w++)                            \
  {                                               \
     *dataA++ += (_Float16)v * (_Float16)*dataB++;\
  }                                               \
}

#define MAS_ROW_F16(COL,A,i,v,B,j)                \
{                                                 \
  int32_t w;                                      \
  float16_t *dataA = (A)->pData;                  \
  float16_t *dataB = (B)->pData;                  \
  const int32_t numCols = (A)->numCols;           \
  const int32_t nb = numCols-(COL);               \
                                                  \
  dataA += i*numCols + (COL);                     \
  dataB += j*numCols + (COL);                     \
                                                  \
  for(w=0;w < nb; w++)                            \
  {                                               \
     *dataA++ -= (_Float16)v * (_Float16)*dataB++;\
  }                                               \
}

#endif /*defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)*/


#define SCALE_COL_F16(A,ROW,v,i)        \
  SCALE_COL_T(float16_t,(_Float16),A,ROW,v,i)
  
#endif /* defined(ARM_FLOAT16_SUPPORTED)*/

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)

#define SWAP_ROWS_F32(A,COL,i,j)                  \
  {                                               \
    int cnt = ((A)->numCols)-(COL);               \
    float32_t *data = (A)->pData;                 \
    const int32_t numCols = (A)->numCols;        \
    int32_t w;                                   \
                                                  \
    for(w=(COL);w < numCols; w+=4)                \
    {                                             \
       f32x4_t tmpa,tmpb;                         \
       mve_pred16_t p0 = vctp32q(cnt);            \
                                                  \
       tmpa=vldrwq_z_f32(&data[i*numCols + w],p0);\
       tmpb=vldrwq_z_f32(&data[j*numCols + w],p0);\
                                                  \
       vstrwq_p(&data[i*numCols + w], tmpb, p0);  \
       vstrwq_p(&data[j*numCols + w], tmpa, p0);  \
                                                  \
       cnt -= 4;                                  \
    }                                             \
  }

#define MAC_ROW_F32(COL,A,i,v,B,j)                   \
{                                                    \
  int cnt = ((A)->numCols)-(COL);                    \
  float32_t *dataA = (A)->pData;                     \
  float32_t *dataB = (B)->pData;                     \
  const int32_t numCols = (A)->numCols;             \
  int32_t w;                                        \
                                                     \
  for(w=(COL);w < numCols; w+=4)                     \
  {                                                  \
       f32x4_t tmpa,tmpb;                            \
       mve_pred16_t p0 = vctp32q(cnt);               \
       tmpa = vldrwq_z_f32(&dataA[i*numCols + w],p0);\
       tmpb = vldrwq_z_f32(&dataB[j*numCols + w],p0);\
       tmpa = vfmaq_n_f32(tmpa,tmpb,v);              \
       vstrwq_p(&dataA[i*numCols + w], tmpa, p0);    \
       cnt -= 4;                                     \
  }                                                  \
                                                     \
}

#define MAS_ROW_F32(COL,A,i,v,B,j)                   \
{                                                    \
  int cnt = ((A)->numCols)-(COL);                    \
  float32_t *dataA = (A)->pData;                     \
  float32_t *dataB = (B)->pData;                     \
  const int32_t numCols = (A)->numCols;             \
  int32_t w;                                        \
  f32x4_t vec=vdupq_n_f32(v);                        \
                                                     \
  for(w=(COL);w < numCols; w+=4)                     \
  {                                                  \
       f32x4_t tmpa,tmpb;                            \
       mve_pred16_t p0 = vctp32q(cnt);               \
       tmpa = vldrwq_z_f32(&dataA[i*numCols + w],p0);\
       tmpb = vldrwq_z_f32(&dataB[j*numCols + w],p0);\
       tmpa = vfmsq_f32(tmpa,tmpb,vec);              \
       vstrwq_p(&dataA[i*numCols + w], tmpa, p0);    \
       cnt -= 4;                                     \
  }                                                  \
                                                     \
}

#define SCALE_ROW_F32(A,COL,v,i)                    \
{                                                   \
  int cnt = ((A)->numCols)-(COL);                   \
  float32_t *data = (A)->pData;                     \
  const int32_t numCols = (A)->numCols;            \
  int32_t w;                                       \
                                                    \
  for(w=(COL);w < numCols; w+=4)                    \
  {                                                 \
       f32x4_t tmpa;                                \
       mve_pred16_t p0 = vctp32q(cnt);              \
       tmpa = vldrwq_z_f32(&data[i*numCols + w],p0);\
       tmpa = vmulq_n_f32(tmpa,v);                  \
       vstrwq_p(&data[i*numCols + w], tmpa, p0);    \
       cnt -= 4;                                    \
  }                                                 \
                                                    \
}

#elif defined(ARM_MATH_NEON) && !defined(ARM_MATH_AUTOVECTORIZE)

#define SWAP_ROWS_F32(A,COL,i,j)       \
{                                      \
  int32_t w;                           \
  float32_t *dataI = (A)->pData;       \
  float32_t *dataJ = (A)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols - COL;    \
                                       \
  dataI += i*numCols + (COL);          \
  dataJ += j*numCols + (COL);          \
                                       \
  float32_t tmp;                       \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     tmp = *dataI;                     \
     *dataI++ = *dataJ;                \
     *dataJ++ = tmp;                   \
  }                                    \
}

#define MAC_ROW_F32(COL,A,i,v,B,j)     \
{                                      \
  float32_t *dataA = (A)->pData;       \
  float32_t *dataB = (B)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols - (COL);  \
  int32_t nbElems;                     \
  f32x4_t vec = vdupq_n_f32(v);        \
                                       \
  nbElems = nb >> 2;                   \
                                       \
  dataA += i*numCols + (COL);          \
  dataB += j*numCols + (COL);          \
                                       \
  while(nbElems>0)                     \
  {                                    \
       f32x4_t tmpa,tmpb;              \
       tmpa = vld1q_f32(dataA,p0);     \
       tmpb = vld1q_f32(dataB,p0);     \
       tmpa = vmlaq_f32(tmpa,tmpb,vec);\
       vst1q_f32(dataA, tmpa, p0);     \
       nbElems--;                      \
       dataA += 4;                     \
       dataB += 4;                     \
  }                                    \
                                       \
  nbElems = nb & 3;                    \
  while(nbElems > 0)                   \
  {                                    \
     *dataA++ += v* *dataB++;          \
     nbElems--;                        \
  }                                    \
}

#define MAS_ROW_F32(COL,A,i,v,B,j)     \
{                                      \
  float32_t *dataA = (A)->pData;       \
  float32_t *dataB = (B)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols - (COL);  \
  int32_t nbElems;                     \
  f32x4_t vec = vdupq_n_f32(v);        \
                                       \
  nbElems = nb >> 2;                   \
                                       \
  dataA += i*numCols + (COL);          \
  dataB += j*numCols + (COL);          \
                                       \
  while(nbElems>0)                     \
  {                                    \
       f32x4_t tmpa,tmpb;              \
       tmpa = vld1q_f32(dataA);        \
       tmpb = vld1q_f32(dataB);        \
       tmpa = vmlsq_f32(tmpa,tmpb,vec);\
       vst1q_f32(dataA, tmpa);         \
       nbElems--;                      \
       dataA += 4;                     \
       dataB += 4;                     \
  }                                    \
                                       \
  nbElems = nb & 3;                    \
  while(nbElems > 0)                   \
  {                                    \
     *dataA++ -= v* *dataB++;          \
     nbElems--;                        \
  }                                    \
}

#define SCALE_ROW_F32(A,COL,v,i)        \
{                                       \
  float32_t *data = (A)->pData;         \
  const int32_t numCols = (A)->numCols; \
  const int32_t nb = numCols - (COL);   \
  int32_t nbElems;                      \
  f32x4_t vec = vdupq_n_f32(v);         \
                                        \
  nbElems = nb >> 2;                    \
                                        \
  data += i*numCols + (COL);            \
  while(nbElems>0)                      \
  {                                     \
       f32x4_t tmpa;                    \
       tmpa = vld1q_f32(data);          \
       tmpa = vmulq_f32(tmpa,vec);      \
       vst1q_f32(data, tmpa);           \
       data += 4;                       \
       nbElems --;                      \
  }                                     \
                                        \
  nbElems = nb & 3;                     \
  while(nbElems > 0)                    \
  {                                     \
     *data++ *= v;                      \
     nbElems--;                         \
  }                                     \
                                        \
}

#else

#define SWAP_ROWS_F32(A,COL,i,j)       \
{                                      \
  int32_t w;                           \
  float32_t tmp;                       \
  float32_t *dataI = (A)->pData;       \
  float32_t *dataJ = (A)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols - COL;    \
                                       \
  dataI += i*numCols + (COL);          \
  dataJ += j*numCols + (COL);          \
                                       \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     tmp = *dataI;                     \
     *dataI++ = *dataJ;                \
     *dataJ++ = tmp;                   \
  }                                    \
}

#define SCALE_ROW_F32(A,COL,v,i)       \
{                                      \
  int32_t w;                           \
  float32_t *data = (A)->pData;        \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols - COL;    \
                                       \
  data += i*numCols + (COL);           \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     *data++ *= v;                     \
  }                                    \
}


#define MAC_ROW_F32(COL,A,i,v,B,j)     \
{                                      \
  int32_t w;                           \
  float32_t *dataA = (A)->pData;       \
  float32_t *dataB = (B)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);    \
                                       \
  dataA = dataA + i*numCols + (COL);   \
  dataB = dataB + j*numCols + (COL);   \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     *dataA++ += v* *dataB++;          \
  }                                    \
}

#define MAS_ROW_F32(COL,A,i,v,B,j)     \
{                                      \
  int32_t w;                           \
  float32_t *dataA = (A)->pData;       \
  float32_t *dataB = (B)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);    \
                                       \
  dataA = dataA + i*numCols + (COL);   \
  dataB = dataB + j*numCols + (COL);   \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     *dataA++ -= v* *dataB++;          \
  }                                    \
}

#endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */

#define SWAP_COLS_F32(A,COL,i,j)               \
{                                              \
  int32_t w;                                  \
  float32_t *data = (A)->pData;                \
  const int32_t numCols = (A)->numCols;       \
  for(w=(COL);w < numCols; w++)                \
  {                                            \
     float32_t tmp;                            \
     tmp = data[w*numCols + i];                \
     data[w*numCols + i] = data[w*numCols + j];\
     data[w*numCols + j] = tmp;                \
  }                                            \
}

#define SCALE_COL_F32(A,ROW,v,i)        \
  SCALE_COL_T(float32_t,,A,ROW,v,i)

#define SWAP_ROWS_F64(A,COL,i,j)       \
{                                      \
  int32_t w;                           \
  float64_t *dataI = (A)->pData;       \
  float64_t *dataJ = (A)->pData;       \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);    \
                                       \
  dataI += i*numCols + (COL);          \
  dataJ += j*numCols + (COL);          \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     float64_t tmp;                    \
     tmp = *dataI;                     \
     *dataI++ = *dataJ;                \
     *dataJ++ = tmp;                   \
  }                                    \
}

#define SWAP_COLS_F64(A,COL,i,j)               \
{                                              \
  int32_t w;                                  \
  float64_t *data = (A)->pData;                \
  const int32_t numCols = (A)->numCols;       \
  for(w=(COL);w < numCols; w++)                \
  {                                            \
     float64_t tmp;                            \
     tmp = data[w*numCols + i];                \
     data[w*numCols + i] = data[w*numCols + j];\
     data[w*numCols + j] = tmp;                \
  }                                            \
}

#define SCALE_ROW_F64(A,COL,v,i)       \
{                                      \
  int32_t w;                           \
  float64_t *data = (A)->pData;        \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);    \
                                       \
  data += i*numCols + (COL);           \
                                       \
  for(w=0;w < nb; w++)                 \
  {                                    \
     *data++ *= v;                     \
  }                                    \
}

#define SCALE_COL_F64(A,ROW,v,i)        \
  SCALE_COL_T(float64_t,,A,ROW,v,i)

#define MAC_ROW_F64(COL,A,i,v,B,j)      \
{                                       \
  int32_t w;                           \
  float64_t *dataA = (A)->pData;        \
  float64_t *dataB = (B)->pData;        \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);     \
                                        \
  dataA += i*numCols + (COL);           \
  dataB += j*numCols + (COL);           \
                                        \
  for(w=0;w < nb; w++)                  \
  {                                     \
     *dataA++ += v* *dataB++;           \
  }                                     \
}

#define MAS_ROW_F64(COL,A,i,v,B,j)      \
{                                       \
  int32_t w;                           \
  float64_t *dataA = (A)->pData;        \
  float64_t *dataB = (B)->pData;        \
  const int32_t numCols = (A)->numCols;\
  const int32_t nb = numCols-(COL);     \
                                        \
  dataA += i*numCols + (COL);           \
  dataB += j*numCols + (COL);           \
                                        \
  for(w=0;w < nb; w++)                  \
  {                                     \
     *dataA++ -= v* *dataB++;           \
  }                                     \
}

#ifdef   __cplusplus
}
#endif

#endif /* ifndef _MATRIX_UTILS_H_ */
