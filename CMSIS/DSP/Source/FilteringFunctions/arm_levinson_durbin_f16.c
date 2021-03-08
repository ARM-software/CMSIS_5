/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_levinson_durbin_f16.c
 * Description:  f16 version of Levinson Durbin algorithm
 *
 *
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

#include "dsp/filtering_functions_f16.h"

/**
  @ingroup groupFilters
 */

/**
  @defgroup LD Levinson Durbin Algorithm

 */

/**
  @addtogroup LD
  @{
 */

/**
  @brief         Levinson Durbin
  @param[in]     phi      autocovariance vector starting with lag 0 (length is nbCoefs + 1)
  @param[out]    a        autoregressive coefficients
  @param[out]    err      prediction error (variance)
  @param[in]     nbCoefs  number of autoregressive coefficients
  @return        none
 */

#if defined(ARM_FLOAT16_SUPPORTED)

void arm_levinson_durbin_f16(const float16_t *phi,
  float16_t *a, 
  float16_t *err,
  int nbCoefs)
{
   float16_t e;

   a[0] = phi[1] / phi[0];

   e = phi[0] - phi[1] * a[0];
   for(int p=1; p < nbCoefs; p++)
   {
      float16_t suma=0.0f16;
      float16_t sumb=0.0f16;
      float16_t k;
      int nb,j;

      for(int i=0; i < p; i++)
      {
         suma += a[i] * phi[p - i];
         sumb += a[i] * phi[i + 1];
      }

      k = (phi[p+1]-suma)/(phi[0] - sumb);


      nb = p >> 1;
      j=0;
      for(int i =0;i < nb ; i++)
      {
          float16_t x,y;

          x=a[j] - k * a[p-1-j];
          y=a[p-1-j] - k * a[j];

          a[j] = x;
          a[p-1-j] = y;

          j++;
      }

      nb = p & 1;
      if (nb)
      {
            a[j]=a[j]- k * a[p-1-j];
      }

      a[p] = k;
      e = e * (1 - k*k);


   }
   *err = e;
}
#endif /* #if defined(ARM_FLOAT16_SUPPORTED) */
/**
  @} end of LD group
 */
