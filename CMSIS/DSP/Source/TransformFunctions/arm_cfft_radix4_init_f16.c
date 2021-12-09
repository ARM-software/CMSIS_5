/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_cfft_radix4_init_f16.c
 * Description:  Radix-4 Decimation in Frequency Floating-point CFFT & CIFFT Initialization function
 *
 * $Date:        23 April 2021
 * $Revision:    V1.9.0
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

#include "dsp/transform_functions_f16.h"
#include "arm_common_tables.h"
#include "arm_common_tables_f16.h"

/**
  @ingroup groupTransforms
 */

/**
  @addtogroup ComplexFFT
  @{
 */

/**
  @brief         Initialization function for the floating-point CFFT/CIFFT.
  @deprecated    Do not use this function. It has been superceded by \ref arm_cfft_f16 and will be removed in the future.
  @param[in,out] S              points to an instance of the floating-point CFFT/CIFFT structure
  @param[in]     fftLen         length of the FFT
  @param[in]     ifftFlag       flag that selects transform direction
                   - value = 0: forward transform
                   - value = 1: inverse transform
  @param[in]     bitReverseFlag flag that enables / disables bit reversal of output
                   - value = 0: disables bit reversal of output
                   - value = 1: enables bit reversal of output
  @return        execution status
                   - \ref ARM_MATH_SUCCESS        : Operation successful
                   - \ref ARM_MATH_ARGUMENT_ERROR : <code>fftLen</code> is not a supported length

  @par           Details
                   The parameter <code>ifftFlag</code> controls whether a forward or inverse transform is computed.
                   Set(=1) ifftFlag for calculation of CIFFT otherwise  CFFT is calculated
  @par
                   The parameter <code>bitReverseFlag</code> controls whether output is in normal order or bit reversed order.
                   Set(=1) bitReverseFlag for output to be in normal order otherwise output is in bit reversed order.
  @par
                   The parameter <code>fftLen</code> Specifies length of CFFT/CIFFT process. Supported FFT Lengths are 16, 64, 256, 1024.
  @par
                   This Function also initializes Twiddle factor table pointer and Bit reversal table pointer.
 */

#if defined(ARM_FLOAT16_SUPPORTED)

arm_status arm_cfft_radix4_init_f16(
  arm_cfft_radix4_instance_f16 * S,
  uint16_t fftLen,
  uint8_t ifftFlag,
  uint8_t bitReverseFlag)
{
    /*  Initialise the default arm status */
  arm_status status = ARM_MATH_ARGUMENT_ERROR;

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_F16_4096)

  /*  Initialise the default arm status */
  status = ARM_MATH_SUCCESS;

  /*  Initialise the FFT length */
  S->fftLen = fftLen;

  /*  Initialise the Twiddle coefficient pointer */
  S->pTwiddle = (float16_t *) twiddleCoefF16;

  /*  Initialise the Flag for selection of CFFT or CIFFT */
  S->ifftFlag = ifftFlag;

  /*  Initialise the Flag for calculation Bit reversal or not */
  S->bitReverseFlag = bitReverseFlag;

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_BITREV_1024)

  /*  Initializations of structure parameters depending on the FFT length */
  switch (S->fftLen)
  {

  case 4096U:
    /*  Initializations of structure parameters for 4096 point FFT */

    /*  Initialise the twiddle coef modifier value */
    S->twidCoefModifier = 1U;
    /*  Initialise the bit reversal table modifier */
    S->bitRevFactor = 1U;
    /*  Initialise the bit reversal table pointer */
    S->pBitRevTable = (uint16_t *) armBitRevTable;
    /*  Initialise the 1/fftLen Value */
    S->onebyfftLen = 0.000244140625;
    break;

  case 1024U:
    /*  Initializations of structure parameters for 1024 point FFT */

    /*  Initialise the twiddle coef modifier value */
    S->twidCoefModifier = 4U;
    /*  Initialise the bit reversal table modifier */
    S->bitRevFactor = 4U;
    /*  Initialise the bit reversal table pointer */
    S->pBitRevTable = (uint16_t *) & armBitRevTable[3];
    /*  Initialise the 1/fftLen Value */
    S->onebyfftLen = 0.0009765625f;
    break;


  case 256U:
    /*  Initializations of structure parameters for 256 point FFT */
    S->twidCoefModifier = 16U;
    S->bitRevFactor = 16U;
    S->pBitRevTable = (uint16_t *) & armBitRevTable[15];
    S->onebyfftLen = 0.00390625f;
    break;

  case 64U:
    /*  Initializations of structure parameters for 64 point FFT */
    S->twidCoefModifier = 64U;
    S->bitRevFactor = 64U;
    S->pBitRevTable = (uint16_t *) & armBitRevTable[63];
    S->onebyfftLen = 0.015625f;
    break;

  case 16U:
    /*  Initializations of structure parameters for 16 point FFT */
    S->twidCoefModifier = 256U;
    S->bitRevFactor = 256U;
    S->pBitRevTable = (uint16_t *) & armBitRevTable[255];
    S->onebyfftLen = 0.0625f;
    break;


  default:
    /*  Reporting argument error if fftSize is not valid value */
    status = ARM_MATH_ARGUMENT_ERROR;
    break;
  }

#endif
#endif
#endif
  return (status);
}
#endif /* #if defined(ARM_FLOAT16_SUPPORTED) */
/**
  @} end of ComplexFFT group
 */
