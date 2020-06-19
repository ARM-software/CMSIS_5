/**************************************************************************//**
 * @file     tcm_init.c
 * @brief    TCM initialization functions for
 *           ARMCR52 Device
 * @version  V1.0.0
 * @date     15. June 2020
 ******************************************************************************/
/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
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
 
#if defined (ARMCR52)
  #include "ARMCR52.h"
#elif defined (ARMCR52_FP)
  #include "ARMCR52_FP.h"
#elif defined (ARMCR52_DSP_DP_FP)
  #include "ARMCR52_DSP_DP_FP.h"
#else
  #error device not specified!
#endif


#define TCM_BASEADDR_MASK   0xFFFFE000
#define TCM_ENABLEEL10      (0x1 << 0)
#define TCM_ENABLEEL2       (0x1 << 1)
#define TCM_SIZE_32KB       (0x6 << 2)

/*----------------------------------------------------------------------------
  External References
 *----------------------------------------------------------------------------*/
extern uint32_t __TCMA_Start, __TCMB_Start, __TCMC_Start;

/*----------------------------------------------------------------------------
  TCM initialization
 *----------------------------------------------------------------------------*/
void TcmInit(void)
{
  uint32_t config;

  config = ((uint32_t)&__TCMA_Start)&TCM_BASEADDR_MASK;
  config |= TCM_SIZE_32KB|TCM_ENABLEEL2|TCM_ENABLEEL10;                /* 32k; EL0/1=ON L2=ON */
  __set_IMP_ATCMREGIONR(config);     /* Write to A-TCM config reg */

  config = ((uint32_t)&__TCMB_Start)&TCM_BASEADDR_MASK;
  config |= TCM_SIZE_32KB|TCM_ENABLEEL2|TCM_ENABLEEL10;                /* 32k; EL0/1=ON L2=ON */
  __set_IMP_BTCMREGIONR(config);     /* Write to B-TCM config reg */

  config = ((uint32_t)&__TCMC_Start)&TCM_BASEADDR_MASK;
  config |= TCM_SIZE_32KB|TCM_ENABLEEL2|TCM_ENABLEEL10;                /* 32k; EL0/1=ON L2=ON */
  __set_IMP_CTCMREGIONR(config);     /* Write to B-TCM config reg */
}

