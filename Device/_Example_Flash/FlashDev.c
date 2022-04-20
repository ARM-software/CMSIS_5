/**************************************************************************//**
 * @file     FlashDev.c
 * @brief    Flash Device Description for ST STM32G4xx 512 Kb Flash
 * @version  V1.0.0
 * @date     10. February 2021
 ******************************************************************************/
/*
 * Copyright (c) 2010-2021 Arm Limited. All rights reserved.
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

#include "FlashOS.h"                     /* FlashOS Structures */

extern
struct FlashDevice const FlashDevice;
struct FlashDevice const FlashDevice  =  {
  FLASH_DRV_VERS,                        /* Driver Version, do not modify! */
  "STM32G4xx 512 KB Flash",              /* Device Name */
  ONCHIP,                                /* Device Type */
  0x08000000,                            /* Device Start Address */
  0x00080000,                            /* Device Size */
  1024,                                  /* Programming Page Size */
  0,                                     /* Reserved, must be 0 */
  0xFF,                                  /* Initial Content of Erased Memory */
  400,                                   /* Program Page Timeout 400 mSec */
  400,                                   /* Erase Sector Timeout 400 mSec */
  
  /* Specify Size and Address of Sectors */
  {
    {0x00001000, 0x00000000},            /* Sector Size  4kB */
    {SECTOR_END            }
  }
};
