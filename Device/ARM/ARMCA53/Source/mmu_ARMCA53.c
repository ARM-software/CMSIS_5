/**************************************************************************//**
 * @file     mmu_ARMCA53.c
 * @brief    CMSIS Device System Source File for
 *           ARMCA53 Device
 * @version  V1.0.0
 * @date     20. August 2020
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

#if defined (ARMCA53)
  #include "ARMCA53.h"
#elif defined (ARMCA53_FP)
  #include "ARMCA53_FP.h"
#elif defined (ARMCA53_DSP_DP_FP)
  #include "ARMCA53_DSP_DP_FP.h"
#else
  #error device not specified!
#endif

#define TTB_ENTRIES     512
#define TTB_ENTRY_SIZE  64

#define TTB_SIZE        (TTB_ENTRIES)

static uint64_t __ALIGNED(4096) __attribute__((section(".xlat_tables"))) ttb0_base[TTB_SIZE] = {0};
static uint64_t __ALIGNED(4096) __attribute__((section(".xlat_tables"))) level2_pagetable[TTB_SIZE] = {0};


/*----------------------------------------------------------------------------
  MMU initialization
 *----------------------------------------------------------------------------*/
void MMU_CreateTranslationTable(void)
{
  uint64_t mair;
  uint64_t ttb_entry;
  TCR_EL3_Type tcr_el3;

  //According to ARM application note "DAI0527A Bare-metal Boot Code for ARMv8-A Processors"

  // Initialize translation table control registers

  tcr_el3.w = 0;
  tcr_el3.b.T0SZ = 32; //The region size is 2^(64-T0SZ) bytes
  tcr_el3.b.IRGN0 = 1; //Normal memory, Inner Write-Back Read-Allocate Write-Allocate Cacheable.
  tcr_el3.b.ORGN0 = 1; //Normal memory, Outer Write-Back Read-Allocate Write-Allocate Cacheable.
  tcr_el3.b.SH0   = 3; //Inner Shareable.
  tcr_el3.b.TG0   = 0; //Granule size 4KB for the TTBR0_EL3
  tcr_el3.b.PS    = 0; //Physical Address Size 32 bits, 4GB.
  __set_TCR_EL3(tcr_el3.w);

    /* ATTR0 Device-nGnRnE, ATTR1 Device, ATTR2 Normal Non-Cacheable , ATTR3 Normal Cacheable. */
  mair = 0xFF440400;
  __set_MAIR_EL3(mair);

  /* ttb0_base must be a 4KB-aligned address. */
  __set_TTBR0_EL3((uint64_t)&ttb0_base);


  // Set up translation table entries in memory with looped store
  // instructions.
  // Set the level 1 translation table.
  // The first entry points to level2_pagetable.
  ttb_entry = (uint64_t)&level2_pagetable; // Must be a 4KB align address.
  ttb_entry &= 0xFFFFF000; // NSTable=0 APTable=0 XNTable=0 PXNTable=0.
  ttb_entry |= 0x3;
  ttb0_base[0] = ttb_entry;

  // The second entry is 1GB block from 0x40000000 to 0x7FFFFFFF.
  // Executable Inner and Outer Shareable.
  // R/W at all ELs secure memory
  // AttrIdx=000 Device-nGnRnE.
  ttb_entry = 0x40000741;
  ttb0_base[1] = ttb_entry;

  // The third entry is 1GB block from 0x80000000 to 0xBFFFFFFF.
  ttb_entry = 0x80000741;
  ttb0_base[2] = ttb_entry;

  // The fourth entry is 1GB block from 0xC0000000 to 0xFFFFFFFF.
  ttb_entry = 0xC0000741;
  ttb0_base[3] = ttb_entry;

  // Set level 2 translation table.
  ttb_entry = 0x0000074D;
  for(uint32_t l2_idx = 0; l2_idx < TTB_ENTRIES; l2_idx++ ) {
    level2_pagetable[l2_idx] = ttb_entry;
    ttb_entry += 0x00200000; // Increase 2MB address each time.
  }

  __ISB();

}
