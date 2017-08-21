/******************************************************************************
 * @file     mpu_armv8.h
 * @brief    CMSIS MPU API for ARMv8 MPU
 * @version  V5.0.3
 * @date     09. August 2017
 ******************************************************************************/
/*
 * Copyright (c) 2017 ARM Limited. All rights reserved.
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
 
#ifndef ARM_MPU_ARMV8_H
#define ARM_MPU_ARMV8_H

/** \brief Attribute for device memory (outer only) */
#define ARM_MPU_ATTR_DEVICE                           ( 0u )

/** \brief Attribute for non-cacheable, normal memory */
#define ARM_MPU_ATTR_NON_CACHEABLE                    ( 4u )

/** \brief Attribute for normal memory (outer and inner)
* \param NT Non-Transient: Set to 1 for non-transient data.
* \param WB Write-Back: Set to 1 to use write-back update policy.
* \param RA Read Allocation: Set to 1 to use cache allocation on read miss.
* \param WA Write Allocation: Set to 1 to use cache allocation on write miss.
*/
#define ARM_MPU_ATTR_MEMORY_(NT, WB, RA, WA) \
  (((NT & 1u) << 3u) | ((WB & 1u) << 2u) | ((RA & 1u) << 1u) | (WA & 1u))

/** \brief Device memory type non Gathering, non Re-ordering, non Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_nGnRnE (0u)

/** \brief Device memory type non Gathering, non Re-ordering, Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_nGnRE  (1u)

/** \brief Device memory type non Gathering, Re-ordering, Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_nGRE   (2u)

/** \brief Device memory type Gathering, Re-ordering, Early Write Acknowledgement */
#define ARM_MPU_ATTR_DEVICE_GRE    (3u)

/** \brief Memory Attribute
* \param O Outer memory attributes
* \param I O == ARM_MPU_ATTR_DEVICE: Device memory attributes, else: Inner memory attributes
*/
#define ARM_MPU_ATTR(O, I) (((O & 0xFu) << 4u) | (((O & 0xFu) != 0u) ? (I & 0xFu) : ((I & 0x3u) << 2u)))

/** \brief Normal memory non-shareable  */
#define ARM_MPU_SH_NON   (0u)

/** \brief Normal memory outer shareable  */
#define ARM_MPU_SH_OUTER (2u)

/** \brief Normal memory inner shareable  */
#define ARM_MPU_SH_INNER (3u)

/** \brief Memory access permissions
* \param RO Read-Only: Set to 1 for read-only memory.
* \param NP Non-Privileged: Set to 1 for non-privileged memory.
*/
#define ARM_MPU_AP_(RO, NP) (((RO & 1u) << 1u) | (NP & 1u))

/** \brief Region Base Address Register value
* \param BASE The base address bits [31:5] of a memory region. The value is zero extended. Effective address gets 32 byte aligned.
* \param SH Defines the Shareability domain for this memory region.
* \param RO Read-Only: Set to 1 for a read-only memory region.
* \param NP Non-Privileged: Set to 1 for a non-privileged memory region.
* \oaram XN eXecute Never: Set to 1 for a non-executable memory region.
*/
#define ARM_MPU_RBAR(BASE, SH, RO, NP, XN) \
  ((BASE & MPU_RBAR_ADDR_Msk) | \
  ((SH << MPU_RBAR_SH_Pos) & MPU_RBAR_SH_Msk) | \
  ((ARM_MPU_AP_(RO, NP) << MPU_RBAR_AP_Pos) & MPU_RBAR_AP_Msk) | \
  ((XN << MPU_RBAR_XN_Pos) & MPU_RBAR_XN_Msk))

/** \brief Region Limit Address Register value
* \param LIMIT The limit address bits [31:5] for this memory region. The value is one extended.
* \param IDX The attribute index to be associated with this memory region.
*/
#define ARM_MPU_RLAR(LIMIT, IDX) \
  ((LIMIT & MPU_RLAR_LIMIT_Msk) | \
  ((IDX << MPU_RLAR_AttrIndx_Pos) & MPU_RLAR_AttrIndx_Msk) | \
  (MPU_RLAR_EN_Msk))

/**
* Struct for a single MPU Region
*/
typedef struct _ARM_MPU_Region_t {
  __IOM uint32_t RBAR;                   /*!< Region Base Address Register value */
  __IOM uint32_t RLAR;                   /*!< Region Limit Address Register value */
} ARM_MPU_Region_t;
    
/** Enable the MPU.
* \param MPU_Control Default access permissions for unconfigured regions.
*/
__STATIC_INLINE void ARM_MPU_Enable(uint32_t MPU_Control)
{
  __DSB();
  __ISB();
  MPU->CTRL = MPU_Control | MPU_CTRL_ENABLE_Msk;
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
  SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
#endif
}

/** Disable the MPU.
*/
__STATIC_INLINE void ARM_MPU_Disable()
{
  __DSB();
  __ISB();
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
  SCB->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk;
#endif
  MPU->CTRL  &= ~MPU_CTRL_ENABLE_Msk;
}

#ifdef MPU_NS
/** Enable the Non-secure MPU.
* \param MPU_Control Default access permissions for unconfigured regions.
*/
__STATIC_INLINE void ARM_MPU_Enable_NS(uint32_t MPU_Control)
{
  __DSB();
  __ISB();
  MPU_NS->CTRL = MPU_Control | MPU_CTRL_ENABLE_Msk;
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
  SCB_NS->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
#endif
}

/** Disable the Non-secure MPU.
*/
__STATIC_INLINE void ARM_MPU_Disable_NS()
{
  __DSB();
  __ISB();
#ifdef SCB_SHCSR_MEMFAULTENA_Msk
  SCB_NS->SHCSR &= ~SCB_SHCSR_MEMFAULTENA_Msk;
#endif
  MPU_NS->CTRL  &= ~MPU_CTRL_ENABLE_Msk;
}
#endif

/** Set the memory attribute encoding to the given MPU.
* \param mpu Pointer to the MPU to be configured.
* \param idx The attribute index to be set [0-7]
* \param attr The attribute value to be set.
*/
__STATIC_INLINE void ARM_MPU_SetMemAttrEx(MPU_Type* mpu, uint8_t idx, uint8_t attr)
{
  const uint8_t reg = idx / 4u;
  const uint32_t pos = ((idx % 4u) * 8u);
  const uint32_t mask = 0xFFu << pos;
  
  if (reg >= (sizeof(MPU->MAIR) / sizeof(MPU->MAIR[0]))) {
    return; // invalid index
  }
  
  MPU->MAIR[reg] = ((MPU->MAIR[reg] & ~mask) | ((attr << pos) & mask));
}

/** Set the memory attribute encoding.
* \param idx The attribute index to be set [0-7]
* \param attr The attribute value to be set.
*/
__STATIC_INLINE void ARM_MPU_SetMemAttr(uint8_t idx, uint8_t attr)
{
  ARM_MPU_SetMemAttrEx(MPU, idx, attr);
}

#ifdef MPU_NS
/** Set the memory attribute encoding to the Non-secure MPU.
* \param idx The attribute index to be set [0-7]
* \param attr The attribute value to be set.
*/
__STATIC_INLINE void ARM_MPU_SetMemAttr_NS(uint8_t idx, uint8_t attr)
{
  ARM_MPU_SetMemAttrEx(MPU_NS, idx, attr);
}
#endif

/** Clear and disable the given MPU region of the given MPU.
* \param mpu Pointer to MPU to be used.
* \param rnr Region number to be cleared.
*/
__STATIC_INLINE void ARM_MPU_ClrRegionEx(MPU_Type* mpu, uint32_t rnr)
{
  MPU->RNR = rnr;
  MPU->RLAR = 0u;
}

/** Clear and disable the given MPU region.
* \param rnr Region number to be cleared.
*/
__STATIC_INLINE void ARM_MPU_ClrRegion(uint32_t rnr)
{
  ARM_MPU_ClrRegionEx(MPU, rnr);
}

#ifdef MPU_NS
/** Clear and disable the given Non-secure MPU region.
* \param rnr Region number to be cleared.
*/
__STATIC_INLINE void ARM_MPU_ClrRegion_NS(uint32_t rnr)
{  
  ARM_MPU_ClrRegionEx(MPU_NS, rnr);
}
#endif

/** Configure the given MPU region of the given MPU.
* \param mpu Pointer to MPU to be used.
* \param rnr Region number to be configured.
* \param rbar Value for RBAR register.
* \param rlar Value for RLAR register.
*/   
__STATIC_INLINE void ARM_MPU_SetRegionEx(MPU_Type* mpu, uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  MPU->RNR = rnr;
  MPU->RBAR = rbar;
  MPU->RLAR = rlar;
}

/** Configure the given MPU region.
* \param rnr Region number to be configured.
* \param rbar Value for RBAR register.
* \param rlar Value for RLAR register.
*/   
__STATIC_INLINE void ARM_MPU_SetRegion(uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  ARM_MPU_SetRegionEx(MPU, rnr, rbar, rlar);
}

#ifdef MPU_NS
/** Configure the given Non-secure MPU region.
* \param rnr Region number to be configured.
* \param rbar Value for RBAR register.
* \param rlar Value for RLAR register.
*/   
__STATIC_INLINE void ARM_MPU_SetRegion_NS(uint32_t rnr, uint32_t rbar, uint32_t rlar)
{
  ARM_MPU_SetRegionEx(MPU_NS, rnr, rbar, rlar);  
}
#endif

/** Memcopy with strictly ordered memory access, e.g. for register targets.
* \param dst Destination data is copied to.
* \param src Source data is copied from.
* \param len Amount of data words to be copied.
*/
__STATIC_INLINE void orderedCpy(volatile uint32_t* dst, const uint32_t* __RESTRICT src, uint32_t len)
{
  uint32_t i;
  for (i = 0u; i < len; ++i) 
  {
    dst[i] = src[i];
  }
}

/** Load the given number of MPU regions from a table to the given MPU.
* \param mpu Pointer to the MPU registers to be used.
* \param rnr First region number to be configured.
* \param table Pointer to the MPU configuration table.
* \param cnt Amount of regions to be configured.
*/
__STATIC_INLINE void ARM_MPU_LoadEx(MPU_Type* mpu, uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt) 
{
  static const uint32_t rowWordSize = sizeof(ARM_MPU_Region_t)/4u;
  if (cnt == 1u) {
    mpu->RNR = rnr;
    orderedCpy(&(mpu->RBAR), &(table->RBAR), rowWordSize);
  } else {
    uint32_t rnrBase   = rnr & ~(MPU_TYPE_RALIASES-1u);
    uint32_t rnrOffset = rnr % MPU_TYPE_RALIASES;
    
    mpu->RNR = rnrBase;
    if ((rnrOffset + cnt) > MPU_TYPE_RALIASES) {
      uint32_t c = MPU_TYPE_RALIASES - rnrOffset;
      orderedCpy(&(mpu->RBAR)+(rnrOffset*2u), &(table->RBAR), c*rowWordSize);
      ARM_MPU_LoadEx(mpu, rnr + c, table + c, cnt - c);
    } else {
      orderedCpy(&(mpu->RBAR)+(rnrOffset*2u), &(table->RBAR), cnt*rowWordSize);
    }
  }
}

/** Load the given number of MPU regions from a table.
* \param rnr First region number to be configured.
* \param table Pointer to the MPU configuration table.
* \param cnt Amount of regions to be configured.
*/
__STATIC_INLINE void ARM_MPU_Load(uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt) 
{
  ARM_MPU_LoadEx(MPU, rnr, table, cnt);
}

#ifdef MPU_NS
/** Load the given number of MPU regions from a table to the Non-secure MPU.
* \param rnr First region number to be configured.
* \param table Pointer to the MPU configuration table.
* \param cnt Amount of regions to be configured.
*/
__STATIC_INLINE void ARM_MPU_Load_NS(uint32_t rnr, ARM_MPU_Region_t const* table, uint32_t cnt) 
{
  ARM_MPU_LoadEx(MPU_NS, rnr, table, cnt);
}
#endif

#endif

