/******************************************************************************
 * @file     pmu_armv8.h
 * @brief    CMSIS PMU API for Armv8.1-M PMU
 * @version  V1.0.0
 * @date     24. February 2020
 ******************************************************************************/
/*
 * Copyright (c) 2017-2020 Arm Limited. All rights reserved.
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

#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header    /* treat file as system include file */
#endif

#ifndef ARM_PMU_ARMV8_H
#define ARM_PMU_ARMV8_H

/** \brief PMU Events */

#define ARM_PMU_SW_INCR                              0x0000
#define ARM_PMU_L1I_CACHE_REFILL                     0x0001
#define ARM_PMU_L1D_CACHE_REFILL                     0x0003
#define ARM_PMU_L1D_CACHE                            0x0004
#define ARM_PMU_LD_RETIRED                           0x0006
#define ARM_PMU_ST_RETIRED                           0x0007
#define ARM_PMU_INST_RETIRED                         0x0008
#define ARM_PMU_EXC_TAKEN                            0x0009
#define ARM_PMU_EXC_RETURN                           0x000A
#define ARM_PMU_PC_WRITE_RETIRED                     0x000C
#define ARM_PMU_BR_IMMED_RETIRED                     0x000D
#define ARM_PMU_BR_RETURN_RETIRED                    0x000E
#define ARM_PMU_UNALIGNED_LDST_RETIRED               0x000F
#define ARM_PMU_BR_MIS_PRED                          0x0010
#define ARM_PMU_CPU_CYCLES                           0x0011
#define ARM_PMU_BR_PRED                              0x0012
#define ARM_PMU_MEM_ACCESS                           0x0013
#define ARM_PMU_L1I_CACHE                            0x0014
#define ARM_PMU_L1D_CACHE_WB                         0x0015
#define ARM_PMU_L2D_CACHE                            0x0016
#define ARM_PMU_L2D_CACHE_REFILL                     0x0017
#define ARM_PMU_L2D_CACHE_WB                         0x0018
#define ARM_PMU_BUS_ACCESS                           0x0019
#define ARM_PMU_MEMORY_ERROR                         0x001A
#define ARM_PMU_INST_SPEC                            0x001B
#define ARM_PMU_BUS_CYCLES                           0x001D
#define ARM_PMU_CHAIN                                0x001E 
#define ARM_PMU_L1D_CACHE_ALLOCATE                   0x001F
#define ARM_PMU_L2D_CACHE_ALLOCATE                   0x0020
#define ARM_PMU_BR_RETIRED                           0x0021
#define ARM_PMU_BR_MIS_PRED_RETIRED                  0x0022
#define ARM_PMU_STALL_FRONTEND                       0x0023
#define ARM_PMU_STALL_BACKEND                        0x0024
#define ARM_PMU_L2I_CACHE                            0x0027
#define ARM_PMU_L2I_CACHE_REFILL                     0x0028
#define ARM_PMU_L3D_CACHE_ALLOCATE                   0x0029
#define ARM_PMU_L3D_CACHE_REFILL                     0x002A
#define ARM_PMU_L3D_CACHE                            0x002B
#define ARM_PMU_L3D_CACHE_WB                         0x002C
#define ARM_PMU_LL_CACHE_RD                          0x0036
#define ARM_PMU_LL_CACHE_MISS_RD                     0x0037
#define ARM_PMU_L1D_CACHE_MISS_RD                    0x0039
#define ARM_PMU_OP_COMPLETE                          0x003A
#define ARM_PMU_OP_SPEC                              0x003B
#define ARM_PMU_STALL                                0x003C
#define ARM_PMU_STALL_OP_BACKEND                     0x003D
#define ARM_PMU_STALL_OP_FRONTEND                    0x003E
#define ARM_PMU_STALL_OP                             0x003F
#define ARM_PMU_L1D_CACHE_RD                         0x0040
#define ARM_PMU_LE_RETIRED                           0x0100
#define ARM_PMU_LE_SPEC                              0x0101
#define ARM_PMU_BF_RETIRED                           0x0104
#define ARM_PMU_BF_SPEC                              0x0105
#define ARM_PMU_LE_CANCEL                            0x0108
#define ARM_PMU_BF_CANCEL                            0x0109
#define ARM_PMU_SE_CALL_S                            0x0114
#define ARM_PMU_SE_CALL_NS                           0x0115
#define ARM_PMU_DWT_CMPMATCH0                        0x0118
#define ARM_PMU_DWT_CMPMATCH1                        0x0119
#define ARM_PMU_DWT_CMPMATCH2                        0x011A
#define ARM_PMU_DWT_CMPMATCH3                        0x011B
#define ARM_PMU_MVE_INST_RETIRED                     0x0200
#define ARM_PMU_MVE_INST_SPEC                        0x0201
#define ARM_PMU_MVE_FP_RETIRED                       0x0204
#define ARM_PMU_MVE_FP_SPEC                          0x0205
#define ARM_PMU_MVE_FP_HP_RETIRED                    0x0208
#define ARM_PMU_MVE_FP_HP_SPEC                       0x0209
#define ARM_PMU_MVE_FP_SP_RETIRED                    0x020C
#define ARM_PMU_MVE_FP_SP_SPEC                       0x020D
#define ARM_PMU_MVE_FP_MAC_RETIRED                   0x0214
#define ARM_PMU_MVE_FP_MAC_SPEC                      0x0215
#define ARM_PMU_MVE_INT_RETIRED                      0x0224
#define ARM_PMU_MVE_INT_SPEC                         0x0225
#define ARM_PMU_MVE_INT_MAC_RETIRED                  0x0228
#define ARM_PMU_MVE_INT_MAC_SPEC                     0x0229
#define ARM_PMU_MVE_LDST_RETIRED                     0x0238
#define ARM_PMU_MVE_LDST_SPEC                        0x0239
#define ARM_PMU_MVE_LD_RETIRED                       0x023C
#define ARM_PMU_MVE_LD_SPEC                          0x023D
#define ARM_PMU_MVE_ST_RETIRED                       0x0240
#define ARM_PMU_MVE_ST_SPEC                          0x0241
#define ARM_PMU_MVE_LDST_CONTIG_RETIRED              0x0244
#define ARM_PMU_MVE_LDST_CONTIG_SPEC                 0x0245
#define ARM_PMU_MVE_LD_CONTIG_RETIRED                0x0248
#define ARM_PMU_MVE_LD_CONTIG_SPEC                   0x0249
#define ARM_PMU_MVE_ST_CONTIG_RETIRED                0x024C
#define ARM_PMU_MVE_ST_CONTIG_SPEC                   0x024D
#define ARM_PMU_MVE_LDST_NONCONTIG_RETIRED           0x0250
#define ARM_PMU_MVE_LDST_NONCONTIG_SPEC              0x0251
#define ARM_PMU_MVE_LD_NONCONTIG_RETIRED             0x0254
#define ARM_PMU_MVE_LD_NONCONTIG_SPEC                0x0255
#define ARM_PMU_MVE_ST_NONCONTIG_RETIRED             0x0258
#define ARM_PMU_MVE_ST_NONCONTIG_SPEC                0x0259
#define ARM_PMU_MVE_LDST_MULTI_RETIRED               0x025C
#define ARM_PMU_MVE_LDST_MULTI_SPEC                  0x025D
#define ARM_PMU_MVE_LD_MULTI_RETIRED                 0x0260
#define ARM_PMU_MVE_LD_MULTI_SPEC                    0x0261
#define ARM_PMU_MVE_ST_MULTI_RETIRED                 0x0261
#define ARM_PMU_MVE_ST_MULTI_SPEC                    0x0265
#define ARM_PMU_MVE_LDST_UNALIGNED_RETIRED           0x028C
#define ARM_PMU_MVE_LDST_UNALIGNED_SPEC              0x028D
#define ARM_PMU_MVE_LD_UNALIGNED_RETIRED             0x0290
#define ARM_PMU_MVE_LD_UNALIGNED_SPEC                0x0291
#define ARM_PMU_MVE_ST_UNALIGNED_RETIRED             0x0294
#define ARM_PMU_MVE_ST_UNALIGNED_SPEC                0x0295
#define ARM_PMU_MVE_LDST_UNALIGNED_NONCONTIG_RETIRED 0x0298
#define ARM_PMU_MVE_LDST_UNALIGNED_NONCONTIG_SPEC    0x0299
#define ARM_PMU_MVE_VREDUCE_RETIRED                  0x02A0
#define ARM_PMU_MVE_VREDUCE_SPEC                     0x02A1
#define ARM_PMU_MVE_VREDUCE_FP_RETIRED               0x02A4
#define ARM_PMU_MVE_VREDUCE_FP_SPEC                  0x02A5
#define ARM_PMU_MVE_VREDUCE_INT_RETIRED              0x02A8
#define ARM_PMU_MVE_VREDUCE_INT_SPEC                 0x02A9
#define ARM_PMU_MVE_PRED                             0x02B8
#define ARM_PMU_MVE_STALL                            0x02CC
#define ARM_PMU_MVE_STALL_RESOURCE                   0x02CD
#define ARM_PMU_MVE_STALL_RESOURCE_MEM               0x02CE
#define ARM_PMU_MVE_STALL_RESOURCE_FP                0x02CF
#define ARM_PMU_MVE_STALL_RESOURCE_INT               0x02D0
#define ARM_PMU_MVE_STALL_BREAK                      0x02D3
#define ARM_PMU_MVE_STALL_DEPENDENCY                 0x02D4 
#define ARM_PMU_ITCM_ACCESS                          0x4007
#define ARM_PMU_DTCM_ACCESS                          0x4008
#define ARM_PMU_TRCEXTOUT0                           0x4010  
#define ARM_PMU_TRCEXTOUT1                           0x4011
#define ARM_PMU_TRCEXTOUT2                           0x4012 
#define ARM_PMU_TRCEXTOUT3                           0x4013
#define ARM_PMU_CTI_TRIGOUT4                         0x4018 
#define ARM_PMU_CTI_TRIGOUT5                         0x4019 
#define ARM_PMU_CTI_TRIGOUT6                         0x401A
#define ARM_PMU_CTI_TRIGOUT7                         0x401B

/** \brief PMU Functions */

__STATIC_INLINE void ARM_PMU_Enable(void);
__STATIC_INLINE void ARM_PMU_Disable(void);

__STATIC_INLINE void ARM_PMU_Set_EVTYPER(uint32_t num, uint32_t type);

__STATIC_INLINE void ARM_PMU_CYCCNT_Reset(void);
__STATIC_INLINE void ARM_PMU_EVCNTR_ALL_Reset(void);

__STATIC_INLINE void ARM_PMU_CNTR_Enable(uint32_t mask);
__STATIC_INLINE void ARM_PMU_CNTR_Disable(uint32_t mask);

__STATIC_INLINE uint32_t ARM_PMU_Get_CCNTR(void);
__STATIC_INLINE uint32_t ARM_PMU_Get_EVCNTR(uint32_t num);

__STATIC_INLINE uint32_t ARM_PMU_CNTR_Get_OVSSET(uint32_t mask);
__STATIC_INLINE uint32_t ARM_PMU_CNTR_Set_OVSCLR(uint32_t mask);

__STATIC_INLINE uint32_t ARM_PMU_Set_INTSET(uint32_t mask);
__STATIC_INLINE uint32_t ARM_PMU_Set_INTCLR(uint32_t mask);

__STATIC_INLINE uint32_t ARM_PMU_CNTR_Increment(uint32_t mask);

/** 
  \brief   Enable the PMU
*/
__STATIC_INLINE void ARM_PMU_Enable(void) 
{
  PMU->CTRL |= PMU_CTRL_ENABLE_Msk;
}

/** 
  \brief   Disable the PMU
*/
__STATIC_INLINE void ARM_PMU_Disable(void) 
{
  PMU->CTRL &= ~PMU_CTRL_ENABLE_Msk;
}

/** 
  \brief   Set event to count for PMU eventer counter
  \param [in]    num     Event counter (0-30) to configure
  \param [in]    type    Event to count
*/
__STATIC_INLINE void ARM_PMU_Set_EVTYPER(uint32_t num, uint32_t type)
{
  PMU->EVTYPER[num] = type;
}

/** 
  \brief  Reset cycle counter
*/
__STATIC_INLINE void ARM_PMU_CYCCNT_Reset(void)
{
  PMU->CTRL |= PMU_CTRL_CYCCNT_RESET_Msk;
}

/** 
  \brief  Reset all event counters
*/
__STATIC_INLINE void ARM_PMU_EVCNTR_ALL_Reset(void)
{
  PMU->CTRL |= PMU_CTRL_EVENTCNT_RESET_Msk;
}

/** 
  \brief  Enable counters 
  \param [in]     mask    Counters to enable
  \note   Enables one or more of the following:
          - event counters (0-30)
          - cycle counter
*/
__STATIC_INLINE void ARM_PMU_CNTR_Enable(uint32_t mask)
{
  PMU->CNTENSET = mask;
}

/** 
  \brief  Disable counters
  \param [in]     mask    Counters to enable
  \note   Disables one or more of the following:
          - event counters (0-30)
          - cycle counter
*/
__STATIC_INLINE void ARM_PMU_CNTR_Disable(uint32_t mask)
{
  PMU->CNTENCLR = mask;
}

/** 
  \brief  Read cycle counter
  \return                 Cycle count
*/
__STATIC_INLINE uint32_t ARM_PMU_Get_CCNTR(void)
{
  return PMU->CCNTR;
}

/** 
  \brief   Read event counter
  \param [in]     num     Event counter (0-30) to read
  \return                 Event count
*/
__STATIC_INLINE uint32_t ARM_PMU_Get_EVCNTR(uint32_t num)
{
  return PMU->EVCNTR[num];
}

/** 
  \brief   Read counter overflow status
  \return  Counter overflow status bits for the following:
          - event counters (0-30)
          - cycle counter
*/
__STATIC_INLINE uint32_t ARM_PMU_Get_CNTR_OVS(void)
{
  return PMU->OVSSET;	
}

/** 
  \brief   Clear counter overflow status
  \param [in]     mask    Counter overflow status bits to clear
  \note    Clears overflow status bits for one or more of the following:
           - event counters (0-30)
           - cycle counter
*/
__STATIC_INLINE uint32_t ARM_PMU_Set_CNTR_OVS(uint32_t mask)
{
  PMU->OVSCLR = mask;
}

/** 
  \brief   Enable counter overflow interrupt request 
  \param [in]     mask    Counter overflow interrupt request bits to set
  \note    Sets overflow interrupt request bits for one or more of the following:
           - event counters (0-30)
           - cycle counter
*/
__STATIC_INLINE uint32_t ARM_PMU_Set_CNTR_IRQ_Enable(uint32_t mask)
{
  PMU->INTENSET = mask;
}

/** 
  \brief   Disable counter overflow interrupt request 
  \param [in]     mask    Counter overflow interrupt request bits to clear
  \note    Clears overflow interrupt request bits for one or more of the following:
           - event counters (0-30)
           - cycle counter
*/
__STATIC_INLINE uint32_t ARM_PMU_Set_CNTR_IRQ_Disable(uint32_t mask)
{
  PMU->INTENCLR = mask;
}

/** 
  \brief   Software increment event counter 
  \param [in]     mask    Counters to increment
  \note    Software increment bits for one or more event counters (0-30)
*/
__STATIC_INLINE uint32_t ARM_PMU_CNTR_Increment(uint32_t mask)
{
  PMU->SWINC = mask;
}

#endif /* ARM_PMU_ARMV8_H */
