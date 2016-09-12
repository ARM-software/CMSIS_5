/**************************************************************************//**
 * @file     ARMCM4.h
 * @brief    CMSIS Core Peripheral Access Layer Header File for
 *           ARMCM4 Device Series (configured for CM4 without FPU)
 * @version  V5.00
 * @date     07. September 2016
 ******************************************************************************/
/*
 * Copyright (c) 2009-2016 ARM Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ARMCM4_H
#define ARMCM4_H

#ifdef __cplusplus
extern "C" {
#endif


/* -------------------------  Interrupt Number Definition  ------------------------ */

typedef enum IRQn
{
/* -------------------  Cortex-M4 Processor Exceptions Numbers  ------------------- */
  NonMaskableInt_IRQn           = -14,      /*  2 Non Maskable Interrupt */
  HardFault_IRQn                = -13,      /*  3 HardFault Interrupt */
  MemoryManagement_IRQn         = -12,      /*  4 Memory Management Interrupt */
  BusFault_IRQn                 = -11,      /*  5 Bus Fault Interrupt */
  UsageFault_IRQn               = -10,      /*  6 Usage Fault Interrupt */
  SVCall_IRQn                   =  -5,      /* 11 SV Call Interrupt */
  DebugMonitor_IRQn             =  -4,      /* 12 Debug Monitor Interrupt */
  PendSV_IRQn                   =  -2,      /* 14 Pend SV Interrupt */
  SysTick_IRQn                  =  -1,      /* 15 System Tick Interrupt */

/* ----------------------  ARMCM4 Specific Interrupt Numbers  --------------------- */
  WDT_IRQn                      =   0,      /* Watchdog Timer Interrupt */
  RTC_IRQn                      =   1,      /* Real Time Clock Interrupt */
  TIM0_IRQn                     =   2,      /* Timer0 / Timer1 Interrupt */
  TIM2_IRQn                     =   3,      /* Timer2 / Timer3 Interrupt */
  MCIA_IRQn                     =   4,      /* MCIa Interrupt */
  MCIB_IRQn                     =   5,      /* MCIb Interrupt */
  UART0_IRQn                    =   6,      /* UART0 Interrupt */
  UART1_IRQn                    =   7,      /* UART1 Interrupt */
  UART2_IRQn                    =   8,      /* UART2 Interrupt */
  UART4_IRQn                    =   9,      /* UART4 Interrupt */
  AACI_IRQn                     =  10,      /* AACI / AC97 Interrupt */
  CLCD_IRQn                     =  11,      /* CLCD Combined Interrupt */
  ENET_IRQn                     =  12,      /* Ethernet Interrupt */
  USBDC_IRQn                    =  13,      /* USB Device Interrupt */
  USBHC_IRQn                    =  14,      /* USB Host Controller Interrupt */
  CHLCD_IRQn                    =  15,      /* Character LCD Interrupt */
  FLEXRAY_IRQn                  =  16,      /* Flexray Interrupt */
  CAN_IRQn                      =  17,      /* CAN Interrupt */
  LIN_IRQn                      =  18,      /* LIN Interrupt */
  I2C_IRQn                      =  19,      /* I2C ADC/DAC Interrupt */
  CPU_CLCD_IRQn                 =  28,      /* CPU CLCD Combined Interrupt */
  UART3_IRQn                    =  30,      /* UART3 Interrupt */
  SPI_IRQn                      =  31       /* SPI Touchscreen Interrupt */
} IRQn_Type;


/* ================================================================================ */
/* ================      Processor and Core Peripheral Section     ================ */
/* ================================================================================ */

/* -------  Start of section using anonymous unions and disabling warnings  ------- */
#if   defined (__CC_ARM)
  #pragma push
  #pragma anon_unions
#elif defined (__ICCARM__)
  #pragma language=extended
#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wc11-extensions"
  #pragma clang diagnostic ignored "-Wreserved-id-macro"
#elif defined (__GNUC__)
  /* anonymous unions are enabled by default */
#elif defined (__TMS470__)
  /* anonymous unions are enabled by default */
#elif defined (__TASKING__)
  #pragma warning 586
#elif defined (__CSMC__)
  /* anonymous unions are enabled by default */
#else
  #warning Not supported compiler type
#endif


/* --------  Configuration of the Cortex-M4 Processor and Core Peripherals  ------- */
#define __CM4_REV                 0x0001U   /* Core revision r0p1 */
#define __MPU_PRESENT             1U        /* MPU present */
#define __VTOR_PRESENT            1U        /* VTOR present */
#define __NVIC_PRIO_BITS          3U        /* Number of Bits used for Priority Levels */
#define __Vendor_SysTickConfig    0U        /* Set to 1 if different SysTick Config is used */
#define __FPU_PRESENT             0U        /* no FPU present */

#include "core_cm4.h"                       /* Processor and core peripherals */
#include "system_ARMCM4.h"                  /* System Header */


/* ================================================================================ */
/* ================       Device Specific Peripheral Section       ================ */
/* ================================================================================ */


/* ================================================================================ */
/* ================            CPU FPGA System (CPU_SYS)           ================ */
/* ================================================================================ */
typedef struct
{
  __IM  uint32_t ID;              /* Offset: 0x000 (R/ )  Board and FPGA Identifier */
  __IOM uint32_t MEMCFG;          /* Offset: 0x004 (R/W)  Remap and Alias Memory Control */
  __IM  uint32_t SW;              /* Offset: 0x008 (R/ )  Switch States */
  __IOM uint32_t LED;             /* Offset: 0x00C (R/W)  LED Output States */
  __IM  uint32_t TS;              /* Offset: 0x010 (R/ )  Touchscreen Register */
  __IOM uint32_t CTRL1;           /* Offset: 0x014 (R/W)  Misc Control Functions */
        uint32_t RESERVED0[2U];
  __IOM uint32_t CLKCFG;          /* Offset: 0x020 (R/W)  System Clock Configuration */
  __IOM uint32_t WSCFG;           /* Offset: 0x024 (R/W)  Flash Waitstate Configuration */
  __IOM uint32_t CPUCFG;          /* Offset: 0x028 (R/W)  Processor Configuration */
        uint32_t RESERVED1[3U];
  __IOM uint32_t BASE;            /* Offset: 0x038 (R/W)  ROM Table base Address */
  __IOM uint32_t ID2;             /* Offset: 0x03C (R/W)  Secondary Identification Register */
} ARM_CPU_SYS_TypeDef;


/* ================================================================================ */
/* ================            DUT FPGA System (DUT_SYS)           ================ */
/* ================================================================================ */
typedef struct
{
  __IM  uint32_t ID;              /* Offset: 0x000 (R/ )  Board and FPGA Identifier */
  __IOM uint32_t PERCFG;          /* Offset: 0x004 (R/W)  Peripheral Control Signals */
  __IM  uint32_t SW;              /* Offset: 0x008 (R/ )  Switch States */
  __IOM uint32_t LED;             /* Offset: 0x00C (R/W)  LED Output States */
  __IOM uint32_t SEG7;            /* Offset: 0x010 (R/W)  7-segment LED Output States */
  __IM  uint32_t CNT25MHz;        /* Offset: 0x014 (R/ )  Freerunning counter incrementing at 25MHz */
  __IM  uint32_t CNT100Hz;        /* Offset: 0x018 (R/ )  Freerunning counter incrementing at 100Hz */
} ARM_DUT_SYS_TypeDef;


/* ================================================================================ */
/* ================                   Timer (TIM)                  ================ */
/* ================================================================================ */
typedef struct
{
  __IOM uint32_t Timer1Load;      /* Offset: 0x000 (R/W)  Timer 1 Load */
  __IM  uint32_t Timer1Value;     /* Offset: 0x004 (R/ )  Timer 1 Counter Current Value */
  __IOM uint32_t Timer1Control;   /* Offset: 0x008 (R/W)  Timer 1 Control */
  __OM  uint32_t Timer1IntClr;    /* Offset: 0x00C ( /W)  Timer 1 Interrupt Clear */
  __IM  uint32_t Timer1RIS;       /* Offset: 0x010 (R/ )  Timer 1 Raw Interrupt Status */
  __IM  uint32_t Timer1MIS;       /* Offset: 0x014 (R/ )  Timer 1 Masked Interrupt Status */
  __IOM uint32_t Timer1BGLoad;    /* Offset: 0x018 (R/W)  Background Load Register */
        uint32_t RESERVED0[1U];
  __IOM uint32_t Timer2Load;      /* Offset: 0x020 (R/W)  Timer 2 Load */
  __IM  uint32_t Timer2Value;     /* Offset: 0x024 (R/ )  Timer 2 Counter Current Value */
  __IOM uint32_t Timer2Control;   /* Offset: 0x028 (R/W)  Timer 2 Control */
  __OM  uint32_t Timer2IntClr;    /* Offset: 0x02C ( /W)  Timer 2 Interrupt Clear */
  __IM  uint32_t Timer2RIS;       /* Offset: 0x030 (R/ )  Timer 2 Raw Interrupt Status */
  __IM  uint32_t Timer2MIS;       /* Offset: 0x034 (R/ )  Timer 2 Masked Interrupt Status */
  __IOM uint32_t Timer2BGLoad;    /* Offset: 0x038 (R/W)  Background Load Register */
} ARM_TIM_TypeDef;


/* ================================================================================ */
/* ============== Universal Asyncronous Receiver / Transmitter (UART) ============= */
/* ================================================================================ */
typedef struct
{
  __IOM uint32_t DR;              /* Offset: 0x000 (R/W)  Data */
  union {
  __IM  uint32_t RSR;             /* Offset: 0x000 (R/ )  Receive Status */
  __OM  uint32_t ECR;             /* Offset: 0x000 ( /W)  Error Clear */
  };
        uint32_t RESERVED0[4U];
  __IOM uint32_t FR;              /* Offset: 0x018 (R/W)  Flags */
        uint32_t RESERVED1[1U];
  __IOM uint32_t ILPR;            /* Offset: 0x020 (R/W)  IrDA Low-power Counter */
  __IOM uint32_t IBRD;            /* Offset: 0x024 (R/W)  Interger Baud Rate */
  __IOM uint32_t FBRD;            /* Offset: 0x028 (R/W)  Fractional Baud Rate */
  __IOM uint32_t LCR_H;           /* Offset: 0x02C (R/W)  Line Control */
  __IOM uint32_t CR;              /* Offset: 0x030 (R/W)  Control */
  __IOM uint32_t IFLS;            /* Offset: 0x034 (R/W)  Interrupt FIFO Level Select */
  __IOM uint32_t IMSC;            /* Offset: 0x038 (R/W)  Interrupt Mask Set / Clear */
  __IOM uint32_t RIS;             /* Offset: 0x03C (R/W)  Raw Interrupt Status */
  __IOM uint32_t MIS;             /* Offset: 0x040 (R/W)  Masked Interrupt Status */
  __OM  uint32_t ICR;             /* Offset: 0x044 ( /W)  Interrupt Clear */
  __IOM uint32_t DMACR;           /* Offset: 0x048 (R/W)  DMA Control */
} ARM_UART_TypeDef;


/* --------  End of section using anonymous unions and disabling warnings  -------- */
#if   defined (__CC_ARM)
  #pragma pop
#elif defined (__ICCARM__)
  /* leave anonymous unions enabled */
#elif (__ARMCC_VERSION >= 6010050)
  #pragma clang diagnostic pop
#elif defined (__GNUC__)
  /* anonymous unions are enabled by default */
#elif defined (__TMS470__)
  /* anonymous unions are enabled by default */
#elif defined (__TASKING__)
  #pragma warning restore
#elif defined (__CSMC__)
  /* anonymous unions are enabled by default */
#else
  #warning Not supported compiler type
#endif




/* ================================================================================ */
/* ================              Peripheral memory map             ================ */
/* ================================================================================ */
/* --------------------------  CPU FPGA memory map  ------------------------------- */
#define ARM_FLASH_BASE            (0x00000000UL)
#define ARM_RAM_BASE              (0x20000000UL)
#define ARM_RAM_FPGA_BASE         (0x1EFF0000UL)
#define ARM_CPU_CFG_BASE          (0xDFFF0000UL)

#define ARM_CPU_SYS_BASE          (ARM_CPU_CFG_BASE  + 0x00000UL)
#define ARM_UART3_BASE            (ARM_CPU_CFG_BASE  + 0x05000UL)

/* --------------------------  DUT FPGA memory map  ------------------------------- */
#define ARM_APB_BASE              (0x40000000UL)
#define ARM_AHB_BASE              (0x4FF00000UL)
#define ARM_DMC_BASE              (0x60000000UL)
#define ARM_SMC_BASE              (0xA0000000UL)

#define ARM_TIM0_BASE             (ARM_APB_BASE      + 0x02000UL)
#define ARM_TIM2_BASE             (ARM_APB_BASE      + 0x03000UL)
#define ARM_DUT_SYS_BASE          (ARM_APB_BASE      + 0x04000UL)
#define ARM_UART0_BASE            (ARM_APB_BASE      + 0x06000UL)
#define ARM_UART1_BASE            (ARM_APB_BASE      + 0x07000UL)
#define ARM_UART2_BASE            (ARM_APB_BASE      + 0x08000UL)
#define ARM_UART4_BASE            (ARM_APB_BASE      + 0x09000UL)


/* ================================================================================ */
/* ================             Peripheral declaration             ================ */
/* ================================================================================ */
/* --------------------------  CPU FPGA Peripherals  ------------------------------ */
#define ARM_CPU_SYS               ((ARM_CPU_SYS_TypeDef *)  ARM_CPU_SYS_BASE)
#define ARM_UART3                 ((   ARM_UART_TypeDef *)    ARM_UART3_BASE)

/* --------------------------  DUT FPGA Peripherals  ------------------------------ */
#define ARM_DUT_SYS               ((ARM_DUT_SYS_TypeDef *)  ARM_DUT_SYS_BASE)
#define ARM_TIM0                  ((    ARM_TIM_TypeDef *)     ARM_TIM0_BASE)
#define ARM_TIM2                  ((    ARM_TIM_TypeDef *)     ARM_TIM2_BASE)
#define ARM_UART0                 ((   ARM_UART_TypeDef *)    ARM_UART0_BASE)
#define ARM_UART1                 ((   ARM_UART_TypeDef *)    ARM_UART1_BASE)
#define ARM_UART2                 ((   ARM_UART_TypeDef *)    ARM_UART2_BASE)
#define ARM_UART4                 ((   ARM_UART_TypeDef *)    ARM_UART4_BASE)


#ifdef __cplusplus
}
#endif

#endif  /* ARMCM4_H */
