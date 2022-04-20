/**************************************************************************//**
 * @file     FlashPrg.c
 * @brief    Flash Programming Functions for ST STM32G4xx 512 Kb Flash
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

#include "FlashOS.h"            /* FlashOS Structures */

#define UNUSED(x) (void)(x)     /* macro to get rid of 'unused parameter' warning */

typedef volatile unsigned long    vu32;
typedef          unsigned long     u32;

#define M32(adr) (*((vu32 *) (adr)))

/* Peripheral Memory Map */
#define FLASH_BASE        (0x40022000)
#define DBGMCU_BASE       (0xE0042000)
#define FLASHSIZE_BASE    (0x1FFF75E0)

#define FLASH           ((FLASH_TypeDef  *) FLASH_BASE)
#define DBGMCU          ((DBGMCU_TypeDef *) DBGMCU_BASE)

/* Debug MCU */
typedef struct {
  vu32 IDCODE;
} DBGMCU_TypeDef;

/* Flash Registers */
typedef struct {
  vu32 ACR;              /* Offset: 0x00  Access Control Register */
  vu32 PDKEYR;           /* Offset: 0x04  Power Down Key Register */
  vu32 KEYR;             /* Offset: 0x08  Key Register */
  vu32 OPTKEYR;          /* Offset: 0x0C  Option Key Register */
  vu32 SR;               /* Offset: 0x10  Status Register */
  vu32 CR;               /* Offset: 0x14  Control Register */
  vu32 ECCR;             /* Offset: 0x18  ECC Register */
  vu32 RESERVED0;
  vu32 OPTR;             /* Offset: 0x20  Option Register */
  vu32 PCROP1SR;         /* Offset: 0x24  Bank1 PCROP Start Address Register */
  vu32 PCROP1ER;         /* Offset: 0x28  Bank1 PCROP End Address Register */
  vu32 WRP1AR;           /* Offset: 0x2C  Bank1 WRP Area A Address Register */
  vu32 WRP1BR;           /* Offset: 0x30  Bank1 WRP Area B Address Register */
  vu32 RESERVED1[4];
  vu32 PCROP2SR;         /* Offset: 0x44  Bank2 PCROP Start Address Register */
  vu32 PCROP2ER;         /* Offset: 0x48  Bank2 PCROP End Address Register */
  vu32 WRP2AR;           /* Offset: 0x4C  Bank2 WRP Area A Address Register */
  vu32 WRP2BR;           /* Offset: 0x50  Bank2 WRP Area B Address Register */
  vu32 RESERVED2[7];
  vu32 SEC1R;            /* Offset: 0x70  Securable Memory Register Bank1 */
  vu32 SEC2R;            /* Offset: 0x74  Securable Memory Register Bank2 */
} FLASH_TypeDef;


/* Flash Keys */
#define FLASH_KEY1               0x45670123
#define FLASH_KEY2               0xCDEF89AB
#define FLASH_OPTKEY1            0x08192A3B
#define FLASH_OPTKEY2            0x4C5D6E7F

/* Flash Control Register definitions */
#define FLASH_CR_PG             ((u32)(  1U      ))
#define FLASH_CR_PER            ((u32)(  1U <<  1))
#define FLASH_CR_MER1           ((u32)(  1U <<  2))
#define FLASH_CR_PNB_MSK        ((u32)(0x7F <<  3))
#define FLASH_CR_BKER           ((u32)(  1U << 11))
#define FLASH_CR_MER2           ((u32)(  1U << 15))
#define FLASH_CR_STRT           ((u32)(  1U << 16))
#define FLASH_CR_OPTSTRT        ((u32)(  1U << 17))
#define FLASH_CR_FSTPG          ((u32)(  1U << 18))
#define FLASH_OBL_LAUNCH        ((u32)(  1U << 27))
#define FLASH_CR_OPTLOCK        ((u32)(  1U << 30))
#define FLASH_CR_LOCK           ((u32)(  1U << 31))


/* Flash Status Register definitions */
#define FLASH_SR_EOP            ((u32)(  1U      ))
#define FLASH_SR_OPERR          ((u32)(  1U <<  1))
#define FLASH_SR_PROGERR        ((u32)(  1U <<  3))
#define FLASH_SR_WRPERR         ((u32)(  1U <<  4))
#define FLASH_SR_PGAERR         ((u32)(  1U <<  5))
#define FLASH_SR_SIZERR         ((u32)(  1U <<  6))
#define FLASH_SR_PGSERR         ((u32)(  1U <<  7))
#define FLASH_SR_MISSERR        ((u32)(  1U <<  8))
#define FLASH_SR_FASTERR        ((u32)(  1U <<  9))
#define FLASH_SR_RDERR          ((u32)(  1U << 14))
#define FLASH_SR_OPTVERR        ((u32)(  1U << 16))
#define FLASH_SR_BSY            ((u32)(  1U << 16))

#define FLASH_PGERR             (FLASH_SR_OPERR   | FLASH_SR_PROGERR | FLASH_SR_WRPERR  | \
                                 FLASH_SR_PGAERR  | FLASH_SR_SIZERR  | FLASH_SR_PGSERR  | \
                                 FLASH_SR_MISSERR | FLASH_SR_FASTERR | FLASH_SR_RDERR   | FLASH_SR_OPTVERR )

/* Flash option register definitions */
#define FLASH_OPTR_RDP          ((u32)(0xFF      ))
#define FLASH_OPTR_RDP_NO       ((u32)(0xAA      ))
#define FLASH_OPTR_DBANK        ((u32)(  1U << 22))

static u32 flashBase;                                    /* Flash base address */
static u32 flashSize;                                    /* Flash size in bytes */
static u32 flashBankSize;                                /* Flash bank size in bytes */
static u32 flashPageSize;                                /* Flash page size in bytes */

static void DSB(void) {
    __asm("DSB");
}

/*
 * Get Flash Type
 *    Return Value:   0 = Single-Bank flash
 *                    1 = Dual-Bank Flash (configurable)
 */
static u32 GetFlashType (void) {
  u32 flashType;

  switch ((DBGMCU->IDCODE & 0xFFFU)) {
    case 0x468:                                          /* Flash Category 2 devices, 2k sectors */
                                                         /* devices have only a singe bank flash */
      flashType = 0U;                                    /* Single-Bank Flash type */
    break;                                               
                                                         
    case 0x469:                                          /* Flash Category 3 devices, 2k or 4k sectors */
    default:                                             /* devices have a dual bank flash, configurable via FLASH_OPTR.DBANK */
      flashType = 1U;                                    /* Dual-Bank Flash type */
    break;
  }

  return (flashType);
}

/*
 * Get Flash Bank Mode
 *    Return Value:   0 = Single-Bank mode
 *                    1 = Dual-Bank mode
 */
static u32 GetFlashBankMode (void) {
  u32 flashBankMode;

  flashBankMode = (FLASH->OPTR & FLASH_OPTR_DBANK) ? 1U : 0U;

  return (flashBankMode);
}

/*
 * Get Flash Bank Number
 *    Parameter:      adr:  Sector Address
 *    Return Value:   Bank Number (0..1)
 */
static u32 GetFlashBankNum(u32 adr) {
  u32 flashBankNum;

  if (GetFlashType() == 1U) {
    /* Dual-Bank Flash */
    if (GetFlashBankMode() == 1U) {
      /* Dual-Bank Flash configured as Dual-Bank */
      if (adr >= (flashBase + flashBankSize)) {
        flashBankNum = 1U;
      }
      else {
        flashBankNum = 0U;
      }
    }
    else {
      /* Dual-Bank Flash configured as Single-Bank */
      flashBankNum = 0U;
    }
  }
  else {
    /* Single-Bank Flash */
    flashBankNum = 0u;
  }

  return (flashBankNum);
}


/*
 * Get Flash Page Number
 *    Parameter:      adr:  Page Address
 *    Return Value:   Page Number (0..127)
 */
static u32 GetFlashPageNum (unsigned long adr) {
  u32 flashPageNum;

  if (GetFlashType() == 1U) {
    /* Dual-Bank Flash */
    if (GetFlashBankMode() == 1U) {
      /* Dual-Bank Flash configured as Dual-Bank */
      flashPageNum = (((adr & (flashBankSize - 1U)) ) >> 11); /* 2K sector size */
    }
    else {
      /* Dual-Bank Flash configured as Single-Bank */
      flashPageNum = (((adr & (flashSize     - 1U)) ) >> 12); /* 4K sector size */
    }
  }
  else {
    /* Single-Bank Flash */
        flashPageNum = (((adr & (flashSize   - 1U)) ) >> 11); /* 2K sector size */
  }

  return (flashPageNum);
}


/*
 * Get Flash Page Size
 *    Return Value:   flash page size (in Bytes)
 */
static u32 GetFlashPageSize (void) {
  u32 pageSize;

  if (GetFlashType() == 1U) {
    /* Dual-Bank Flash */
    if (GetFlashBankMode() == 1U) {
      /* Dual-Bank Flash configured as Dual-Bank */
      pageSize = 0x0800;                                 /* 2K sector size */
    }
    else {
      /* Dual-Bank Flash configured as Single-Bank */
      pageSize = 0x1000;                                 /* 4K sector size */
    }
  }
  else {
    /* Single-Bank Flash */
        pageSize = 0x0800;                               /* 2K sector size */
  }

  return (pageSize);
}


/*
 *  Initialize Flash Programming Functions
 *    Parameter:      adr:  Device Base Address
 *                    clk:  Clock Frequency (Hz)
 *                    fnc:  Function Code (1 - Erase, 2 - Program, 3 - Verify)
 *    Return Value:   0 - OK,  1 - Failed
 */
int Init (unsigned long adr, unsigned long clk, unsigned long fnc) {
  UNUSED(clk);
  UNUSED(fnc);

  FLASH->KEYR = FLASH_KEY1;                              /* Unlock Flash operation */
  FLASH->KEYR = FLASH_KEY2;

  if (((FLASH->OPTR & FLASH_OPTR_RDP) != FLASH_OPTR_RDP_NO))
  {
    FLASH->OPTKEYR = FLASH_OPTKEY1;
    FLASH->OPTKEYR = FLASH_OPTKEY2;

    /* clear  read protection */
    FLASH->OPTR &= ~(FLASH_OPTR_RDP);
    FLASH->OPTR |= FLASH_OPTR_RDP_NO;

    FLASH->CR |= FLASH_CR_OPTSTRT;
    FLASH->CR |= FLASH_OBL_LAUNCH;
    DSB();

    /* Wait until option bytes are updated */
    while (FLASH->CR & FLASH_OBL_LAUNCH);
  }

  /* Wait until the flash is ready */
  while (FLASH->SR & FLASH_SR_BSY);

  flashBase = adr;
  flashSize = ((*((u32 *)FLASHSIZE_BASE)) & 0x0000FFFF) << 10;
  flashBankSize = flashSize >> 1;
  flashPageSize = GetFlashPageSize();

  return (0);                                            /* Finished without Errors */
}


/*
 *  De-Initialize Flash Programming Functions
 *    Parameter:      fnc:  Function Code (1 - Erase, 2 - Program, 3 - Verify)
 *    Return Value:   0 - OK,  1 - Failed
 */
int UnInit (unsigned long fnc) {
  UNUSED(fnc);

  FLASH->CR |= FLASH_CR_LOCK;                            /* Lock Flash operation */
  DSB();

  return (0);                                            /* Finished without Errors */
}


/*
 *  Blank Check Checks if Memory is Blank
 *    Parameter:      adr:  Block Start Address
 *                    sz:   Block Size (in bytes)
 *                    pat:  Block Pattern
 *    Return Value:   0 - OK,  1 - Failed
 */
int BlankCheck (unsigned long adr, unsigned long sz, unsigned char pat) {
  UNUSED(adr);
  UNUSED(sz);
  UNUSED(pat);
  /* force erase even if the content is 'Initial Content of Erased Memory'.
     Only a erased sector can be programmed. I think this is because of ECC */
  return (1);
}


/*
 *  Erase complete Flash Memory
 *    Return Value:   0 - OK,  1 - Failed
 */
int EraseChip (void) {

  FLASH->SR  = FLASH_PGERR;                              /* Reset Error Flags */

  FLASH->CR  = (FLASH_CR_MER1 | FLASH_CR_MER2);          /* Bank A/B mass erase enabled */
  FLASH->CR |=  FLASH_CR_STRT;                           /* Start erase */
  DSB();

  while (FLASH->SR & FLASH_SR_BSY);

  return (0);                                            /* Finished without Errors */
}


/*
 *  Erase Sector in Flash Memory
 *    Parameter:      adr:  Sector Address
 *    Return Value:   0 - OK,  1 - Failed
 */
int EraseSector (unsigned long adr) {
  u32 b, p;

  b = GetFlashBankNum(adr);                              /* Get Bank Number 0..1  */
  p = GetFlashPageNum(adr);                              /* Get Page Number 0..127 */

  FLASH->SR  = FLASH_PGERR;                              /* Reset Error Flags */

  FLASH->CR  = (FLASH_CR_PER |                           /* Page Erase Enabled */
                (p <<  3) |                              /* page Number. 0 to 127 for each bank */
                (b << 11)  );
  FLASH->CR |=  FLASH_CR_STRT;                           /* Start Erase */
  DSB();

  while (FLASH->SR & FLASH_SR_BSY);

  if (FLASH->SR & FLASH_PGERR) {                         /* Check for Error */
    FLASH->SR  = FLASH_PGERR;                            /* Reset Error Flags */
    return (1);                                          /* Failed */
  }

  /* erase 2nd page if we hase 2K physical page size */
  if (flashPageSize == 0x0800U) {
    FLASH->SR  = FLASH_PGERR;                              /* Reset Error Flags */

    FLASH->CR  = (FLASH_CR_PER  |                          /* Page Erase Enabled */
                  ((p+1) <<  3) |                          /* Page Number. 0 to 127 for each bank */
                  ( b    << 11)  );
    FLASH->CR |=  FLASH_CR_STRT;                           /* Start Erase */
    DSB();

    while (FLASH->SR & FLASH_SR_BSY);

    if (FLASH->SR & FLASH_PGERR) {                         /* Check for Error */
      FLASH->SR  = FLASH_PGERR;                            /* Reset Error Flags */
      return (1);                                          /* Failed */
    }
  }

  return (0);                                            /* Finished without Errors */
}


/*
 *  Program Page in Flash Memory
 *    Parameter:      adr:  Page Start Address
 *                    sz:   Page Size
 *                    buf:  Page Data
 *    Return Value:   0 - OK,  1 - Failed
 */
int ProgramPage (unsigned long adr, unsigned long sz, unsigned char *buf) {

  sz = (sz + 7U) & ~7U;                                 /* Adjust size for two words */

  FLASH->SR  = FLASH_PGERR;                              /* Reset Error Flags */

  FLASH->CR = FLASH_CR_PG ;	                             /* Programming Enabled */

  while (sz) {
    M32(adr    ) = *((u32 *)(buf + 0));                  /* Program the first word of the Double Word */
    M32(adr + 4) = *((u32 *)(buf + 4));                  /* Program the second word of the Double Word */
    DSB();

    while (FLASH->SR & FLASH_SR_BSY);

    if (FLASH->SR & FLASH_PGERR) {                       /* Check for Error */
      FLASH->SR  = FLASH_PGERR;                          /* Reset Error Flags */
      return (1);                                        /* Failed */
    }

    adr += 8;                                            /* Go to next DoubleWord */
    buf += 8;
    sz  -= 8;
  }

  FLASH->CR = 0U;                                        /* Reset CR */

  return (0);                                            /* Finished without Errors */
}
