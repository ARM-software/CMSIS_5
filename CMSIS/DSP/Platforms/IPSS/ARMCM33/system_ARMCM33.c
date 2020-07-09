/**************************************************************************//**
 * @file     system_ARMCM33.c
 * @brief    CMSIS Device System Source File for
 *           ARMCM33 Device
 * @version  V5.3.1
 * @date     09. July 2018
 ******************************************************************************/
/*
 * Copyright (c) 2009-2018 Arm Limited. All rights reserved.
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

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#if defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6100100)
#include <rt_sys.h>
#else
#define GCCCOMPILER
struct __FILE {int handle;};
FILE __stdout;
FILE __stdin;
FILE __stderr;
#endif

#if defined (ARMCM33)
  #include "ARMCM33.h"
#elif defined (ARMCM33_TZ)
  #include "ARMCM33_TZ.h"

  #if defined (__ARM_FEATURE_CMSE) &&  (__ARM_FEATURE_CMSE == 3U)
    #include "partition_ARMCM33.h"
  #endif
#elif defined (ARMCM33_DSP_FP)
  #include "ARMCM33_DSP_FP.h"
#elif defined (ARMCM33_DSP_FP_TZ)
  #include "ARMCM33_DSP_FP_TZ.h"

  #if defined (__ARM_FEATURE_CMSE) &&  (__ARM_FEATURE_CMSE == 3U)
    #include "partition_ARMCM33.h"
  #endif
#else
  #error device not specified!
#endif


/* ================================================================================ */
/* ================             Peripheral declaration             ================ */
/* ================================================================================ */

#define SERIAL_BASE_ADDRESS (0x70000ul)

#define SERIAL_DATA  *((volatile unsigned *) SERIAL_BASE_ADDRESS)


#define SOFTWARE_MARK  *((volatile unsigned *) (SERIAL_BASE_ADDRESS+4))

void start_ipss_measurement()
{
  SOFTWARE_MARK = 1;
}

void stop_ipss_measurement()
{
  SOFTWARE_MARK = 0;
}

#include "cmsis_compiler.h"

//! \name The macros to identify the compiler
//! @{

//! \note for IAR
#ifdef __IS_COMPILER_IAR__
#   undef __IS_COMPILER_IAR__
#endif
#if defined(__IAR_SYSTEMS_ICC__)
#   define __IS_COMPILER_IAR__                 1
#endif




//! \note for arm compiler 5
#ifdef __IS_COMPILER_ARM_COMPILER_5__
#   undef __IS_COMPILER_ARM_COMPILER_5__
#endif
#if ((__ARMCC_VERSION >= 5000000) && (__ARMCC_VERSION < 6000000))
#   define __IS_COMPILER_ARM_COMPILER_5__      1
#endif
//! @}

//! \note for arm compiler 6
#ifdef __IS_COMPILER_ARM_COMPILER_6__
#   undef __IS_COMPILER_ARM_COMPILER_6__
#endif
#if ((__ARMCC_VERSION >= 6000000) && (__ARMCC_VERSION < 7000000))
#   define __IS_COMPILER_ARM_COMPILER_6__      1
#endif

#ifdef __IS_COMPILER_LLVM__
#   undef  __IS_COMPILER_LLVM__
#endif
#if defined(__clang__) && !__IS_COMPILER_ARM_COMPILER_6__
#   define __IS_COMPILER_LLVM__                1
#else
//! \note for gcc
#ifdef __IS_COMPILER_GCC__
#   undef __IS_COMPILER_GCC__
#endif
#if defined(__GNUC__) && !(__IS_COMPILER_ARM_COMPILER_6__ || __IS_COMPILER_LLVM__)
#   define __IS_COMPILER_GCC__                 1
#endif
//! @}
#endif
//! @}

#define SAFE_ATOM_CODE(...)             \
{                                       \
    uint32_t wOrig = __disable_irq();   \
    __VA_ARGS__;                        \
    __set_PRIMASK(wOrig);               \
}

/* IO definitions (access restrictions to peripheral registers) */
/**
    \defgroup CMSIS_glob_defs CMSIS Global Defines

    <strong>IO Type Qualifiers</strong> are used
    \li to specify the access to peripheral variables.
    \li for automatic generation of peripheral register debug information.
*/
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

/* following defines should be used for structure members */
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */

/*@} end of group Cortex_M */

/*----------------------------------------------------------------------------
  Define clocks
 *----------------------------------------------------------------------------*/
#define  XTAL            ( 5000000UL)      /* Oscillator frequency */

#define  SYSTEM_CLOCK    (5U * XTAL)

#define DEBUG_DEMCR  (*((unsigned int *)0xE000EDFC))
#define DEBUG_TRCENA (1<<24) //Global debug enable bit

#define CCR      (*((volatile unsigned int *)0xE000ED14))
#define CCR_DL   (1 << 19)

/*----------------------------------------------------------------------------
  Externals
 *----------------------------------------------------------------------------*/
#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
  extern uint32_t __VECTOR_TABLE;
#endif

/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
uint32_t SystemCoreClock = SYSTEM_CLOCK;


/*----------------------------------------------------------------------------
  System Core Clock update function
 *----------------------------------------------------------------------------*/
void SystemCoreClockUpdate (void)
{
  SystemCoreClock = SYSTEM_CLOCK;
}

/*----------------------------------------------------------------------------
  UART functions
 *----------------------------------------------------------------------------*/
 
/*------------- Universal Asynchronous Receiver Transmitter (UART) -----------*/
typedef struct
{
  __IOM  uint32_t  DATA;                     /* Offset: 0x000 (R/W) Data Register    */
  __IOM  uint32_t  STATE;                    /* Offset: 0x004 (R/W) Status Register  */
  __IOM  uint32_t  CTRL;                     /* Offset: 0x008 (R/W) Control Register */
  union {
    __IM   uint32_t  INTSTATUS;              /* Offset: 0x00C (R/ ) Interrupt Status Register */
    __OM   uint32_t  INTCLEAR;               /* Offset: 0x00C ( /W) Interrupt Clear Register  */
    };
  __IOM  uint32_t  BAUDDIV;                  /* Offset: 0x010 (R/W) Baudrate Divider Register */

} CMSDK_UART_TypeDef;

/* CMSDK_UART DATA Register Definitions */
#define CMSDK_UART_DATA_Pos               0                                                  /* CMSDK_UART_DATA_Pos: DATA Position */
#define CMSDK_UART_DATA_Msk              (0xFFUL /*<< CMSDK_UART_DATA_Pos*/)                 /* CMSDK_UART DATA: DATA Mask */

/* CMSDK_UART STATE Register Definitions */
#define CMSDK_UART_STATE_RXOR_Pos         3                                                  /* CMSDK_UART STATE: RXOR Position */
#define CMSDK_UART_STATE_RXOR_Msk        (0x1UL << CMSDK_UART_STATE_RXOR_Pos)                /* CMSDK_UART STATE: RXOR Mask */

#define CMSDK_UART_STATE_TXOR_Pos         2                                                  /* CMSDK_UART STATE: TXOR Position */
#define CMSDK_UART_STATE_TXOR_Msk        (0x1UL << CMSDK_UART_STATE_TXOR_Pos)                /* CMSDK_UART STATE: TXOR Mask */

#define CMSDK_UART_STATE_RXBF_Pos         1                                                  /* CMSDK_UART STATE: RXBF Position */
#define CMSDK_UART_STATE_RXBF_Msk        (0x1UL << CMSDK_UART_STATE_RXBF_Pos)                /* CMSDK_UART STATE: RXBF Mask */

#define CMSDK_UART_STATE_TXBF_Pos         0                                                  /* CMSDK_UART STATE: TXBF Position */
#define CMSDK_UART_STATE_TXBF_Msk        (0x1UL /*<< CMSDK_UART_STATE_TXBF_Pos*/)            /* CMSDK_UART STATE: TXBF Mask */

/* CMSDK_UART CTRL Register Definitions */
#define CMSDK_UART_CTRL_HSTM_Pos          6                                                  /* CMSDK_UART CTRL: HSTM Position */
#define CMSDK_UART_CTRL_HSTM_Msk         (0x01UL << CMSDK_UART_CTRL_HSTM_Pos)                /* CMSDK_UART CTRL: HSTM Mask */

#define CMSDK_UART_CTRL_RXORIRQEN_Pos     5                                                  /* CMSDK_UART CTRL: RXORIRQEN Position */
#define CMSDK_UART_CTRL_RXORIRQEN_Msk    (0x01UL << CMSDK_UART_CTRL_RXORIRQEN_Pos)           /* CMSDK_UART CTRL: RXORIRQEN Mask */

#define CMSDK_UART_CTRL_TXORIRQEN_Pos     4                                                  /* CMSDK_UART CTRL: TXORIRQEN Position */
#define CMSDK_UART_CTRL_TXORIRQEN_Msk    (0x01UL << CMSDK_UART_CTRL_TXORIRQEN_Pos)           /* CMSDK_UART CTRL: TXORIRQEN Mask */

#define CMSDK_UART_CTRL_RXIRQEN_Pos       3                                                  /* CMSDK_UART CTRL: RXIRQEN Position */
#define CMSDK_UART_CTRL_RXIRQEN_Msk      (0x01UL << CMSDK_UART_CTRL_RXIRQEN_Pos)             /* CMSDK_UART CTRL: RXIRQEN Mask */

#define CMSDK_UART_CTRL_TXIRQEN_Pos       2                                                  /* CMSDK_UART CTRL: TXIRQEN Position */
#define CMSDK_UART_CTRL_TXIRQEN_Msk      (0x01UL << CMSDK_UART_CTRL_TXIRQEN_Pos)             /* CMSDK_UART CTRL: TXIRQEN Mask */

#define CMSDK_UART_CTRL_RXEN_Pos          1                                                  /* CMSDK_UART CTRL: RXEN Position */
#define CMSDK_UART_CTRL_RXEN_Msk         (0x01UL << CMSDK_UART_CTRL_RXEN_Pos)                /* CMSDK_UART CTRL: RXEN Mask */

#define CMSDK_UART_CTRL_TXEN_Pos          0                                                  /* CMSDK_UART CTRL: TXEN Position */
#define CMSDK_UART_CTRL_TXEN_Msk         (0x01UL /*<< CMSDK_UART_CTRL_TXEN_Pos*/)            /* CMSDK_UART CTRL: TXEN Mask */

#define CMSDK_UART_INTSTATUS_RXORIRQ_Pos  3                                                  /* CMSDK_UART CTRL: RXORIRQ Position */
#define CMSDK_UART_CTRL_RXORIRQ_Msk      (0x01UL << CMSDK_UART_INTSTATUS_RXORIRQ_Pos)        /* CMSDK_UART CTRL: RXORIRQ Mask */

#define CMSDK_UART_CTRL_TXORIRQ_Pos       2                                                  /* CMSDK_UART CTRL: TXORIRQ Position */
#define CMSDK_UART_CTRL_TXORIRQ_Msk      (0x01UL << CMSDK_UART_CTRL_TXORIRQ_Pos)             /* CMSDK_UART CTRL: TXORIRQ Mask */

#define CMSDK_UART_CTRL_RXIRQ_Pos         1                                                  /* CMSDK_UART CTRL: RXIRQ Position */
#define CMSDK_UART_CTRL_RXIRQ_Msk        (0x01UL << CMSDK_UART_CTRL_RXIRQ_Pos)               /* CMSDK_UART CTRL: RXIRQ Mask */

#define CMSDK_UART_CTRL_TXIRQ_Pos         0                                                  /* CMSDK_UART CTRL: TXIRQ Position */
#define CMSDK_UART_CTRL_TXIRQ_Msk        (0x01UL /*<< CMSDK_UART_CTRL_TXIRQ_Pos*/)           /* CMSDK_UART CTRL: TXIRQ Mask */

/* CMSDK_UART BAUDDIV Register Definitions */
#define CMSDK_UART_BAUDDIV_Pos            0                                                  /* CMSDK_UART BAUDDIV: BAUDDIV Position */
#define CMSDK_UART_BAUDDIV_Msk           (0xFFFFFUL /*<< CMSDK_UART_BAUDDIV_Pos*/)           /* CMSDK_UART BAUDDIV: BAUDDIV Mask */





int stdout_putchar(char txchar)
{
    SERIAL_DATA = txchar;    
    return(txchar);                 
}

int stderr_putchar(char txchar)
{
    return stdout_putchar(txchar);
}

void ttywrch (int ch)
{
  stdout_putchar(ch);
}

/*----------------------------------------------------------------------------
  System initialization function
 *----------------------------------------------------------------------------*/
void SystemInit (void)
{

#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
  SCB->VTOR = (uint32_t) &__VECTOR_TABLE;
#endif

#if defined (__FPU_USED) && (__FPU_USED == 1U)
  SCB->CPACR |= ((3U << 10U*2U) |           /* enable CP10 Full Access */
                 (3U << 11U*2U)  );         /* enable CP11 Full Access */
#endif

#ifdef UNALIGNED_SUPPORT_DISABLE
  SCB->CCR |= SCB_CCR_UNALIGN_TRP_Msk;
#endif

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  TZ_SAU_Setup();
#endif

  SystemCoreClock = SYSTEM_CLOCK;
}

__attribute__((constructor(255)))
void platform_init(void)
{
    printf("\n_[TEST START]____________________________________________________\n");
}


#if __IS_COMPILER_ARM_COMPILER_6__
__asm(".global __use_no_semihosting\n\t");
#   ifndef __MICROLIB
__asm(".global __ARM_use_no_argv\n\t");
#   endif
#endif

/**
   Writes the character specified by c (converted to an unsigned char) to
   the output stream pointed to by stream, at the position indicated by the
   associated file position indicator (if defined), and advances the
   indicator appropriately. If the file position indicator is not defined,
   the character is appended to the output stream.
 
  \param[in] c       Character
  \param[in] stream  Stream handle
 
  \return    The character written. If a write error occurs, the error
             indicator is set and fputc returns EOF.
*/
__attribute__((weak))
int fputc (int c, FILE * stream) 
{
    if (stream == &__stdout) {
        return (stdout_putchar(c));
    }

    if (stream == &__stderr) {
        return (stderr_putchar(c));
    }

    return (-1);
}

#ifndef GCCCOMPILER
/* IO device file handles. */
#define FH_STDIN    0x8001
#define FH_STDOUT   0x8002
#define FH_STDERR   0x8003

const char __stdin_name[]  = ":STDIN";
const char __stdout_name[] = ":STDOUT";
const char __stderr_name[] = ":STDERR";

#define RETARGET_SYS        1
#define RTE_Compiler_IO_STDOUT  1
#define RTE_Compiler_IO_STDERR  1
/**
  Defined in rt_sys.h, this function opens a file.
 
  The _sys_open() function is required by fopen() and freopen(). These
  functions in turn are required if any file input/output function is to
  be used.
  The openmode parameter is a bitmap whose bits mostly correspond directly to
  the ISO mode specification. Target-dependent extensions are possible, but
  freopen() must also be extended.
 
  \param[in] name     File name
  \param[in] openmode Mode specification bitmap
 
  \return    The return value is ?1 if an error occurs.
*/
#ifdef RETARGET_SYS
__attribute__((weak))
FILEHANDLE _sys_open (const char *name, int openmode) {
#if (!defined(RTE_Compiler_IO_File))
  (void)openmode;
#endif
 
  if (name == NULL) {
    return (-1);
  }
 
  if (name[0] == ':') {
    if (strcmp(name, ":STDIN") == 0) {
      return (FH_STDIN);
    }
    if (strcmp(name, ":STDOUT") == 0) {
      return (FH_STDOUT);
    }
    if (strcmp(name, ":STDERR") == 0) {
      return (FH_STDERR);
    }
    return (-1);
  }
 
#ifdef RTE_Compiler_IO_File
#ifdef RTE_Compiler_IO_File_FS
  return (__sys_open(name, openmode));
#endif
#else
  return (-1);
#endif
}
#endif
 
 
/**
  Defined in rt_sys.h, this function closes a file previously opened
  with _sys_open().
  
  This function must be defined if any input/output function is to be used.
 
  \param[in] fh File handle
 
  \return    The return value is 0 if successful. A nonzero value indicates
             an error.
*/
#ifdef RETARGET_SYS
__attribute__((weak))
int _sys_close (FILEHANDLE fh) {
 
  switch (fh) {
    case FH_STDIN:
      return (0);
    case FH_STDOUT:
      return (0);
    case FH_STDERR:
      return (0);
  }
 
#ifdef RTE_Compiler_IO_File
#ifdef RTE_Compiler_IO_File_FS
  return (__sys_close(fh));
#endif
#else
  return (-1);
#endif
}
#endif
 
 
/**
  Defined in rt_sys.h, this function writes the contents of a buffer to a file
  previously opened with _sys_open().
 
  \note The mode parameter is here for historical reasons. It contains
        nothing useful and must be ignored.
 
  \param[in] fh   File handle
  \param[in] buf  Data buffer
  \param[in] len  Data length
  \param[in] mode Ignore this parameter
 
  \return    The return value is either:
             - a positive number representing the number of characters not
               written (so any nonzero return value denotes a failure of
               some sort)
             - a negative number indicating an error.
*/
#ifdef RETARGET_SYS
__attribute__((weak))
int _sys_write (FILEHANDLE fh, const uint8_t *buf, uint32_t len, int mode) {
#if (defined(RTE_Compiler_IO_STDOUT) || defined(RTE_Compiler_IO_STDERR))
  int ch;
#elif (!defined(RTE_Compiler_IO_File))
  (void)buf;
  (void)len;
#endif
  (void)mode;
 
  switch (fh) {
    case FH_STDIN:
      return (-1);
    case FH_STDOUT:
#ifdef RTE_Compiler_IO_STDOUT
      for (; len; len--) {
        ch = *buf++;

        stdout_putchar(ch);
      }
#endif
      return (0);
    case FH_STDERR:
#ifdef RTE_Compiler_IO_STDERR
      for (; len; len--) {
        ch = *buf++;

        stderr_putchar(ch);
      }
#endif
      return (0);
  }
 
#ifdef RTE_Compiler_IO_File
#ifdef RTE_Compiler_IO_File_FS
  return (__sys_write(fh, buf, len));
#endif
#else
  return (-1);
#endif
}
#endif
 
 
/**
  Defined in rt_sys.h, this function reads the contents of a file into a buffer.
 
  Reading up to and including the last byte of data does not turn on the EOF
  indicator. The EOF indicator is only reached when an attempt is made to read
  beyond the last byte of data. The target-independent code is capable of
  handling:
    - the EOF indicator being returned in the same read as the remaining bytes
      of data that precede the EOF
    - the EOF indicator being returned on its own after the remaining bytes of
      data have been returned in a previous read.
 
  \note The mode parameter is here for historical reasons. It contains
        nothing useful and must be ignored.
 
  \param[in] fh   File handle
  \param[in] buf  Data buffer
  \param[in] len  Data length
  \param[in] mode Ignore this parameter
 
  \return     The return value is one of the following:
              - The number of bytes not read (that is, len - result number of
                bytes were read).
              - An error indication.
              - An EOF indicator. The EOF indication involves the setting of
                0x80000000 in the normal result.
*/
#ifdef RETARGET_SYS
__attribute__((weak))
int _sys_read (FILEHANDLE fh, uint8_t *buf, uint32_t len, int mode) {
#ifdef RTE_Compiler_IO_STDIN
  int ch;
#elif (!defined(RTE_Compiler_IO_File))
  (void)buf;
  (void)len;
#endif
  (void)mode;
 
  switch (fh) {
    case FH_STDIN:
#ifdef RTE_Compiler_IO_STDIN
      ch = stdin_getchar();
      if (ch < 0) {
        return ((int)(len | 0x80000000U));
      }
      *buf++ = (uint8_t)ch;
#if (STDIN_ECHO != 0)
      stdout_putchar(ch);
#endif
      len--;
      return ((int)(len));
#else
      return ((int)(len | 0x80000000U));
#endif
    case FH_STDOUT:
      return (-1);
    case FH_STDERR:
      return (-1);
  }
 
#ifdef RTE_Compiler_IO_File
#ifdef RTE_Compiler_IO_File_FS
  return (__sys_read(fh, buf, len));
#endif
#else
  return (-1);
#endif
}
#endif
 
 

 
 
/**
  Defined in rt_sys.h, this function determines if a file handle identifies
  a terminal.
 
  When a file is connected to a terminal device, this function is used to
  provide unbuffered behavior by default (in the absence of a call to
  set(v)buf) and to prohibit seeking.
 
  \param[in] fh File handle
 
  \return    The return value is one of the following values:
             - 0:     There is no interactive device.
             - 1:     There is an interactive device.
             - other: An error occurred.
*/
#ifdef RETARGET_SYS
__attribute__((weak))
int _sys_istty (FILEHANDLE fh) {
 
  switch (fh) {
    case FH_STDIN:
      return (1);
    case FH_STDOUT:
      return (1);
    case FH_STDERR:
      return (1);
  }
 
  return (0);
}
#endif
 
 
/**
  Defined in rt_sys.h, this function puts the file pointer at offset pos from
  the beginning of the file.
 
  This function sets the current read or write position to the new location pos
  relative to the start of the current file fh.
 
  \param[in] fh  File handle
  \param[in] pos File pointer offset
 
  \return    The result is:
             - non-negative if no error occurs
             - negative if an error occurs
*/
#ifdef RETARGET_SYS
__attribute__((weak))
int _sys_seek (FILEHANDLE fh, long pos) {
#if (!defined(RTE_Compiler_IO_File))
  (void)pos;
#endif
 
  switch (fh) {
    case FH_STDIN:
      return (-1);
    case FH_STDOUT:
      return (-1);
    case FH_STDERR:
      return (-1);
  }
 
#ifdef RTE_Compiler_IO_File
#ifdef RTE_Compiler_IO_File_FS
  return (__sys_seek(fh, (uint32_t)pos));
#endif
#else
  return (-1);
#endif
}
#endif
 
 
/**
  Defined in rt_sys.h, this function returns the current length of a file.
 
  This function is used by _sys_seek() to convert an offset relative to the
  end of a file into an offset relative to the beginning of the file.
  You do not have to define _sys_flen() if you do not intend to use fseek().
  If you retarget at system _sys_*() level, you must supply _sys_flen(),
  even if the underlying system directly supports seeking relative to the
  end of a file.
 
  \param[in] fh File handle
 
  \return    This function returns the current length of the file fh,
             or a negative error indicator.
*/
#ifdef RETARGET_SYS
__attribute__((weak))
long _sys_flen (FILEHANDLE fh) {
 
  switch (fh) {
    case FH_STDIN:
      return (0);
    case FH_STDOUT:
      return (0);
    case FH_STDERR:
      return (0);
  }
 
#ifdef RTE_Compiler_IO_File
#ifdef RTE_Compiler_IO_File_FS
  return (__sys_flen(fh));
#endif
#else
  return (0);
#endif
}
#endif
 
#else /* gcc compiler */
int _write(int   file,
        char *ptr,
        int   len)
{
  int i;
  (void)file;
  
  for(i=0; i < len;i++)
  {
     stdout_putchar(*ptr++);
  }
  return len;
}

#endif

#define log_str(...)                                \
    do {                                                \
        const char *pchSrc = __VA_ARGS__;               \
        uint_fast16_t hwSize = sizeof(__VA_ARGS__);     \
        do {                                            \
            stdout_putchar(*pchSrc++);                  \
        } while(--hwSize);                              \
    } while(0)


#ifdef GCCCOMPILER
void _exit(int return_code)
{
    (void)return_code;
    log_str("\n");
    log_str("_[TEST COMPLETE]_________________________________________________\n");
    log_str("\n\n");
    *((volatile unsigned *) (SERIAL_BASE_ADDRESS-0x10000)) = 0xa;
    while(1);
}
#else
void _sys_exit(int n)
{
    (void)n;
  log_str("\n");
  log_str("_[TEST COMPLETE]_________________________________________________\n");
  log_str("\n\n");
  *((volatile unsigned *) (SERIAL_BASE_ADDRESS-0x10000)) = 0xa;
  while(1);
}
#endif

extern void ttywrch (int ch);
__attribute__((weak))
void _ttywrch (int ch) 
{
    ttywrch(ch);
}

