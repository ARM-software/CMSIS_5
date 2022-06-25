/**************************************************************************//**
 * @file     system_ARMCM0.c
 * @brief    CMSIS Device System Source File for
 *           ARMCM0 Device
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
#if !defined(__ICCARM__)
struct __FILE {int handle;};
FILE __stdout;
FILE __stdin;
FILE __stderr;
#endif
#endif

#include "ARMCM0.h"

/*----------------------------------------------------------------------------
  Define clocks
 *----------------------------------------------------------------------------*/
#define  XTAL            (50000000UL)     /* Oscillator frequency */

#define  SYSTEM_CLOCK    (XTAL / 2U)

#define SERIAL_BASE_ADDRESS (0x40000000ul)

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

/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
uint32_t SystemCoreClock = SYSTEM_CLOCK;  /* System Core Clock Frequency */


/*----------------------------------------------------------------------------
  System Core Clock update function
 *----------------------------------------------------------------------------*/
void SystemCoreClockUpdate (void)
{
  SystemCoreClock = SYSTEM_CLOCK;
}

/*----------------------------------------------------------------------------
  System initialization function
 *----------------------------------------------------------------------------*/
void SystemInit (void)
{
  SystemCoreClock = SYSTEM_CLOCK;
}

#if 1
int stdout_putchar(char txchar)
{
#if defined(__ICCARM__)
    putchar(txchar);
#else
    SERIAL_DATA = txchar;
#endif
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

#if __IS_COMPILER_ARM_COMPILER_6__
__asm(".global __use_no_semihosting\n\t");
#   ifndef __MICROLIB
__asm(".global __ARM_use_no_argv\n\t");
#   endif
#endif


#if !defined(__ICCARM__)
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
#endif

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

#define log_str(...)                                    \
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
    stdout_putchar(4);
    while(1);
}
#else
void _sys_exit(int n)
{
    (void)n;
    log_str("\n");
    log_str("_[TEST COMPLETE]_________________________________________________\n");
    log_str("\n\n");
    stdout_putchar(4);
    while(1);
}
#endif

extern void ttywrch (int ch);
__attribute__((weak))
void _ttywrch (int ch)
{
    ttywrch(ch);
}

#endif
