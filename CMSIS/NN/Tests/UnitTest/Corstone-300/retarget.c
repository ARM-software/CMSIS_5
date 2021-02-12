/*
 * Copyright (C) 2010-2021 Arm Limited or its affiliates. All rights reserved.
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

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6100100) && !defined(GCCCOMPILER)
#include <rt_misc.h>
#include <rt_sys.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#else
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#endif

#include "uart.h"

unsigned char UartPutc(unsigned char ch) { return uart_putc(ch); }

unsigned char UartGetc(void) { return uart_putc(uart_getc()); }

__attribute__((noreturn)) void UartEndSimulation(int code)
{
    UartPutc((char)0x4);  // End of simulation
    UartPutc((char)code); // Exit code
    while (1)
    {
    }
}

void exit(int code)
{
    UartEndSimulation(code);
    while (1)
    {
    }
}

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6100100) && !defined(GCCCOMPILER)
int fputc(int ch, FILE *f)
{
    (void)(f);
    return UartPutc(ch);
}

int fgetc(FILE *f)
{
    (void)f;
    return UartPutc(UartGetc());
}
#else
int SER_PutChar(int c) { return UartPutc(c); }

int SER_GetChar(void) { return UartPutc(UartGetc()); }
#endif

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6100100) && !defined(GCCCOMPILER)
/**
   Copied from CMSIS/DSP/Platforms/FVP/ARMv81MML/system_ARMv81MML.c
*/

#define FH_STDIN 0x8001
#define FH_STDOUT 0x8002
#define FH_STDERR 0x8003

const char __stdin_name[] = ":STDIN";
const char __stdout_name[] = ":STDOUT";
const char __stderr_name[] = ":STDERR";

/**
  The following _sys_xxx functions are defined in rt_sys.h.
*/

__attribute__((weak)) FILEHANDLE _sys_open(const char *name, int openmode)
{
    (void)openmode;

    if (name == NULL)
    {
        return (-1);
    }

    if (name[0] == ':')
    {
        if (strcmp(name, ":STDIN") == 0)
        {
            return (FH_STDIN);
        }
        if (strcmp(name, ":STDOUT") == 0)
        {
            return (FH_STDOUT);
        }
        if (strcmp(name, ":STDERR") == 0)
        {
            return (FH_STDERR);
        }
        return (-1);
    }

    return (-1);
}

__attribute__((weak)) int _sys_close(FILEHANDLE fh)
{

    switch (fh)
    {
    case FH_STDIN:
        return (0);
    case FH_STDOUT:
        return (0);
    case FH_STDERR:
        return (0);
    }

    return (-1);
}

__attribute__((weak)) int _sys_write(FILEHANDLE fh, const uint8_t *buf, uint32_t len, int mode)
{
    (void)buf;
    (void)len;
    (void)mode;

    switch (fh)
    {
    case FH_STDIN:
        return (-1);
    case FH_STDOUT:
        return (0);
    case FH_STDERR:
        return (0);
    }

    return (-1);
}

__attribute__((weak)) int _sys_read(FILEHANDLE fh, uint8_t *buf, uint32_t len, int mode)
{
    (void)buf;
    (void)len;
    (void)mode;

    switch (fh)
    {
    case FH_STDIN:
        return ((int)(len | 0x80000000U));
    case FH_STDOUT:
        return (-1);
    case FH_STDERR:
        return (-1);
    }

    return (-1);
}

__attribute__((weak)) int _sys_istty(FILEHANDLE fh)
{

    switch (fh)
    {
    case FH_STDIN:
        return (1);
    case FH_STDOUT:
        return (1);
    case FH_STDERR:
        return (1);
    }

    return (0);
}

__attribute__((weak)) int _sys_seek(FILEHANDLE fh, long pos)
{
    (void)pos;

    switch (fh)
    {
    case FH_STDIN:
        return (-1);
    case FH_STDOUT:
        return (-1);
    case FH_STDERR:
        return (-1);
    }

    return (-1);
}

__attribute__((weak)) long _sys_flen(FILEHANDLE fh)
{

    switch (fh)
    {
    case FH_STDIN:
        return (0);
    case FH_STDOUT:
        return (0);
    case FH_STDERR:
        return (0);
    }

    return (0);
}

__attribute__((weak)) char *(_sys_command_string)(char *cmd, int len)
{
    (void)len;

    return cmd;
}

__attribute__((weak)) void(_sys_exit)(int return_code) { exit(return_code); }

#else
/**
   Copied from CMSIS/DSP/DSP_Lib_TestSuite/Common/platform/GCC/Retarget.c
*/

int _open(const char *path, int flags, ...) { return (-1); }

int _close(int fd) { return (-1); }

int _lseek(int fd, int ptr, int dir) { return (0); }

int __attribute__((weak)) _fstat(int fd, struct stat *st)
{
    memset(st, 0, sizeof(*st));
    st->st_mode = S_IFCHR;
    return (0);
}

int _isatty(int fd) { return (1); }

int _read(int fd, char *ptr, int len)
{
    char c;
    int i;

    for (i = 0; i < len; i++)
    {
        c = SER_GetChar();
        if (c == 0x0D)
            break;
        *ptr++ = c;
        SER_PutChar(c);
    }
    return (len - i);
}

int _write(int fd, char *ptr, int len)
{
    int i;

    for (i = 0; i < len; i++)
        SER_PutChar(*ptr++);
    return (i);
}
#endif
