/*
 * Copyright (c) 2019-2021 Arm Limited. All rights reserved.
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

#include "uart.h"
#include <stdint.h>
#include <stdio.h>

#define UART0_BASE 0x49303000
#define UART0_BAUDRATE 115200
#define SYSTEM_CORE_CLOCK 25000000

/*------------- Universal Asynchronous Receiver Transmitter (UART) -----------*/

#define __IO volatile
#define __I volatile const
#define __O volatile

typedef struct
{
    __IO uint32_t DATA;  /* Offset: 0x000 (R/W) Data Register    */
    __IO uint32_t STATE; /* Offset: 0x004 (R/W) Status Register  */
    __IO uint32_t CTRL;  /* Offset: 0x008 (R/W) Control Register */
    union
    {
        __I uint32_t INTSTATUS; /* Offset: 0x00C (R/ ) Interrupt Status Register */
        __O uint32_t INTCLEAR;  /* Offset: 0x00C ( /W) Interrupt Clear Register  */
    };
    __IO uint32_t BAUDDIV; /* Offset: 0x010 (R/W) Baudrate Divider Register */
} CMSDK_UART_TypeDef;

#define CMSDK_UART0_BASE UART0_BASE
#define CMSDK_UART0 ((CMSDK_UART_TypeDef *)CMSDK_UART0_BASE)
#define CMSDK_UART0_BAUDRATE UART0_BAUDRATE

void uart_init(void)
{
    // SystemCoreClock / 9600
    CMSDK_UART0->BAUDDIV = SYSTEM_CORE_CLOCK / CMSDK_UART0_BAUDRATE;

    CMSDK_UART0->CTRL = ((1ul << 0) | /* TX enable */
                         (1ul << 1)); /* RX enable */
}

// Output a character
unsigned char uart_putc(unsigned char my_ch)
{
    while ((CMSDK_UART0->STATE & 1))
        ; // Wait if Transmit Holding register is full

    if (my_ch == '\n')
    {
        CMSDK_UART0->DATA = '\r';
        while ((CMSDK_UART0->STATE & 1))
            ; // Wait if Transmit Holding register is full
    }

    CMSDK_UART0->DATA = my_ch; // write to transmit holding register

    return (my_ch);
}

// Get a character
unsigned char uart_getc(void)
{
    unsigned char my_ch;
    // unsigned int  cnt;

    while ((CMSDK_UART0->STATE & 2) == 0) // Wait if Receive Holding register is empty
    {
#if 0
        cnt = MPS3_FPGAIO->CLK100HZ / 50;
        if (cnt & 0x8)
            MPS3_FPGAIO->LED = 0x01 << (cnt & 0x7);
        else
            MPS3_FPGAIO->LED = 0x80 >> (cnt & 0x7);
#endif
    }

    my_ch = CMSDK_UART0->DATA;

    // Convert CR to LF
    if (my_ch == '\r')
        my_ch = '\n';

    return (my_ch);
}
