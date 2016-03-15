/*
 *
 * Copyright (c) 2015, Infineon Technologies AG
 * All rights reserved.                        
 *                                             
 * Redistribution and use in source and binary forms, with or without modification,are permitted provided that the 
 * following conditions are met:   
 *                                                                              
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
 * disclaimer.                        
 * 
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
 * disclaimer in the documentation and/or other materials provided with the distribution.                       
 * 
 * Neither the name of the copyright holders nor the names of its contributors may be used to endorse or promote 
 * products derived from this software without specific prior written permission.                                           
 *                                                                              
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE  
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE  FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR  
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                                  
 *                                                                              
 * To improve the quality of the software, users are encouraged to share modifications, enhancements or bug fixes with 
 * Infineon Technologies AG dave@infineon.com).                                                          
 *
 */

/**
 * @file RTE_Device.h
 * @date 24 July, 2015
 * @version 1.0.1
 *
 * @brief RTE Device Configuration for Infineon XMC4200_Q48
 *
 * History
 *
 * Version 1.0.1 
 *  Fix pin assignment
 * Version 1.0.0 
 *  Initial version
 */

//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------

#ifndef __RTE_DEVICE_H
#define __RTE_DEVICE_H

#include "xmc_device.h"
#include "xmc4_gpio_map.h"
#include "xmc4_usic_map.h"

#define NO_FIFO 0
#define FIFO_SIZE_2 1
#define FIFO_SIZE_4 2
#define FIFO_SIZE_8 3
#define FIFO_SIZE_16 4
#define FIFO_SIZE_32 5
#define FIFO_SIZE_64 6

// <e> UART0 (Universal asynchronous receiver transmitter) [Driver_USART0]
// <i> Configuration settings for Driver_USART0 in component ::Drivers:UART
#define RTE_UART0                      0

//   <o> UART0_TX Pin <0=>P1_5  
#define RTE_UART0_TX_ID                0
#if    (RTE_UART0_TX_ID == 0)
#define RTE_UART0_TX_PORT              P1_5
#define RTE_UART0_TX_AF                P1_5_AF_U0C0_DOUT0
#else
#error "Invalid UART0_TX Pin Configuration!"
#endif

//   <o> UART0_RX Pin <0=>P1_5 <1=>P1_4  
#define RTE_UART0_RX_ID                1
#if    (RTE_UART0_RX_ID == 0)
#define RTE_UART0_RX_PORT              P1_5
#define RTE_UART0_RX_INPUT             USIC0_C0_DX0_P1_5
#elif  (RTE_UART0_RX_ID == 1)
#define RTE_UART0_RX_PORT              P1_4
#define RTE_UART0_RX_INPUT             USIC0_C0_DX0_P1_4
#else
#error "Invalid UART0_RX Pin Configuration!"
#endif

//   <o> UART0_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_UART0_RX_FIFO_SIZE_ID                5
#if    (RTE_UART0_RX_FIFO_SIZE_ID == 0)
#define RTE_UART0_RX_FIFO_SIZE         NO_FIFO
#define RTE_UART0_RX_FIFO_SIZE_NUM     0
#elif  (RTE_UART0_RX_FIFO_SIZE_ID == 1)
#define RTE_UART0_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART0_RX_FIFO_SIZE_NUM     2
#elif  (RTE_UART0_RX_FIFO_SIZE_ID == 2)
#define RTE_UART0_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART0_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART0_RX_FIFO_SIZE_ID == 3)
#define RTE_UART0_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART0_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART0_RX_FIFO_SIZE_ID == 4)
#define RTE_UART0_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART0_RX_FIFO_SIZE_NUM     16
#elif  (RTE_UART0_RX_FIFO_SIZE_ID == 5)
#define RTE_UART0_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART0_RX_FIFO_SIZE_NUM     32
#elif  (RTE_UART0_RX_FIFO_SIZE_ID == 6)
#define RTE_UART0_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART0_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART0_RX FIFO SIZE Configuration!"
#endif

//   <o> UART0_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_UART0_TX_FIFO_SIZE_ID                5
#if    (RTE_UART0_TX_FIFO_SIZE_ID == 0)
#define RTE_UART0_TX_FIFO_SIZE         NO_FIFO
#define RTE_UART0_TX_FIFO_SIZE_NUM     0
#elif  (RTE_UART0_TX_FIFO_SIZE_ID == 1)
#define RTE_UART0_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART0_TX_FIFO_SIZE_NUM     2
#elif  (RTE_UART0_TX_FIFO_SIZE_ID == 2)
#define RTE_UART0_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART0_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART0_TX_FIFO_SIZE_ID == 3)
#define RTE_UART0_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART0_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART0_TX_FIFO_SIZE_ID == 4)
#define RTE_UART0_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART0_TX_FIFO_SIZE_NUM     16
#elif  (RTE_UART0_TX_FIFO_SIZE_ID == 5)
#define RTE_UART0_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART0_TX_FIFO_SIZE_NUM     32
#elif  (RTE_UART0_TX_FIFO_SIZE_ID == 6)
#define RTE_UART0_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART0_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART0_TX FIFO SIZE Configuration!"
#endif
//</e>

// <e> UART1 (Universal asynchronous receiver transmitter) [Driver_USART1]
// <i> Configuration settings for Driver_USART1 in component ::Drivers:UART
#define RTE_UART1                      0

//   <o> UART1_TX Pin <0=>P2_5    
#define RTE_UART1_TX_ID                0
#if    (RTE_UART1_TX_ID == 0)
#define RTE_UART1_TX_PORT              P2_5
#define RTE_UART1_TX_AF                P2_5_AF_U0C1_DOUT0
#else
#error "Invalid UART1_TX Pin Configuration!"
#endif

//   <o> UART1_RX Pin <0=>P2_2 <1=>P2_5 
#define RTE_UART1_RX_ID                1
#if    (RTE_UART1_RX_ID == 0)
#define RTE_UART1_RX_PORT              P2_2
#define RTE_UART1_RX_INPUT             USIC0_C1_DX0_P2_2
#elif  (RTE_UART1_RX_ID == 1)
#define RTE_UART1_RX_PORT              P2_5
#define RTE_UART1_RX_INPUT             USIC0_C1_DX0_P2_5
#else
#error "Invalid UART1_RX Pin Configuration!"
#endif

//   <o> UART1_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_UART1_RX_FIFO_SIZE_ID                2
#if    (RTE_UART1_RX_FIFO_SIZE_ID == 0)
#define RTE_UART1_RX_FIFO_SIZE         NO_FIFO
#define RTE_UART1_RX_FIFO_SIZE_NUM     0
#elif  (RTE_UART1_RX_FIFO_SIZE_ID == 1)
#define RTE_UART1_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART1_RX_FIFO_SIZE_NUM     2
#elif  (RTE_UART1_RX_FIFO_SIZE_ID == 2)
#define RTE_UART1_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART1_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART1_RX_FIFO_SIZE_ID == 3)
#define RTE_UART1_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART1_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART1_RX_FIFO_SIZE_ID == 4)
#define RTE_UART1_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART1_RX_FIFO_SIZE_NUM     16
#elif  (RTE_UART1_RX_FIFO_SIZE_ID == 5)
#define RTE_UART1_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART1_RX_FIFO_SIZE_NUM     32
#elif  (RTE_UART1_RX_FIFO_SIZE_ID == 6)
#define RTE_UART1_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART1_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART1_RX FIFO SIZE Configuration!"
#endif

//   <o> UART1_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_UART1_TX_FIFO_SIZE_ID                2
#if    (RTE_UART1_TX_FIFO_SIZE_ID == 0)
#define RTE_UART1_TX_FIFO_SIZE         NO_FIFO
#define RTE_UART1_TX_FIFO_SIZE_NUM     0
#elif  (RTE_UART1_TX_FIFO_SIZE_ID == 1)
#define RTE_UART1_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART1_TX_FIFO_SIZE_NUM     2
#elif  (RTE_UART1_TX_FIFO_SIZE_ID == 2)
#define RTE_UART1_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART1_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART1_TX_FIFO_SIZE_ID == 3)
#define RTE_UART1_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART1_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART1_TX_FIFO_SIZE_ID == 4)
#define RTE_UART1_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART1_TX_FIFO_SIZE_NUM     16
#elif  (RTE_UART1_TX_FIFO_SIZE_ID == 5)
#define RTE_UART1_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART1_TX_FIFO_SIZE_NUM     32
#elif  (RTE_UART1_TX_FIFO_SIZE_ID == 6)
#define RTE_UART1_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART1_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART1_TX FIFO SIZE Configuration!"
#endif
//</e>
// <e> UART2 (Universal asynchronous receiver transmitter) [Driver_USART2]
// <i> Configuration settings for Driver_USART2 in component ::Drivers:UART
#define RTE_UART2                      1

//   <o> UART2_TX Pin <0=>P0_5
#define RTE_UART2_TX_ID                0
#if    (RTE_UART2_TX_ID == 0)
#define RTE_UART2_TX_PORT              P0_5
#define RTE_UART2_TX_AF                P0_5_AF_U1C0_DOUT0
#else
#error "Invalid UART2_TX Pin Configuration!"
#endif

//   <o> UART2_RX Pin <0=> P0_4 <1=> P0_5 
#define RTE_UART2_RX_ID                0
#if    (RTE_UART2_RX_ID == 0)
#define RTE_UART2_RX_PORT              P0_4
#define RTE_UART2_RX_INPUT             USIC1_C0_DX0_P0_4
#elif  (RTE_UART2_RX_ID == 1)
#define RTE_UART2_RX_PORT              P0_5
#define RTE_UART2_RX_INPUT             USIC1_C0_DX0_P0_5
#else
#error "Invalid UART2_RX Pin Configuration!"
#endif

//   <o> UART2_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_UART2_RX_FIFO_SIZE_ID                0
#if    (RTE_UART2_RX_FIFO_SIZE_ID == 0)
#define RTE_UART2_RX_FIFO_SIZE         NO_FIFO
#define RTE_UART2_RX_FIFO_SIZE_NUM     0
#elif  (RTE_UART2_RX_FIFO_SIZE_ID == 1)
#define RTE_UART2_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART2_RX_FIFO_SIZE_NUM     2
#elif  (RTE_UART2_RX_FIFO_SIZE_ID == 2)
#define RTE_UART2_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART2_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART2_RX_FIFO_SIZE_ID == 3)
#define RTE_UART2_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART2_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART2_RX_FIFO_SIZE_ID == 4)
#define RTE_UART2_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART2_RX_FIFO_SIZE_NUM     16
#elif  (RTE_UART2_RX_FIFO_SIZE_ID == 5)
#define RTE_UART2_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART2_RX_FIFO_SIZE_NUM     32
#elif  (RTE_UART2_RX_FIFO_SIZE_ID == 6)
#define RTE_UART2_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART2_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART2_RX FIFO SIZE Configuration!"
#endif

//   <o> UART2_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_UART2_TX_FIFO_SIZE_ID                0
#if    (RTE_UART2_TX_FIFO_SIZE_ID == 0)
#define RTE_UART2_TX_FIFO_SIZE         NO_FIFO
#define RTE_UART2_TX_FIFO_SIZE_NUM     0
#elif  (RTE_UART2_TX_FIFO_SIZE_ID == 1)
#define RTE_UART2_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART2_TX_FIFO_SIZE_NUM     2
#elif  (RTE_UART2_TX_FIFO_SIZE_ID == 2)
#define RTE_UART2_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART2_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART2_TX_FIFO_SIZE_ID == 3)
#define RTE_UART2_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART2_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART2_TX_FIFO_SIZE_ID == 4)
#define RTE_UART2_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART2_TX_FIFO_SIZE_NUM     16
#elif  (RTE_UART2_TX_FIFO_SIZE_ID == 5)
#define RTE_UART2_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART2_TX_FIFO_SIZE_NUM     32
#elif  (RTE_UART2_TX_FIFO_SIZE_ID == 6)
#define RTE_UART2_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART2_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART2_TX FIFO SIZE Configuration!"
#endif
//</e>
// <e> UART3 (Universal asynchronous receiver transmitter) [Driver_USART3]
// <i> Configuration settings for Driver_USART3 in component ::Drivers:UART
#define RTE_UART3                      0

//   <o> UART3_TX Pin <0=>P0_1 
#define RTE_UART3_TX_ID                0
#if    (RTE_UART3_TX_ID == 0)
#define RTE_UART3_TX_PORT              P0_1
#define RTE_UART3_TX_AF                P0_1_AF_U1C1_DOUT0
#else
#error "Invalid UART3_TX Pin Configuration!"
#endif

//   <o> UART3_RX Pin <0=>P0_1  
#define RTE_UART3_RX_ID                0
#if    (RTE_UART3_RX_ID == 0)
#define RTE_UART3_RX_PORT              P0_0
#define RTE_UART3_RX_INPUT             USIC1_C1_DX0_P0_0
#else
#error "Invalid UART3_RX Pin Configuration!"
#endif

//   <o> UART3_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_UART3_RX_FIFO_SIZE_ID                2
#if    (RTE_UART3_RX_FIFO_SIZE_ID == 0)
#define RTE_UART3_RX_FIFO_SIZE         NO_FIFO
#define RTE_UART3_RX_FIFO_SIZE_NUM     0
#elif  (RTE_UART3_RX_FIFO_SIZE_ID == 1)
#define RTE_UART3_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART3_RX_FIFO_SIZE_NUM     2
#elif  (RTE_UART3_RX_FIFO_SIZE_ID == 2)
#define RTE_UART3_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART3_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART3_RX_FIFO_SIZE_ID == 3)
#define RTE_UART3_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART3_RX_FIFO_SIZE_NUM     4
#elif  (RTE_UART3_RX_FIFO_SIZE_ID == 4)
#define RTE_UART3_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART3_RX_FIFO_SIZE_NUM     16
#elif  (RTE_UART3_RX_FIFO_SIZE_ID == 5)
#define RTE_UART3_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART3_RX_FIFO_SIZE_NUM     32
#elif  (RTE_UART3_RX_FIFO_SIZE_ID == 6)
#define RTE_UART3_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART3_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART3_RX FIFO SIZE Configuration!"
#endif

//   <o> UART3_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_UART3_TX_FIFO_SIZE_ID                2
#if    (RTE_UART3_TX_FIFO_SIZE_ID == 0)
#define RTE_UART3_TX_FIFO_SIZE         NO_FIFO
#define RTE_UART3_TX_FIFO_SIZE_NUM     0
#elif  (RTE_UART3_TX_FIFO_SIZE_ID == 1)
#define RTE_UART3_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_UART3_TX_FIFO_SIZE_NUM     2
#elif  (RTE_UART3_TX_FIFO_SIZE_ID == 2)
#define RTE_UART3_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART3_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART3_TX_FIFO_SIZE_ID == 3)
#define RTE_UART3_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_UART3_TX_FIFO_SIZE_NUM     4
#elif  (RTE_UART3_TX_FIFO_SIZE_ID == 4)
#define RTE_UART3_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_UART3_TX_FIFO_SIZE_NUM     16
#elif  (RTE_UART3_TX_FIFO_SIZE_ID == 5)
#define RTE_UART3_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_UART3_TX_FIFO_SIZE_NUM     32
#elif  (RTE_UART3_TX_FIFO_SIZE_ID == 6)
#define RTE_UART3_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_UART3_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid UART3_TX FIFO SIZE Configuration!"
#endif


// </e>
// <e> SPI0 (Serial peripheral interface) [Driver_SPI0]
// <i> Configuration settings for Driver_SPI0 in component ::Drivers:SPI
#define RTE_SPI0                      0

//   <o> SPI0 TX: MOSI(master) MISO(slave) Pin <0=>P1_5   
#define RTE_SPI0_TX_ID                0
#if    (RTE_SPI0_TX_ID == 0)
#define RTE_SPI0_TX_PORT              P1_5
#define RTE_SPI0_TX_AF                P1_5_AF_U0C0_DOUT0
#else
#error "Invalid SPI0_TX Pin Configuration!"
#endif

//   <o> SPI0 RX MISO(master) MOSI(slave) Pin <0=>P1_5  
#define RTE_SPI0_RX_ID                0
#if    (RTE_SPI0_RX_ID == 0)
#define RTE_SPI0_RX_PORT              P1_5
#define RTE_SPI0_RX_INPUT             USIC0_C0_DX0_P1_5
#else
#error "Invalid SPI0_RX Pin Configuration!"
#endif

//   <o> SPI0_CLK OUTPUT Pin <0=>P0_8 <1=>P1_1 
#define RTE_SPI0_CLK_OUTPUT_ID                0
#if    (RTE_SPI0_CLK_OUTPUT_ID == 0)
#define RTE_SPI0_CLK_OUTPUT_PORT              P0_8
#define RTE_SPI0_CLK_AF                       P0_8_AF_U0C0_SCLKOUT 
#elif  (RTE_SPI0_CLK_OUTPUT_ID == 1)
#define RTE_SPI0_CLK_OUTPUT_PORT              P1_1
#define RTE_SPI0_CLK_AF                       P1_1_AF_U0C0_SCLKOUT
#else
#error "Invalid SPI0 CLOCK OUTPUT Pin Configuration!"
#endif
//   <h> SPI0_SLAVE SELECT Pins 
// <e> SLAVE SELECT LINE 0
// <i> Enable slave select line 0
#define RTE_SPI0_SLAVE_SELECT_LINE0 1
//   <o> SPI0_SLAVE SELECT LINE 0 Pin <0=>P0_7 <1=>P1_0 
#define RTE_SPI0_SLAVE_SELECT_LINE_0_ID                1
#if    (RTE_SPI0_SLAVE_SELECT_LINE_0_ID == 0)
#define RTE_SPI0_SLAVE_SELECT_LINE_0_PORT              P0_7
#define RTE_SPI0_SLAVE_SELECT_LINE_0_AF                P0_7_AF_U0C0_SELO0
#elif  (RTE_SPI0_SLAVE_SELECT_LINE_0_ID == 1)
#define RTE_SPI0_SLAVE_SELECT_LINE_0_PORT              P1_0
#define RTE_SPI0_SLAVE_SELECT_LINE_0_AF                P1_0_AF_U0C0_SELO0
#else
#error "Invalid SPI0 SLAVE SELECT LINE 0 Pin Configuration!"
#endif
// </e>
// <e> SLAVE SELECT LINE 1
// <i> Enable slave select line 1
#define RTE_SPI0_SLAVE_SELECT_LINE1 1 
//   <o> SPI0_SLAVE SELECT LINE 1 Pin <0=>P1_8
#define RTE_SPI0_SLAVE_SELECT_LINE_1_ID                0
#if    (RTE_SPI0_SLAVE_SELECT_LINE_1_ID == 0)
#define RTE_SPI0_SLAVE_SELECT_LINE_1_PORT              P1_8
#define RTE_SPI0_SLAVE_SELECT_LINE_1_AF                P1_8_AF_U0C0_SELO1
#else
#error "Invalid SPI0 SLAVE SELECT LINE 1 Pin Configuration!"
#endif

// </e>
// <e> SLAVE SELECT LINE 2
// <i> Enable slave select line 2
#define RTE_SPI0_SLAVE_SELECT_LINE2 0 
#if    (RTE_SPI0_SLAVE_SELECT_LINE2 == 1)
#error "Invalid SPI0 SLAVE SELECT LINE 2 Pin Configuration!"
#endif
//</e>
// <e> SLAVE SELECT LINE 3
// <i> Enable slave select line 3
#define RTE_SPI0_SLAVE_SELECT_LINE3 0 
#if    (RTE_SPI0_SLAVE_SELECT_LINE3 == 1)
#error "Invalid SPI0 SLAVE SELECT LINE 3 Pin Configuration!"
#endif
//</e>
// <e> SLAVE SELECT LINE 4
// <i> Enable slave select line 4
#define RTE_SPI0_SLAVE_SELECT_LINE4 0 
#if    (RTE_SPI0_SLAVE_SELECT_LINE4 == 1)
#error "Invalid SPI0 SLAVE SELECT LINE 4 Pin Configuration!"
#endif
//</e>
// <e> SLAVE SELECT LINE 5
// <i> Enable slave select line 5
#define RTE_SPI0_SLAVE_SELECT_LINE5 0 
#if    (RTE_SPI0_SLAVE_SELECT_LINE5 == 1)
#error "Invalid SPI0 SLAVE SELECT LINE 5 Pin Configuration!"
#endif
//</e>
// </h>
//   <o> SPI0_CLK INPUT Pin <0=>P1_1 <1=>P0_8 
#define RTE_SPI0_CLK_INPUT_ID                0
#if    (RTE_SPI0_CLK_INPUT_ID == 0)
#define RTE_SPI0_CLK_INPUT_PORT              P1_1
#define RTE_SPI0_CLK_INPUT                   USIC0_C0_DX1_P1_1
#elif  (RTE_SPI0_CLK_INPUT_ID == 1)
#define RTE_SPI0_CLK_INPUT_PORT              P0_8
#define RTE_SPI0_CLK_INPUT                   USIC0_C0_DX1_P0_8
#else
#error "Invalid SPI0 CLOCK INPUT Pin Configuration!"
#endif
//   <o> RTE_SPI0_SLAVE_SELECT INPUT Pin <0=>P1_0 
#define RTE_SPI0_SLAVE_SELECT_INPUT_ID                0
#if    (RTE_SPI0_SLAVE_SELECT_INPUT_ID == 0)
#define RTE_SPI0_SLAVE_SELECT_INPUT_PORT              P1_0
#define RTE_SPI0_SLAVE_SELECT_INPUT                   USIC0_C0_DX2_P1_0
#else
#error "Invalid SPI0 SLAVE SELECT INPUT Pin Configuration!"
#endif

//   <o> SPI0_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_SPI0_RX_FIFO_SIZE_ID                2
#if    (RTE_SPI0_RX_FIFO_SIZE_ID == 0)
#define RTE_SPI0_RX_FIFO_SIZE         NO_FIFO
#define RTE_SPI0_RX_FIFO_SIZE_NUM     0
#elif  (RTE_SPI0_RX_FIFO_SIZE_ID == 1)
#define RTE_SPI0_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_SPI0_RX_FIFO_SIZE_NUM     2
#elif  (RTE_SPI0_RX_FIFO_SIZE_ID == 2)
#define RTE_SPI0_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI0_RX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI0_RX_FIFO_SIZE_ID == 3)
#define RTE_SPI0_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI0_RX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI0_RX_FIFO_SIZE_ID == 4)
#define RTE_SPI0_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_SPI0_RX_FIFO_SIZE_NUM     16
#elif  (RTE_SPI0_RX_FIFO_SIZE_ID == 5)
#define RTE_SPI0_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_SPI0_RX_FIFO_SIZE_NUM     32
#elif  (RTE_SPI0_RX_FIFO_SIZE_ID == 6)
#define RTE_SPI0_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_SPI0_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid SPI0_RX FIFO SIZE Configuration!"
#endif

//   <o> SPI0_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_SPI0_TX_FIFO_SIZE_ID                4
#if    (RTE_SPI0_TX_FIFO_SIZE_ID == 0)
#define RTE_SPI0_TX_FIFO_SIZE         NO_FIFO
#define RTE_SPI0_TX_FIFO_SIZE_NUM     0
#elif  (RTE_SPI0_TX_FIFO_SIZE_ID == 1)
#define RTE_SPI0_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_SPI0_TX_FIFO_SIZE_NUM     2
#elif  (RTE_SPI0_TX_FIFO_SIZE_ID == 2)
#define RTE_SPI0_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI0_TX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI0_TX_FIFO_SIZE_ID == 3)
#define RTE_SPI0_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI0_TX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI0_TX_FIFO_SIZE_ID == 4)
#define RTE_SPI0_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_SPI0_TX_FIFO_SIZE_NUM     16
#elif  (RTE_SPI0_TX_FIFO_SIZE_ID == 5)
#define RTE_SPI0_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_SPI0_TX_FIFO_SIZE_NUM     32
#elif  (RTE_SPI0_TX_FIFO_SIZE_ID == 6)
#define RTE_SPI0_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_SPI0_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid SPI0_TX FIFO SIZE Configuration!"
#endif
//</e>

// <e> SPI1 (Serial peripheral interface) [Driver_SPI1]
// <i> Configuration settings for Driver_SPI1 in component ::Drivers:SPI
#define RTE_SPI1                      0

//   <o> SPI1 TX MOSI(master) MISO(slave) Pin <0=>P2_5   
#define RTE_SPI1_TX_ID                0
#if    (RTE_SPI1_TX_ID == 0)
#define RTE_SPI1_TX_PORT              P2_5
#define RTE_SPI1_TX_AF                P2_5_AF_U0C1_DOUT0
#else
#error "Invalid SPI1_TX Pin Configuration!"
#endif

//   <o> SPI1 RX MISO(master) MOSI(slave) Pin <0=>P2_2 <1=>P2_5
#define RTE_SPI1_RX_ID                1
#if    (RTE_SPI1_RX_ID == 0)
#define RTE_SPI1_RX_PORT              P2_2
#define RTE_SPI1_RX_INPUT             USIC0_C1_DX0_P2_2
#elif  (RTE_SPI1_RX_ID == 1)
#define RTE_SPI1_RX_PORT              P2_5
#define RTE_SPI1_RX_INPUT             USIC0_C1_DX0_P2_5
#else
#error "Invalid SPI1_RX Pin Configuration!"
#endif


//   <o> SPI1_CLK OUTPUT Pin <0=>P2_4 
#define RTE_SPI1_CLK_OUTPUT_ID                0
#if    (RTE_SPI1_CLK_OUTPUT_ID == 0)
#define RTE_SPI1_CLK_OUTPUT_PORT              P2_4
#define RTE_SPI1_CLK_AF                       P2_4_AF_U0C1_SCLKOUT
#else
#error "Invalid SPI1 CLOCK OUTPUT Pin Configuration!"
#endif

// <h> SPI1_SLAVE SELECT Pins
// <e> SLAVE SELECT LINE 0
// <i> Enable slave select line 0
#define RTE_SPI1_SLAVE_SELECT_LINE0 1
//   <o> SPI1_SLAVE SELECT LINE 0 Pin <0=>P2_3
#define RTE_SPI1_SLAVE_SELECT_LINE_0_ID                0
#if    (RTE_SPI1_SLAVE_SELECT_LINE_0_ID == 0)
#define RTE_SPI1_SLAVE_SELECT_LINE_0_PORT              P2_3
#define RTE_SPI1_SLAVE_SELECT_LINE_0_AF                P2_3_AF_U0C1_SELO0
#else
#error "Invalid SPI1 SLAVE SELECT LINE 0 Pin Configuration!"
#endif
// </e>
// <e> SLAVE SELECT LINE 1
// <i> Enable slave select line 1
#define RTE_SPI1_SLAVE_SELECT_LINE1 0 
#if    (RTE_SPI1_SLAVE_SELECT_LINE1 == 1)
#error "Invalid SPI1 SLAVE SELECT LINE 1 Pin Configuration!"
#endif
//</e>
// <e> SLAVE SELECT LINE 2
// <i> Enable slave select line 2
#define RTE_SPI1_SLAVE_SELECT_LINE2 0 
#if    (RTE_SPI1_SLAVE_SELECT_LINE2 == 1)
#error "Invalid SPI1 SLAVE SELECT LINE 2 Pin Configuration!"
#endif
//</e>
// <e> SLAVE SELECT LINE 3
// <i> Enable slave select line 3
#define RTE_SPI1_SLAVE_SELECT_LINE3 0 
#if    (RTE_SPI1_SLAVE_SELECT_LINE3 == 1)
#error "Invalid SPI1 SLAVE SELECT LINE 3 Pin Configuration!"
#endif
//</e>
// </h>

//   <o> SPI1_CLK INPUT Pin <0=>P2_4 
#define RTE_SPI1_CLK_INPUT_ID                0
#if    (RTE_SPI1_CLK_INPUT_ID == 0)
#define RTE_SPI1_CLK_INPUT_PORT              P2_4
#define RTE_SPI1_CLK_INPUT                   USIC0_C1_DX1_P2_4
#else
#error "Invalid SPI1 CLOCK INPUT Pin Configuration!"
#endif

//   <o> RTE_SPI1_SLAVE_SELECT INPUT Pin <0=>P2_3
#define RTE_SPI1_SLAVE_SELECT_INPUT_ID                0
#if    (RTE_SPI1_SLAVE_SELECT_INPUT_ID == 0)
#define RTE_SPI1_SLAVE_SELECT_INPUT_PORT              P2_3
#define RTE_SPI1_SLAVE_SELECT_INPUT                   USIC0_C1_DX2_P2_3
#else
#error "Invalid SPI1 SLAVE SELECT INPUT Pin Configuration!"
#endif

//   <o> SPI1_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_SPI1_RX_FIFO_SIZE_ID                2
#if    (RTE_SPI1_RX_FIFO_SIZE_ID == 0)
#define RTE_SPI1_RX_FIFO_SIZE         NO_FIFO
#define RTE_SPI1_RX_FIFO_SIZE_NUM     0
#elif  (RTE_SPI1_RX_FIFO_SIZE_ID == 1)
#define RTE_SPI1_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_SPI1_RX_FIFO_SIZE_NUM     2
#elif  (RTE_SPI1_RX_FIFO_SIZE_ID == 2)
#define RTE_SPI1_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI1_RX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI1_RX_FIFO_SIZE_ID == 3)
#define RTE_SPI1_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI1_RX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI1_RX_FIFO_SIZE_ID == 4)
#define RTE_SPI1_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_SPI1_RX_FIFO_SIZE_NUM     16
#elif  (RTE_SPI1_RX_FIFO_SIZE_ID == 5)
#define RTE_SPI1_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_SPI1_RX_FIFO_SIZE_NUM     32
#elif  (RTE_SPI1_RX_FIFO_SIZE_ID == 6)
#define RTE_SPI1_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_SPI1_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid SPI1_RX FIFO SIZE Configuration!"
#endif

//   <o> SPI1_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_SPI1_TX_FIFO_SIZE_ID                2
#if    (RTE_SPI1_TX_FIFO_SIZE_ID == 0)
#define RTE_SPI1_TX_FIFO_SIZE         NO_FIFO
#define RTE_SPI1_TX_FIFO_SIZE_NUM     0
#elif  (RTE_SPI1_TX_FIFO_SIZE_ID == 1)
#define RTE_SPI1_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_SPI1_TX_FIFO_SIZE_NUM     2
#elif  (RTE_SPI1_TX_FIFO_SIZE_ID == 2)
#define RTE_SPI1_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI1_TX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI1_TX_FIFO_SIZE_ID == 3)
#define RTE_SPI1_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_SPI1_TX_FIFO_SIZE_NUM     4
#elif  (RTE_SPI1_TX_FIFO_SIZE_ID == 4)
#define RTE_SPI1_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_SPI1_TX_FIFO_SIZE_NUM     16
#elif  (RTE_SPI1_TX_FIFO_SIZE_ID == 5)
#define RTE_SPI1_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_SPI1_TX_FIFO_SIZE_NUM     32
#elif  (RTE_SPI1_TX_FIFO_SIZE_ID == 6)
#define RTE_SPI1_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_SPI1_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid SPI1_TX FIFO SIZE Configuration!"
#endif
//</e>
// <e> I2C0 (Inter-Integrated circuit) [Driver_I2C0]
// <i> Configuration settings for Driver_I2C0 in component ::Drivers:I2C
#define RTE_I2C0                      0

//   <o> I2C0_TX Pin <0=>P1_5 
#define RTE_I2C0_TX_ID                0
#if    (RTE_I2C0_TX_ID == 0)
#define RTE_I2C0_TX_PORT              P1_5
#define RTE_I2C0_TX_AF                P1_5_AF_U0C0_DOUT0
#else
#error "Invalid I2C0_TX Pin Configuration!"
#endif

//   <o> I2C0_RX Pin <0=>P1_5  
#define RTE_I2C0_RX_ID                0
#if    (RTE_I2C0_RX_ID == 0)
#define RTE_I2C0_RX_PORT              P1_5
#define RTE_I2C0_RX_INPUT             USIC0_C0_DX0_P1_5
#else
#error "Invalid I2C0_RX Pin Configuration!"
#endif
//   <o> I2C0_CLK OUTPUT Pin <0=>P0_8 <1=>P1_1
#define RTE_I2C0_CLK_OUTPUT_ID                1
#if    (RTE_I2C0_CLK_OUTPUT_ID == 0)
#define RTE_I2C0_CLK_OUTPUT_PORT              P0_8
#define RTE_I2C0_CLK_AF                       P0_8_AF_U0C0_SCLKOUT 
#elif  (RTE_I2C0_CLK_OUTPUT_ID == 1)
#define RTE_I2C0_CLK_OUTPUT_PORT              P1_1
#define RTE_I2C0_CLK_AF                       P1_1_AF_U0C0_SCLKOUT
#else
#error "Invalid I2C0 CLOCK OUTPUT Pin Configuration!"
#endif
//   <o> I2C0_CLK INPUT Pin <0=>P1_1 <1=>P0_8 
#define RTE_I2C0_CLK_INPUT_ID                1
#if    (RTE_I2C0_CLK_INPUT_ID == 0)
#define RTE_I2C0_CLK_INPUT_PORT              P1_1
#define RTE_I2C0_CLK_INPUT                   USIC0_C0_DX1_P1_1
#elif  (RTE_I2C0_CLK_INPUT_ID == 1)
#define RTE_I2C0_CLK_INPUT_PORT              P0_8
#define RTE_I2C0_CLK_INPUT                   USIC0_C0_DX1_P0_8
#else
#error "Invalid I2C0 CLOCK INPUT Pin Configuration!"
#endif

//   <o> I2C0_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_I2C0_RX_FIFO_SIZE_ID                2
#if    (RTE_I2C0_RX_FIFO_SIZE_ID == 0)
#define RTE_I2C0_RX_FIFO_SIZE         NO_FIFO
#define RTE_I2C0_RX_FIFO_SIZE_NUM     0
#elif  (RTE_I2C0_RX_FIFO_SIZE_ID == 1)
#define RTE_I2C0_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_I2C0_RX_FIFO_SIZE_NUM     2
#elif  (RTE_I2C0_RX_FIFO_SIZE_ID == 2)
#define RTE_I2C0_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C0_RX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C0_RX_FIFO_SIZE_ID == 3)
#define RTE_I2C0_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C0_RX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C0_RX_FIFO_SIZE_ID == 4)
#define RTE_I2C0_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_I2C0_RX_FIFO_SIZE_NUM     16
#elif  (RTE_I2C0_RX_FIFO_SIZE_ID == 5)
#define RTE_I2C0_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_I2C0_RX_FIFO_SIZE_NUM     32
#elif  (RTE_I2C0_RX_FIFO_SIZE_ID == 6)
#define RTE_I2C0_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_I2C0_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid I2C0_RX FIFO SIZE Configuration!"
#endif

//   <o> I2C0_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_I2C0_TX_FIFO_SIZE_ID                2
#if    (RTE_I2C0_TX_FIFO_SIZE_ID == 0)
#define RTE_I2C0_TX_FIFO_SIZE         NO_FIFO
#define RTE_I2C0_TX_FIFO_SIZE_NUM     0
#elif  (RTE_I2C0_TX_FIFO_SIZE_ID == 1)
#define RTE_I2C0_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_I2C0_TX_FIFO_SIZE_NUM     2
#elif  (RTE_I2C0_TX_FIFO_SIZE_ID == 2)
#define RTE_I2C0_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C0_TX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C0_TX_FIFO_SIZE_ID == 3)
#define RTE_I2C0_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C0_TX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C0_TX_FIFO_SIZE_ID == 4)
#define RTE_I2C0_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_I2C0_TX_FIFO_SIZE_NUM     16
#elif  (RTE_I2C0_TX_FIFO_SIZE_ID == 5)
#define RTE_I2C0_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_I2C0_TX_FIFO_SIZE_NUM     32
#elif  (RTE_I2C0_TX_FIFO_SIZE_ID == 6)
#define RTE_I2C0_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_I2C0_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid I2C0_TX FIFO SIZE Configuration!"
#endif

//</e>

// <e> I2C1 (Inter-Integrated circuit) [Driver_I2C1]
// <i> Configuration settings for Driver_I2C1 in component ::Drivers:I2C
#define RTE_I2C1                      0

//   <o> I2C1_TX Pin <0=>P2_5   
#define RTE_I2C1_TX_ID                0
#if    (RTE_I2C1_TX_ID == 0)
#define RTE_I2C1_TX_PORT              P2_5
#define RTE_I2C1_TX_AF                P2_5_AF_U0C1_DOUT0
#else
#error "Invalid I2C1_TX Pin Configuration!"
#endif

//   <o> I2C1_RX Pin <0=>P2_2 <1=>P2_5 
#define RTE_I2C1_RX_ID                1
#if    (RTE_I2C1_RX_ID == 0)
#define RTE_I2C1_RX_PORT              P2_2
#define RTE_I2C1_RX_INPUT             USIC0_C1_DX0_P2_2
#elif  (RTE_I2C1_RX_ID == 1)
#define RTE_I2C1_RX_PORT              P2_5
#define RTE_I2C1_RX_INPUT             USIC0_C1_DX0_P2_5
#else
#error "Invalid I2C1_RX Pin Configuration!"
#endif

//   <o> I2C1_CLK OUTPUT Pin <0=>P2_4 
#define RTE_I2C1_CLK_OUTPUT_ID                0
#if    (RTE_I2C1_CLK_OUTPUT_ID == 0)
#define RTE_I2C1_CLK_OUTPUT_PORT              P2_4
#define RTE_I2C1_CLK_AF                       P2_4_AF_U0C1_SCLKOUT
#else
#error "Invalid I2C1 CLOCK OUTPUT Pin Configuration!"
#endif

//   <o> I2C1_CLK INPUT Pin <0=>P2_4 
#define RTE_I2C1_CLK_INPUT_ID                0
#if    (RTE_I2C1_CLK_INPUT_ID == 0)
#define RTE_I2C1_CLK_INPUT_PORT              P2_4
#define RTE_I2C1_CLK_INPUT                   USIC0_C1_DX1_P2_4
#else
#error "Invalid I2C1 CLOCK INPUT Pin Configuration!"
#endif

//   <o> I2C1_RX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64
#define RTE_I2C1_RX_FIFO_SIZE_ID                2
#if    (RTE_I2C1_RX_FIFO_SIZE_ID == 0)
#define RTE_I2C1_RX_FIFO_SIZE         NO_FIFO
#define RTE_I2C1_RX_FIFO_SIZE_NUM     0
#elif  (RTE_I2C1_RX_FIFO_SIZE_ID == 1)
#define RTE_I2C1_RX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_I2C1_RX_FIFO_SIZE_NUM     2
#elif  (RTE_I2C1_RX_FIFO_SIZE_ID == 2)
#define RTE_I2C1_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C1_RX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C1_RX_FIFO_SIZE_ID == 3)
#define RTE_I2C1_RX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C1_RX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C1_RX_FIFO_SIZE_ID == 4)
#define RTE_I2C1_RX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_I2C1_RX_FIFO_SIZE_NUM     16
#elif  (RTE_I2C1_RX_FIFO_SIZE_ID == 5)
#define RTE_I2C1_RX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_I2C1_RX_FIFO_SIZE_NUM     32
#elif  (RTE_I2C1_RX_FIFO_SIZE_ID == 6)
#define RTE_I2C1_RX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_I2C1_RX_FIFO_SIZE_NUM     64
#else
#error "Invalid I2C1_RX FIFO SIZE Configuration!"
#endif

//   <o> I2C1_TX FIFO ENTRIES <0=>0 <1=>2 <2=>4 <3=>8 <4=>16 <5=>32 <6=>64 
#define RTE_I2C1_TX_FIFO_SIZE_ID                2
#if    (RTE_I2C1_TX_FIFO_SIZE_ID == 0)
#define RTE_I2C1_TX_FIFO_SIZE         NO_FIFO
#define RTE_I2C1_TX_FIFO_SIZE_NUM     0
#elif  (RTE_I2C1_TX_FIFO_SIZE_ID == 1)
#define RTE_I2C1_TX_FIFO_SIZE         FIFO_SIZE_2
#define RTE_I2C1_TX_FIFO_SIZE_NUM     2
#elif  (RTE_I2C1_TX_FIFO_SIZE_ID == 2)
#define RTE_I2C1_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C1_TX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C1_TX_FIFO_SIZE_ID == 3)
#define RTE_I2C1_TX_FIFO_SIZE         FIFO_SIZE_4
#define RTE_I2C1_TX_FIFO_SIZE_NUM     4
#elif  (RTE_I2C1_TX_FIFO_SIZE_ID == 4)
#define RTE_I2C1_TX_FIFO_SIZE         FIFO_SIZE_16
#define RTE_I2C1_TX_FIFO_SIZE_NUM     16
#elif  (RTE_I2C1_TX_FIFO_SIZE_ID == 5)
#define RTE_I2C1_TX_FIFO_SIZE         FIFO_SIZE_32
#define RTE_I2C1_TX_FIFO_SIZE_NUM     32
#elif  (RTE_I2C1_TX_FIFO_SIZE_ID == 6)
#define RTE_I2C1_TX_FIFO_SIZE         FIFO_SIZE_64
#define RTE_I2C1_TX_FIFO_SIZE_NUM     64
#else
#error "Invalid I2C1_TX FIFO SIZE Configuration!"
#endif
//</e>

#if ((RTE_UART0+RTE_I2C0+RTE_SPI0)>1)
#error "Choose just one Driver_I2C/SPI/UART0 driver !"
#elif ((RTE_UART1+RTE_I2C1+RTE_SPI1)>1)
#error "Choose just one Driver_I2C/SPI/UART1 driver !"
#elif ((RTE_UART2+RTE_I2C2+RTE_SPI2)>1)
#error "Choose just one Driver_I2C/SPI/UART2 driver !"
#elif ((RTE_UART3+RTE_I2C3+RTE_SPI3)>1)
#error "Choose just one Driver_I2C/SPI/UART3 driver !"
#endif


#endif  /* __RTE_DEVICE_H */
