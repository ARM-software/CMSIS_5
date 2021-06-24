/*
 * Copyright (c) 2013-2021 ARM Limited. All rights reserved.
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
 *
 * ----------------------------------------------------------------------
 *
 * $Date:        16. June 2021
 * $Revision:    V2.1.0
 *
 * Project:      CMSIS-DAP Examples LPC-Link2
 * Title:        DAP_config.h CMSIS-DAP Configuration File for LPC-Link2
 *
 *---------------------------------------------------------------------------*/

#ifndef __DAP_CONFIG_H__
#define __DAP_CONFIG_H__


//**************************************************************************************************
/** 
\defgroup DAP_Config_Debug_gr CMSIS-DAP Debug Unit Information
\ingroup DAP_ConfigIO_gr 
@{
Provides definitions about the hardware and configuration of the Debug Unit.

This information includes:
 - Definition of Cortex-M processor parameters used in CMSIS-DAP Debug Unit.
 - Debug Unit Identification strings (Vendor, Product, Serial Number).
 - Debug Unit communication packet size.
 - Debug Access Port supported modes and settings (JTAG/SWD and SWO).
 - Optional information about a connected Target Device (for Evaluation Boards).
*/

#ifdef _RTE_
#include "RTE_Components.h"
#include CMSIS_device_header
#else
#include "device.h"                             // Debug Unit Cortex-M Processor Header File
#endif

/// Processor Clock of the Cortex-M MCU used in the Debug Unit.
/// This value is used to calculate the SWD/JTAG clock speed.
#define CPU_CLOCK               180000000U      ///< Specifies the CPU Clock in Hz.

/// Number of processor cycles for I/O Port write operations.
/// This value is used to calculate the SWD/JTAG clock speed that is generated with I/O
/// Port write operations in the Debug Unit by a Cortex-M MCU. Most Cortex-M processors
/// require 2 processor cycles for a I/O Port Write operation.  If the Debug Unit uses
/// a Cortex-M0+ processor with high-speed peripheral I/O only 1 processor cycle might be 
/// required.
#define IO_PORT_WRITE_CYCLES    2U              ///< I/O Cycles: 2=default, 1=Cortex-M0+ fast I/0.

/// Indicate that Serial Wire Debug (SWD) communication mode is available at the Debug Access Port.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define DAP_SWD                 1               ///< SWD Mode:  1 = available, 0 = not available.

/// Indicate that JTAG communication mode is available at the Debug Port.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define DAP_JTAG                1               ///< JTAG Mode: 1 = available, 0 = not available.

/// Configure maximum number of JTAG devices on the scan chain connected to the Debug Access Port.
/// This setting impacts the RAM requirements of the Debug Unit. Valid range is 1 .. 255.
#define DAP_JTAG_DEV_CNT        8U              ///< Maximum number of JTAG devices on scan chain.

/// Default communication mode on the Debug Access Port.
/// Used for the command \ref DAP_Connect when Port Default mode is selected.
#define DAP_DEFAULT_PORT        1U              ///< Default JTAG/SWJ Port Mode: 1 = SWD, 2 = JTAG.

/// Default communication speed on the Debug Access Port for SWD and JTAG mode.
/// Used to initialize the default SWD/JTAG clock frequency.
/// The command \ref DAP_SWJ_Clock can be used to overwrite this default setting.
#define DAP_DEFAULT_SWJ_CLOCK   1000000U        ///< Default SWD/JTAG clock frequency in Hz.

/// Maximum Package Size for Command and Response data.
/// This configuration settings is used to optimize the communication performance with the
/// debugger and depends on the USB peripheral. Typical vales are 64 for Full-speed USB HID or WinUSB,
/// 1024 for High-speed USB HID and 512 for High-speed USB WinUSB.
#define DAP_PACKET_SIZE         1024U           ///< Specifies Packet Size in bytes.

/// Maximum Package Buffers for Command and Response data.
/// This configuration settings is used to optimize the communication performance with the
/// debugger and depends on the USB peripheral. For devices with limited RAM or USB buffer the
/// setting can be reduced (valid range is 1 .. 255).
#define DAP_PACKET_COUNT        4U              ///< Specifies number of packets buffered.

/// Indicate that UART Serial Wire Output (SWO) trace is available.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define SWO_UART                1               ///< SWO UART:  1 = available, 0 = not available.

/// USART Driver instance number for the UART SWO.
#define SWO_UART_DRIVER         1               ///< USART Driver instance number (Driver_USART#).

/// Maximum SWO UART Baudrate.
#define SWO_UART_MAX_BAUDRATE   10000000U       ///< SWO UART Maximum Baudrate in Hz.

/// Indicate that Manchester Serial Wire Output (SWO) trace is available.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define SWO_MANCHESTER          0               ///< SWO Manchester:  1 = available, 0 = not available.

/// SWO Trace Buffer Size.
#define SWO_BUFFER_SIZE         8192U           ///< SWO Trace Buffer Size in bytes (must be 2^n).

/// SWO Streaming Trace.
#define SWO_STREAM              0               ///< SWO Streaming Trace: 1 = available, 0 = not available.

/// Clock frequency of the Test Domain Timer. Timer value is returned with \ref TIMESTAMP_GET.
#define TIMESTAMP_CLOCK         180000000U      ///< Timestamp clock in Hz (0 = timestamps not supported).

/// Indicate that UART Communication Port is available.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define DAP_UART                0               ///< DAP UART:  1 = available, 0 = not available.

/// USART Driver instance number for the UART Communication Port.
#define DAP_UART_DRIVER         0               ///< USART Driver instance number (Driver_USART#).

/// UART Receive Buffer Size.
#define DAP_UART_RX_BUFFER_SIZE 1024U           ///< Uart Receive Buffer Size in bytes (must be 2^n).

/// UART Transmit Buffer Size.
#define DAP_UART_TX_BUFFER_SIZE 1024U           ///< Uart Transmit Buffer Size in bytes (must be 2^n).

/// Indicate that UART Communication via USB COM Port is available.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define DAP_UART_USB_COM_PORT   0               ///< USB COM Port:  1 = available, 0 = not available.

/// Debug Unit is connected to fixed Target Device.
/// The Debug Unit may be part of an evaluation board and always connected to a fixed
/// known device. In this case a Device Vendor, Device Name, Board Vendor and Board Name strings
/// are stored and may be used by the debugger or IDE to configure device parameters.
#define TARGET_FIXED            0               ///< Target: 1 = known, 0 = unknown;

#define TARGET_DEVICE_VENDOR    "NXP"           ///< String indicating the Silicon Vendor
#define TARGET_DEVICE_NAME      "Cortex-M"      ///< String indicating the Target Device
#define TARGET_BOARD_VENDOR     "NXP"           ///< String indicating the Board Vendor
#define TARGET_BOARD_NAME       "NXP board"     ///< String indicating the Board Name

#if TARGET_FIXED != 0
#include <string.h>
static const char TargetDeviceVendor [] = TARGET_DEVICE_VENDOR;
static const char TargetDeviceName   [] = TARGET_DEVICE_NAME;
static const char TargetBoardVendor  [] = TARGET_BOARD_VENDOR;
static const char TargetBoardName    [] = TARGET_BOARD_NAME;
#endif

/** Get Vendor Name string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetVendorString (char *str) {
  (void)str;
  return (0U);
}

/** Get Product Name string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetProductString (char *str) {
  (void)str;
  return (0U);
}

/** Get Serial Number string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetSerNumString (char *str) {
  (void)str;
  return (0U);
}

/** Get Target Device Vendor string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetTargetDeviceVendorString (char *str) {
#if TARGET_FIXED != 0
  uint8_t len;

  strcpy(str, TargetDeviceVendor);
  len = (uint8_t)(strlen(TargetDeviceVendor) + 1U);
  return (len);
#else
  (void)str;
  return (0U);
#endif
}

/** Get Target Device Name string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetTargetDeviceNameString (char *str) {
#if TARGET_FIXED != 0
  uint8_t len;

  strcpy(str, TargetDeviceName);
  len = (uint8_t)(strlen(TargetDeviceName) + 1U);
  return (len);
#else
  (void)str;
  return (0U);
#endif
}

/** Get Target Board Vendor string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetTargetBoardVendorString (char *str) {
#if TARGET_FIXED != 0
  uint8_t len;

  strcpy(str, TargetBoardVendor);
  len = (uint8_t)(strlen(TargetBoardVendor) + 1U);
  return (len);
#else
  (void)str;
  return (0U);
#endif
}

/** Get Target Board Name string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetTargetBoardNameString (char *str) {
#if TARGET_FIXED != 0
  uint8_t len;

  strcpy(str, TargetBoardName);
  len = (uint8_t)(strlen(TargetBoardName) + 1U);
  return (len);
#else
  (void)str;
  return (0U);
#endif
}

/** Get Product Firmware Version string.
\param str Pointer to buffer to store the string (max 60 characters).
\return String length (including terminating NULL character) or 0 (no string).
*/
__STATIC_INLINE uint8_t DAP_GetProductFirmwareVersionString (char *str) {
  (void)str;
  return (0U);
}

///@}


// LPC43xx peripheral register bit masks (used by macros)
#define CCU_CLK_CFG_RUN         (1U << 0)
#define CCU_CLK_CFG_AUTO        (1U << 1)
#define CCU_CLK_STAT_RUN        (1U << 0)
#define SCU_SFS_EPD             (1U << 3)
#define SCU_SFS_EPUN            (1U << 4)
#define SCU_SFS_EHS             (1U << 5)
#define SCU_SFS_EZI             (1U << 6)
#define SCU_SFS_ZIF             (1U << 7)


// Debug Port I/O Pins

// SWCLK/TCK Pin                P1_17: GPIO0[12]
#define PIN_SWCLK_TCK_PORT      0
#define PIN_SWCLK_TCK_BIT       12

// SWDIO/TMS Pin                P1_6:  GPIO1[9]
#define PIN_SWDIO_TMS_PORT      1
#define PIN_SWDIO_TMS_BIT       9

// SWDIO Output Enable Pin      P1_5:  GPIO1[8]
#define PIN_SWDIO_OE_PORT       1
#define PIN_SWDIO_OE_BIT        8

// TDI Pin                      P1_18: GPIO0[13]
#define PIN_TDI_PORT            0
#define PIN_TDI_BIT             13

// TDO Pin                      P1_14: GPIO1[7]
#define PIN_TDO_PORT            1
#define PIN_TDO_BIT             7

// nTRST Pin                    Not available
#define PIN_nTRST_PORT
#define PIN_nTRST_BIT

// nRESET Pin                   P2_5:  GPIO5[5]
#define PIN_nRESET_PORT         5
#define PIN_nRESET_BIT          5

// nRESET Output Enable Pin     P2_6:  GPIO5[6]
#define PIN_nRESET_OE_PORT      5
#define PIN_nRESET_OE_BIT       6


// Debug Unit LEDs

// Connected LED                P1_1: GPIO0[8]
#define LED_CONNECTED_PORT      0
#define LED_CONNECTED_BIT       8

// Target Running LED           Not available


//**************************************************************************************************
/** 
\defgroup DAP_Config_PortIO_gr CMSIS-DAP Hardware I/O Pin Access
\ingroup DAP_ConfigIO_gr 
@{

Standard I/O Pins of the CMSIS-DAP Hardware Debug Port support standard JTAG mode
and Serial Wire Debug (SWD) mode. In SWD mode only 2 pins are required to implement the debug 
interface of a device. The following I/O Pins are provided:

JTAG I/O Pin                 | SWD I/O Pin          | CMSIS-DAP Hardware pin mode
---------------------------- | -------------------- | ---------------------------------------------
TCK: Test Clock              | SWCLK: Clock         | Output Push/Pull
TMS: Test Mode Select        | SWDIO: Data I/O      | Output Push/Pull; Input (for receiving data)
TDI: Test Data Input         |                      | Output Push/Pull
TDO: Test Data Output        |                      | Input             
nTRST: Test Reset (optional) |                      | Output Open Drain with pull-up resistor
nRESET: Device Reset         | nRESET: Device Reset | Output Open Drain with pull-up resistor


DAP Hardware I/O Pin Access Functions
-------------------------------------
The various I/O Pins are accessed by functions that implement the Read, Write, Set, or Clear to 
these I/O Pins. 

For the SWDIO I/O Pin there are additional functions that are called in SWD I/O mode only.
This functions are provided to achieve faster I/O that is possible with some advanced GPIO 
peripherals that can independently write/read a single I/O pin without affecting any other pins 
of the same I/O port. The following SWDIO I/O Pin functions are provided:
 - \ref PIN_SWDIO_OUT_ENABLE to enable the output mode from the DAP hardware.
 - \ref PIN_SWDIO_OUT_DISABLE to enable the input mode to the DAP hardware.
 - \ref PIN_SWDIO_IN to read from the SWDIO I/O pin with utmost possible speed.
 - \ref PIN_SWDIO_OUT to write to the SWDIO I/O pin with utmost possible speed.
*/


// Configure DAP I/O pins ------------------------------

//   LPC-Link2 HW uses buffers for debug port pins. Therefore it is not
//   possible to disable outputs SWCLK/TCK, TDI and they are left active.
//   Only SWDIO/TMS output can be disabled but it is also left active.
//   nRESET is configured for open drain mode.

/** Setup JTAG I/O pins: TCK, TMS, TDI, TDO, nTRST, and nRESET.
Configures the DAP Hardware I/O pins for JTAG mode:
 - TCK, TMS, TDI, nTRST, nRESET to output mode and set to high level.
 - TDO to input mode.
*/ 
__STATIC_INLINE void PORT_JTAG_SETUP (void) {
  LPC_GPIO_PORT->MASK[PIN_SWDIO_TMS_PORT] = 0U;
  LPC_GPIO_PORT->MASK[PIN_TDI_PORT] = ~(1U << PIN_TDI_BIT);
}
 
/** Setup SWD I/O pins: SWCLK, SWDIO, and nRESET.
Configures the DAP Hardware I/O pins for Serial Wire Debug (SWD) mode:
 - SWCLK, SWDIO, nRESET to output mode and set to default high level.
 - TDI, nTRST to HighZ mode (pins are unused in SWD mode).
*/ 
__STATIC_INLINE void PORT_SWD_SETUP (void) {
  LPC_GPIO_PORT->MASK[PIN_TDI_PORT] = 0U;
  LPC_GPIO_PORT->MASK[PIN_SWDIO_TMS_PORT] = ~(1U << PIN_SWDIO_TMS_BIT);
}

/** Disable JTAG/SWD I/O Pins.
Disables the DAP Hardware I/O pins which configures:
 - TCK/SWCLK, TMS/SWDIO, TDI, TDO, nTRST, nRESET to High-Z mode.
*/
__STATIC_INLINE void PORT_OFF (void) {
  LPC_GPIO_PORT->SET[PIN_SWCLK_TCK_PORT]  =  (1U << PIN_SWCLK_TCK_BIT);
  LPC_GPIO_PORT->SET[PIN_SWDIO_TMS_PORT]  =  (1U << PIN_SWDIO_TMS_BIT);
  LPC_GPIO_PORT->SET[PIN_SWDIO_OE_PORT]   =  (1U << PIN_SWDIO_OE_BIT);
  LPC_GPIO_PORT->SET[PIN_TDI_PORT]        =  (1U << PIN_TDI_BIT);
  LPC_GPIO_PORT->DIR[PIN_nRESET_PORT]    &= ~(1U << PIN_nRESET_BIT);
  LPC_GPIO_PORT->CLR[PIN_nRESET_OE_PORT]  =  (1U << PIN_nRESET_OE_BIT);
}


// SWCLK/TCK I/O pin -------------------------------------

/** SWCLK/TCK I/O pin: Get Input.
\return Current status of the SWCLK/TCK DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_SWCLK_TCK_IN  (void) {
  return ((LPC_GPIO_PORT->PIN[PIN_SWCLK_TCK_PORT] >> PIN_SWCLK_TCK_BIT) & 1U);
}

/** SWCLK/TCK I/O pin: Set Output to High.
Set the SWCLK/TCK DAP hardware I/O pin to high level.
*/
__STATIC_FORCEINLINE void     PIN_SWCLK_TCK_SET (void) {
  LPC_GPIO_PORT->SET[PIN_SWCLK_TCK_PORT] = 1U << PIN_SWCLK_TCK_BIT;
}

/** SWCLK/TCK I/O pin: Set Output to Low.
Set the SWCLK/TCK DAP hardware I/O pin to low level.
*/
__STATIC_FORCEINLINE void     PIN_SWCLK_TCK_CLR (void) {
  LPC_GPIO_PORT->CLR[PIN_SWCLK_TCK_PORT] = 1U << PIN_SWCLK_TCK_BIT;
}


// SWDIO/TMS Pin I/O --------------------------------------

/** SWDIO/TMS I/O pin: Get Input.
\return Current status of the SWDIO/TMS DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_SWDIO_TMS_IN  (void) {
  return ((LPC_GPIO_PORT->PIN[PIN_SWDIO_TMS_PORT] >> PIN_SWDIO_TMS_BIT) & 1U);
}

/** SWDIO/TMS I/O pin: Set Output to High.
Set the SWDIO/TMS DAP hardware I/O pin to high level.
*/
__STATIC_FORCEINLINE void     PIN_SWDIO_TMS_SET (void) {
  LPC_GPIO_PORT->SET[PIN_SWDIO_TMS_PORT] = 1U << PIN_SWDIO_TMS_BIT;
}

/** SWDIO/TMS I/O pin: Set Output to Low.
Set the SWDIO/TMS DAP hardware I/O pin to low level.
*/
__STATIC_FORCEINLINE void     PIN_SWDIO_TMS_CLR (void) {
  LPC_GPIO_PORT->CLR[PIN_SWDIO_TMS_PORT] = 1U << PIN_SWDIO_TMS_BIT;
}

/** SWDIO I/O pin: Get Input (used in SWD mode only).
\return Current status of the SWDIO DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_SWDIO_IN      (void) {
  return (LPC_GPIO_PORT->MPIN[PIN_SWDIO_TMS_PORT] >> PIN_SWDIO_TMS_BIT);
}

/** SWDIO I/O pin: Set Output (used in SWD mode only).
\param bit Output value for the SWDIO DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE void     PIN_SWDIO_OUT     (uint32_t bit) {
  LPC_GPIO_PORT->MPIN[PIN_SWDIO_TMS_PORT] = bit << PIN_SWDIO_TMS_BIT;
}

/** SWDIO I/O pin: Switch to Output mode (used in SWD mode only).
Configure the SWDIO DAP hardware I/O pin to output mode. This function is
called prior \ref PIN_SWDIO_OUT function calls.
*/
__STATIC_FORCEINLINE void     PIN_SWDIO_OUT_ENABLE  (void) {
  LPC_GPIO_PORT->SET[PIN_SWDIO_OE_PORT] = 1U << PIN_SWDIO_OE_BIT;
}

/** SWDIO I/O pin: Switch to Input mode (used in SWD mode only).
Configure the SWDIO DAP hardware I/O pin to input mode. This function is
called prior \ref PIN_SWDIO_IN function calls.
*/
__STATIC_FORCEINLINE void     PIN_SWDIO_OUT_DISABLE (void) {
  LPC_GPIO_PORT->CLR[PIN_SWDIO_OE_PORT] = 1U << PIN_SWDIO_OE_BIT;
}


// TDI Pin I/O ---------------------------------------------

/** TDI I/O pin: Get Input.
\return Current status of the TDI DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_TDI_IN  (void) {
  return ((LPC_GPIO_PORT->PIN [PIN_TDI_PORT] >> PIN_TDI_BIT) & 1U);
}

/** TDI I/O pin: Set Output.
\param bit Output value for the TDI DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE void     PIN_TDI_OUT (uint32_t bit) {
  LPC_GPIO_PORT->MPIN[PIN_TDI_PORT] = bit << PIN_TDI_BIT;
}


// TDO Pin I/O ---------------------------------------------

/** TDO I/O pin: Get Input.
\return Current status of the TDO DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_TDO_IN  (void) {
  return ((LPC_GPIO_PORT->PIN[PIN_TDO_PORT] >> PIN_TDO_BIT) & 1U);
}


// nTRST Pin I/O -------------------------------------------

/** nTRST I/O pin: Get Input.
\return Current status of the nTRST DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_nTRST_IN   (void) {
  return (0U);  // Not available
}

/** nTRST I/O pin: Set Output.
\param bit JTAG TRST Test Reset pin status:
           - 0: issue a JTAG TRST Test Reset.
           - 1: release JTAG TRST Test Reset.
*/
__STATIC_FORCEINLINE void     PIN_nTRST_OUT  (uint32_t bit) {
  (void) bit;
  // Not available
}

// nRESET Pin I/O------------------------------------------

/** nRESET I/O pin: Get Input.
\return Current status of the nRESET DAP hardware I/O pin.
*/
__STATIC_FORCEINLINE uint32_t PIN_nRESET_IN  (void) {
  return ((LPC_GPIO_PORT->PIN[PIN_nRESET_PORT] >> PIN_nRESET_BIT) & 1U);
}

/** nRESET I/O pin: Set Output.
\param bit target device hardware reset pin status:
           - 0: issue a device hardware reset.
           - 1: release device hardware reset.
*/
__STATIC_FORCEINLINE void     PIN_nRESET_OUT (uint32_t bit) {
  if (bit) {
    LPC_GPIO_PORT->DIR[PIN_nRESET_PORT]    &= ~(1U << PIN_nRESET_BIT);
    LPC_GPIO_PORT->CLR[PIN_nRESET_OE_PORT]  =  (1U << PIN_nRESET_OE_BIT);
  } else {
    LPC_GPIO_PORT->SET[PIN_nRESET_OE_PORT]  =  (1U << PIN_nRESET_OE_BIT);
    LPC_GPIO_PORT->DIR[PIN_nRESET_PORT]    |=  (1U << PIN_nRESET_BIT);
  }
}

///@}


//**************************************************************************************************
/** 
\defgroup DAP_Config_LEDs_gr CMSIS-DAP Hardware Status LEDs
\ingroup DAP_ConfigIO_gr
@{

CMSIS-DAP Hardware may provide LEDs that indicate the status of the CMSIS-DAP Debug Unit.

It is recommended to provide the following LEDs for status indication:
 - Connect LED: is active when the DAP hardware is connected to a debugger.
 - Running LED: is active when the debugger has put the target device into running state.
*/

/** Debug Unit: Set status of Connected LED.
\param bit status of the Connect LED.
           - 1: Connect LED ON: debugger is connected to CMSIS-DAP Debug Unit.
           - 0: Connect LED OFF: debugger is not connected to CMSIS-DAP Debug Unit.
*/
__STATIC_INLINE void LED_CONNECTED_OUT (uint32_t bit) {
  LPC_GPIO_PORT->B[32*LED_CONNECTED_PORT + LED_CONNECTED_BIT] = (uint8_t)bit;
}

/** Debug Unit: Set status Target Running LED.
\param bit status of the Target Running LED.
           - 1: Target Running LED ON: program execution in target started.
           - 0: Target Running LED OFF: program execution in target stopped.
*/
__STATIC_INLINE void LED_RUNNING_OUT (uint32_t bit) {
  (void) bit;
  // Not available
}

///@}


//**************************************************************************************************
/** 
\defgroup DAP_Config_Timestamp_gr CMSIS-DAP Timestamp
\ingroup DAP_ConfigIO_gr
@{
Access function for Test Domain Timer.

The value of the Test Domain Timer in the Debug Unit is returned by the function \ref TIMESTAMP_GET. By 
default, the DWT timer is used.  The frequency of this timer is configured with \ref TIMESTAMP_CLOCK.

*/

/** Get timestamp of Test Domain Timer.
\return Current timestamp value.
*/
__STATIC_INLINE uint32_t TIMESTAMP_GET (void) {
  return (DWT->CYCCNT);
}

///@}


//**************************************************************************************************
/** 
\defgroup DAP_Config_Initialization_gr CMSIS-DAP Initialization
\ingroup DAP_ConfigIO_gr
@{

CMSIS-DAP Hardware I/O and LED Pins are initialized with the function \ref DAP_SETUP.
*/

/** Setup of the Debug Unit I/O pins and LEDs (called when Debug Unit is initialized).
This function performs the initialization of the CMSIS-DAP Hardware I/O Pins and the 
Status LEDs. In detail the operation of Hardware I/O and LED pins are enabled and set:
 - I/O clock system enabled.
 - all I/O pins: input buffer enabled, output pins are set to HighZ mode.
 - for nTRST, nRESET a weak pull-up (if available) is enabled.
 - LED output pins are enabled and LEDs are turned off.
*/
__STATIC_INLINE void DAP_SETUP (void) {

  /* Enable clock and init GPIO outputs */
  LPC_CCU1->CLK_M4_GPIO_CFG = CCU_CLK_CFG_AUTO | CCU_CLK_CFG_RUN;
  while (!(LPC_CCU1->CLK_M4_GPIO_STAT & CCU_CLK_STAT_RUN));

  /* Configure I/O pins: function number, input buffer enabled,  */
  /*                     no pull-up/down except nRESET (pull-up) */
  LPC_SCU->SFSP1_17 = 0U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* SWCLK/TCK: GPIO0[12] */
  LPC_SCU->SFSP1_6  = 0U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* SWDIO/TMS: GPIO1[9]  */
  LPC_SCU->SFSP1_5  = 0U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* SWDIO_OE:  GPIO1[8]  */
  LPC_SCU->SFSP1_18 = 0U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* TDI:       GPIO0[13] */
  LPC_SCU->SFSP1_14 = 0U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* TDO:       GPIO1[7]  */
  LPC_SCU->SFSP2_5  = 4U |              SCU_SFS_EZI;  /* nRESET:    GPIO5[5]  */
  LPC_SCU->SFSP2_6  = 4U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* nRESET_OE: GPIO5[6]  */
  LPC_SCU->SFSP1_1  = 0U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* LED:       GPIO0[8]  */
#ifdef TARGET_POWER_EN
  LPC_SCU->SFSP3_1  = 4U | SCU_SFS_EPUN|SCU_SFS_EZI;  /* Target Power enable P3_1 GPIO5[8] */
#endif

  /* Configure: SWCLK/TCK, SWDIO/TMS, SWDIO_OE, TDI as outputs (high level)   */
  /*            TDO as input                                                  */
  /*            nRESET as input with output latch set to low level            */
  /*            nRESET_OE as output (low level)                               */
  LPC_GPIO_PORT->SET[PIN_SWCLK_TCK_PORT]  =  (1U << PIN_SWCLK_TCK_BIT);
  LPC_GPIO_PORT->SET[PIN_SWDIO_TMS_PORT]  =  (1U << PIN_SWDIO_TMS_BIT);
  LPC_GPIO_PORT->SET[PIN_SWDIO_OE_PORT]   =  (1U << PIN_SWDIO_OE_BIT);
  LPC_GPIO_PORT->SET[PIN_TDI_PORT]        =  (1U << PIN_TDI_BIT);
  LPC_GPIO_PORT->CLR[PIN_nRESET_PORT]     =  (1U << PIN_nRESET_BIT);
  LPC_GPIO_PORT->CLR[PIN_nRESET_OE_PORT]  =  (1U << PIN_nRESET_OE_BIT);
  LPC_GPIO_PORT->DIR[PIN_SWCLK_TCK_PORT] |=  (1U << PIN_SWCLK_TCK_BIT);
  LPC_GPIO_PORT->DIR[PIN_SWDIO_TMS_PORT] |=  (1U << PIN_SWDIO_TMS_BIT);
  LPC_GPIO_PORT->DIR[PIN_SWDIO_OE_PORT]  |=  (1U << PIN_SWDIO_OE_BIT);
  LPC_GPIO_PORT->DIR[PIN_TDI_PORT]       |=  (1U << PIN_TDI_BIT);
  LPC_GPIO_PORT->DIR[PIN_TDO_PORT]       &= ~(1U << PIN_TDO_BIT);
  LPC_GPIO_PORT->DIR[PIN_nRESET_PORT]    &= ~(1U << PIN_nRESET_BIT);
  LPC_GPIO_PORT->DIR[PIN_nRESET_OE_PORT] |=  (1U << PIN_nRESET_OE_BIT);

#ifdef TARGET_POWER_EN
  /* Target Power enable as output (turned on)  */
  LPC_GPIO_PORT->SET[5]  = (1U << 8);
  LPC_GPIO_PORT->DIR[5] |= (1U << 8);
#endif

  /* Configure: LED as output (turned off) */
  LPC_GPIO_PORT->CLR[LED_CONNECTED_PORT]  =  (1U << LED_CONNECTED_BIT);
  LPC_GPIO_PORT->DIR[LED_CONNECTED_PORT] |=  (1U << LED_CONNECTED_BIT);

  /* Configure Peripheral Interrupt Priorities */
  NVIC_SetPriority(USB0_IRQn, 1U);
}

/** Reset Target Device with custom specific I/O pin or command sequence.
This function allows the optional implementation of a device specific reset sequence.
It is called when the command \ref DAP_ResetTarget and is for example required 
when a device needs a time-critical unlock sequence that enables the debug port.
\return 0 = no device specific reset sequence is implemented.\n
        1 = a device specific reset sequence is implemented.
*/
__STATIC_INLINE uint8_t RESET_TARGET (void) {
  return (0U);             // change to '1' when a device reset sequence is implemented
}

///@}


#endif /* __DAP_CONFIG_H__ */
