/*
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
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
 * $Date:        20. May 2015
 * $Revision:    V1.10
 *
 * Project:      CMSIS-DAP Examples XMC4200
 * Title:        DAP_config.h CMSIS-DAP Configuration File for XMC4200
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
 - Debug Unit communication packet size.
 - Debug Access Port communication mode (JTAG or SWD).
 - Optional information about a connected Target Device (for Evaluation Boards).
*/

#include "XMC4200.h"                    // Debug Unit Cortex-M Processor Header File
#include "xmc_gpio.h"
#include "xmc4_gpio.h"

/// Processor Clock of the Cortex-M MCU used in the Debug Unit.
/// This value is used to calculate the SWD/JTAG clock speed.
#define CPU_CLOCK               80000000U       ///< Specifies the CPU Clock in Hz

/// Number of processor cycles for I/O Port write operations.
/// This value is used to calculate the SWD/JTAG clock speed that is generated with I/O
/// Port write operations in the Debug Unit by a Cortex-M MCU. Most Cortex-M processors
/// require 2 processor cycles for a I/O Port Write operation.  If the Debug Unit uses
/// a Cortex-M0+ processor with high-speed peripheral I/O only 1 processor cycle might be 
/// required.
#define IO_PORT_WRITE_CYCLES    2U              ///< I/O Cycles: 2=default, 1=Cortex-M0+ fast I/0

/// Indicate that Serial Wire Debug (SWD) communication mode is available at the Debug Access Port.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define DAP_SWD                 1               ///< SWD Mode:  1 = available, 0 = not available

/// Indicate that JTAG communication mode is available at the Debug Port.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define DAP_JTAG                0               ///< JTAG Mode: 1 = available, 0 = not available.

/// Configure maximum number of JTAG devices on the scan chain connected to the Debug Access Port.
/// This setting impacts the RAM requirements of the Debug Unit. Valid range is 1 .. 255.
#define DAP_JTAG_DEV_CNT        8U              ///< Maximum number of JTAG devices on scan chain

/// Default communication mode on the Debug Access Port.
/// Used for the command \ref DAP_Connect when Port Default mode is selected.
#define DAP_DEFAULT_PORT        1U              ///< Default JTAG/SWJ Port Mode: 1 = SWD, 2 = JTAG.

/// Default communication speed on the Debug Access Port for SWD and JTAG mode.
/// Used to initialize the default SWD/JTAG clock frequency.
/// The command \ref DAP_SWJ_Clock can be used to overwrite this default setting.
#define DAP_DEFAULT_SWJ_CLOCK   1000000U        ///< Default SWD/JTAG clock frequency in Hz.

/// Maximum Package Size for Command and Response data.
/// This configuration settings is used to optimized the communication performance with the
/// debugger and depends on the USB peripheral. Change setting to 1024 for High-Speed USB.
#define DAP_PACKET_SIZE         64U             ///< USB: 64 = Full-Speed, 1024 = High-Speed.

/// Maximum Package Buffers for Command and Response data.
/// This configuration settings is used to optimized the communication performance with the
/// debugger and depends on the USB peripheral. For devices with limited RAM or USB buffer the
/// setting can be reduced (valid range is 1 .. 255). Change setting to 4 for High-Speed USB.
#define DAP_PACKET_COUNT        64U             ///< Buffers: 64 = Full-Speed, 4 = High-Speed.

/// Indicate that UART Serial Wire Output (SWO) trace is available.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define SWO_UART                1               ///< SWO UART:  1 = available, 0 = not available

/// Maximum SWO UART Baudrate
#define SWO_UART_MAX_BAUDRATE   1000000U        ///< SWO UART Maximum Baudrate in Hz

/// Indicate that Manchester Serial Wire Output (SWO) trace is available.
/// This information is returned by the command \ref DAP_Info as part of <b>Capabilities</b>.
#define SWO_MANCHESTER          0               ///< SWO Manchester:  1 = available, 0 = not available

/// SWO Trace Buffer Size.
#define SWO_BUFFER_SIZE         8192U           ///< SWO Trace Buffer Size in bytes (must be 2^n)


/// Debug Unit is connected to fixed Target Device.
/// The Debug Unit may be part of an evaluation board and always connected to a fixed
/// known device.  In this case a Device Vendor and Device Name string is stored which
/// may be used by the debugger or IDE to configure device parameters.
#define TARGET_DEVICE_FIXED     0               ///< Target Device: 1 = known, 0 = unknown;

#if TARGET_DEVICE_FIXED
#define TARGET_DEVICE_VENDOR    ""              ///< String indicating the Silicon Vendor
#define TARGET_DEVICE_NAME      ""              ///< String indicating the Target Device
#endif

///@}


// Debug Port I/O Pins

// SWCLK Pin                    P1.1
#define PIN_SWCLK_TCK_PORT      XMC_GPIO_PORT1
#define PIN_SWCLK_TCK_BIT       1

// SWCLK_IN Pin                 P2.4
#define PIN_SWCLK_TCK_IN_PORT   XMC_GPIO_PORT2
#define PIN_SWCLK_TCK_IN_BIT    4

// SWDIO out                    P1.5
#define PIN_SWDIO_OUT_PORT      XMC_GPIO_PORT1
#define PIN_SWDIO_OUT_BIT       5

// SWDIO in                     P1.4
#define PIN_SWDIO_IN_PORT       XMC_GPIO_PORT1
#define PIN_SWDIO_IN_BIT        4

// SWDIO Output Enable Pin      P1.3
#define PIN_SWDIO_OE_PORT       XMC_GPIO_PORT1
#define PIN_SWDIO_OE_BIT        3

// TDI Pin                      Not available
//#define PIN_TDI_PORT          
//#define PIN_TDI_BIT           

// TDO Pin                      Not available
//#define PIN_TDO_PORT          
//#define PIN_TDO_BIT           

// nTRST Pin                    Not available
//#define PIN_nTRST_PORT
//#define PIN_nTRST_BIT

// nRESET Pin                   P0.5
#define PIN_nRESET_PORT         XMC_GPIO_PORT0
#define PIN_nRESET_BIT          5


// Debug Unit LEDs

// Connected LED                P0.2
#define LED_CONNECTED_PORT      XMC_GPIO_PORT0
#define LED_CONNECTED_BIT       2

// Target Running LED           P0.1
#define LED_RUNNING_PORT        XMC_GPIO_PORT0
#define LED_RUNNING_BIT         1


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

//   LPC-Link-II HW uses buffers for debug port pins. Therefore it is not
//   possible to disable outputs SWCLK/TCK, TDI and they are left active.
//   Only SWDIO/TMS output can be disabled but it is also left active.
//   nRESET is configured for open drain mode.

/** Setup JTAG I/O pins: TCK, TMS, TDI, TDO, nTRST, and nRESET.
Configures the DAP Hardware I/O pins for JTAG mode:
 - TCK, TMS, TDI, nTRST, nRESET to output mode and set to high level.
 - TDO to input mode.
*/ 
static __inline void PORT_JTAG_SETUP (void) {
  // Not available
}
 
/** Setup SWD I/O pins: SWCLK, SWDIO, and nRESET.
Configures the DAP Hardware I/O pins for Serial Wire Debug (SWD) mode:
 - SWCLK, SWDIO, nRESET to output mode and set to default high level.
 - TDI, nTRST to HighZ mode (pins are unused in SWD mode).
*/ 
static __inline void PORT_SWD_SETUP (void) {
  XMC_GPIO_CONFIG_t config;

  /* SWCLK: Enable Output */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWCLK_TCK_PORT, PIN_SWCLK_TCK_BIT, &config);
 
  /* SWCLK_IN: Disable Output */
  config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
  XMC_GPIO_Init(PIN_SWCLK_TCK_IN_PORT, PIN_SWCLK_TCK_IN_BIT, &config);

  /* SWDIO out: Enable Output */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT, &config);
 
  /* SWDIO Output Enable: Enable Output */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWDIO_OE_PORT, PIN_SWDIO_OE_BIT, &config);

  /* SWDIO in: Enable Input */
  config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
  XMC_GPIO_Init(PIN_SWDIO_IN_PORT, PIN_SWDIO_IN_BIT, &config);

  /* nRESET: Enable I/O, open-drain, pull-up */    
  /* HS: open-drain is not possible with this hardware. I reused the PC_RXD_DEV/TX signal to create the reset */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_nRESET_PORT, PIN_nRESET_BIT, &config);
}

/** Disable JTAG/SWD I/O Pins.
Disables the DAP Hardware I/O pins which configures:
 - TCK/SWCLK, TMS/SWDIO, TDI, TDO, nTRST, nRESET to High-Z mode.
*/
static __inline void PORT_OFF (void) {
  XMC_GPIO_CONFIG_t config;

  /* SWCLK: Disable Output */
//config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
//configuring SWCLK to tri-state mode does not make sense because the isolation device then has an undefined input 
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWCLK_TCK_PORT, PIN_SWCLK_TCK_BIT, &config);
 
  /* SWCLK_IN: Disable Output */
  config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
  XMC_GPIO_Init(PIN_SWCLK_TCK_IN_PORT, PIN_SWCLK_TCK_IN_BIT, &config);

  /* SWDIO out: Disable Output */
//config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
//configuring SWDIO out to tri-state mode does not make sense because the isolation device then has an undefined input 
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_WEAK;               /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT, &config);
 
  /* SWDIO Output Enable: Disable Output */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_LOW;                   /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWDIO_OE_PORT, PIN_SWDIO_OE_BIT, &config);

  /* SWDIO in: Enable Input */
  config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
  XMC_GPIO_Init(PIN_SWDIO_IN_PORT, PIN_SWDIO_IN_BIT, &config);

  /* nRESET: Enable I/O, open-drain, pull-up */    
  /* HS: open-drain is not possible with this hardware. I reused the PC_RXD_DEV/TX signal to create the reset */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_WEAK;               /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_nRESET_PORT, PIN_nRESET_BIT, &config);
}


// SWCLK/TCK I/O pin -------------------------------------

/** SWCLK/TCK I/O pin: Get Input.
\return Current status of the SWCLK/TCK DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_SWCLK_TCK_IN  (void) {
  return (XMC_GPIO_GetInput(PIN_SWCLK_TCK_IN_PORT, PIN_SWCLK_TCK_IN_BIT));
}

/** SWCLK/TCK I/O pin: Set Output to High.
Set the SWCLK/TCK DAP hardware I/O pin to high level.
*/
static __forceinline void     PIN_SWCLK_TCK_SET (void) {
  XMC_GPIO_SetOutputHigh(PIN_SWCLK_TCK_PORT, PIN_SWCLK_TCK_BIT);
}

/** SWCLK/TCK I/O pin: Set Output to Low.
Set the SWCLK/TCK DAP hardware I/O pin to low level.
*/
static __forceinline void     PIN_SWCLK_TCK_CLR (void) {
  XMC_GPIO_SetOutputLow(PIN_SWCLK_TCK_PORT, PIN_SWCLK_TCK_BIT);
}


// SWDIO/TMS Pin I/O --------------------------------------

/** SWDIO/TMS I/O pin: Get Input.
\return Current status of the SWDIO/TMS DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_SWDIO_TMS_IN  (void) {
  return (XMC_GPIO_GetInput(PIN_SWDIO_IN_PORT, PIN_SWDIO_IN_BIT));
}

/** SWDIO/TMS I/O pin: Set Output to High.
Set the SWDIO/TMS DAP hardware I/O pin to high level.
*/
static __forceinline void     PIN_SWDIO_TMS_SET (void) {
  XMC_GPIO_SetOutputHigh(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT);
}

/** SWDIO/TMS I/O pin: Set Output to Low.
Set the SWDIO/TMS DAP hardware I/O pin to low level.
*/
static __forceinline void     PIN_SWDIO_TMS_CLR (void) {
  XMC_GPIO_SetOutputLow(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT);
}

/** SWDIO I/O pin: Get Input (used in SWD mode only).
\return Current status of the SWDIO DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_SWDIO_IN      (void) {
  return (XMC_GPIO_GetInput(PIN_SWDIO_IN_PORT, PIN_SWDIO_IN_BIT));
}

/** SWDIO I/O pin: Set Output (used in SWD mode only).
\param bit Output value for the SWDIO DAP hardware I/O pin.
*/
static __forceinline void     PIN_SWDIO_OUT     (uint32_t bit) {
  if (bit & 0x0001) {
    XMC_GPIO_SetOutputHigh(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT);
  } else {
    XMC_GPIO_SetOutputLow(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT);
  }
}

/** SWDIO I/O pin: Switch to Output mode (used in SWD mode only).
Configure the SWDIO DAP hardware I/O pin to output mode. This function is
called prior \ref PIN_SWDIO_OUT function calls.
*/
static __forceinline void     PIN_SWDIO_OUT_ENABLE  (void) {
  XMC_GPIO_SetOutputHigh(PIN_SWDIO_OE_PORT, PIN_SWDIO_OE_BIT);
}

/** SWDIO I/O pin: Switch to Input mode (used in SWD mode only).
Configure the SWDIO DAP hardware I/O pin to input mode. This function is
called prior \ref PIN_SWDIO_IN function calls.
*/
static __forceinline void     PIN_SWDIO_OUT_DISABLE (void) {
  XMC_GPIO_SetOutputLow(PIN_SWDIO_OE_PORT, PIN_SWDIO_OE_BIT);
}


// TDI Pin I/O ---------------------------------------------

/** TDI I/O pin: Get Input.
\return Current status of the TDI DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_TDI_IN  (void) {
  return (0U);  // Not available
}

/** TDI I/O pin: Set Output.
\param bit Output value for the TDI DAP hardware I/O pin.
*/
static __forceinline void     PIN_TDI_OUT (uint32_t bit) {
  // Not available
}


// TDO Pin I/O ---------------------------------------------

/** TDO I/O pin: Get Input.
\return Current status of the TDO DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_TDO_IN  (void) {
  return (0U);  // Not available
}


// nTRST Pin I/O -------------------------------------------

/** nTRST I/O pin: Get Input.
\return Current status of the nTRST DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_nTRST_IN   (void) {
  return (0U);  // Not available
}

/** nTRST I/O pin: Set Output.
\param bit JTAG TRST Test Reset pin status:
           - 0: issue a JTAG TRST Test Reset.
           - 1: release JTAG TRST Test Reset.
*/
static __forceinline void     PIN_nTRST_OUT  (uint32_t bit) {
  ;             // Not available
}

// nRESET Pin I/O------------------------------------------

/** nRESET I/O pin: Get Input.
\return Current status of the nRESET DAP hardware I/O pin.
*/
static __forceinline uint32_t PIN_nRESET_IN  (void) {
//return (XMC_GPIO_GetInput(PIN_nRESET_PORT, PIN_nRESET_BIT));
  return (0);   // Not available
}

/** nRESET I/O pin: Set Output.
\param bit target device hardware reset pin status:
           - 0: issue a device hardware reset.
           - 1: release device hardware reset.
*/
static __forceinline void     PIN_nRESET_OUT (uint32_t bit) {
  if (bit) {
    XMC_GPIO_SetOutputHigh(PIN_nRESET_PORT, PIN_nRESET_BIT);
  } else {
    XMC_GPIO_SetOutputLow(PIN_nRESET_PORT, PIN_nRESET_BIT);
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
static __inline void LED_CONNECTED_OUT (uint32_t bit) {

  if (bit) {
    XMC_GPIO_SetOutputLow(LED_CONNECTED_PORT, LED_CONNECTED_BIT);
  } else {
    XMC_GPIO_SetOutputHigh(LED_CONNECTED_PORT, LED_CONNECTED_BIT);
  }
}

/** Debug Unit: Set status Target Running LED.
\param bit status of the Target Running LED.
           - 1: Target Running LED ON: program execution in target started.
           - 0: Target Running LED OFF: program execution in target stopped.
*/
static __inline void LED_RUNNING_OUT (uint32_t bit) {

  if (bit) {
    XMC_GPIO_SetOutputLow(LED_RUNNING_PORT, LED_RUNNING_BIT);
  } else {
    XMC_GPIO_SetOutputHigh(LED_RUNNING_PORT, LED_RUNNING_BIT);
  }
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
static __inline void DAP_SETUP (void) {
  XMC_GPIO_CONFIG_t config;

  /* Enable clock and init GPIO outputs */
//LPC_CCU1->CLK_M4_GPIO_CFG = CCU_CLK_CFG_AUTO | CCU_CLK_CFG_RUN;
//while (!(LPC_CCU1->CLK_M4_GPIO_STAT & CCU_CLK_STAT_RUN));

  /* Configure I/O pins: function number, input buffer enabled,  */
  /*                     no pull-up/down except nRESET (pull-up) */
  /* SWCLK: Disable Output */
//config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
//configuring SWCLK to tri-state mode does not make sense because the isolation device then has an undefined input 
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SHARP_EDGE;  /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWCLK_TCK_PORT, PIN_SWCLK_TCK_BIT, &config);
 
  /* SWCLK_IN: Disable Output */
  config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
  XMC_GPIO_Init(PIN_SWCLK_TCK_IN_PORT, PIN_SWCLK_TCK_IN_BIT, &config);
 
  /* SWDIO out: Disable Output */
//config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
// onfiguring SWDIO out to tri-state mode does not make sense because the isolation device then has an undefined input 
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_WEAK;               /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWDIO_OUT_PORT, PIN_SWDIO_OUT_BIT, &config);
 
  /* SWDIO Output Enable: Disable Output */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_LOW;                   /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_WEAK;               /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_SWDIO_OE_PORT, PIN_SWDIO_OE_BIT, &config);

  /* SWDIO in: Enable Input */
  config.mode            = XMC_GPIO_MODE_INPUT_TRISTATE;                /**< Defines the direction and characteristics of a pin */
  XMC_GPIO_Init(PIN_SWDIO_IN_PORT, PIN_SWDIO_IN_BIT, &config);

  /* TDI: Enable I/O */
  // not available
 
  /* TDO: Enable I/O */
  // not available

  /* nTRST: Enable I/O, open-drain, pull-up */
  // not available

  /* nRESET: Enable I/O, open-drain, pull-up */    
  /* HS: open-drain is not possible with this hardware. I reused the PC_RXD_DEV/TX signal to create the reset */
  config.mode            = XMC_GPIO_MODE_OUTPUT_PUSH_PULL;              /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_WEAK;               /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(PIN_nRESET_PORT, PIN_nRESET_BIT, &config);

  /* Configure: Connect LED as output (turned off) */
  config.mode            = XMC_GPIO_MODE_OUTPUT_OPEN_DRAIN;             /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SLOW_EDGE;   /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(LED_CONNECTED_PORT, LED_CONNECTED_BIT, &config);

  /* Configure: Running LED as output (turned off) */
  config.mode            = XMC_GPIO_MODE_OUTPUT_OPEN_DRAIN;             /**< Defines the direction and characteristics of a pin */
  config.output_level    = XMC_GPIO_OUTPUT_LEVEL_HIGH;                  /**< Defines output level of a pin */
  config.output_strength = XMC_GPIO_OUTPUT_STRENGTH_STRONG_SLOW_EDGE;   /**< Defines pad driver mode of a pin */
  XMC_GPIO_Init(LED_RUNNING_PORT, LED_RUNNING_BIT, &config);

  /* Configure Peripheral Interrupt Priorities */
//NVIC_SetPriority(USB0_IRQn, 1U);
}

/** Reset Target Device with custom specific I/O pin or command sequence.
This function allows the optional implementation of a device specific reset sequence.
It is called when the command \ref DAP_ResetTarget and is for example required 
when a device needs a time-critical unlock sequence that enables the debug port.
\return 0 = no device specific reset sequence is implemented.\n
        1 = a device specific reset sequence is implemented.
*/
static __inline uint32_t RESET_TARGET (void) {
  return (0);              // change to '1' when a device reset sequence is implemented
}

///@}


#endif /* __DAP_CONFIG_H__ */
