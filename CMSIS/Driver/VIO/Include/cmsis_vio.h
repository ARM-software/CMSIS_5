/**************************************************************************//**
 * @file     cmsis_vio.h
 * @brief    CMSIS Virtual I/O header file
 * @version  V1.4.0
 * @date     11. March 2020
 ******************************************************************************/
/*
 * Copyright (c) 2019-2020 Arm Limited. All rights reserved.
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

#ifndef __CMSIS_VIO_H
#define __CMSIS_VIO_H

#include <stdint.h>

/*******************************************************************************
 * Generic I/O mapping recommended for CMSIS-VIO
 * Note: not every I/O must be physically available
 */
 
// cvSetSignal: mask values 
#define cvLED0             (1U << 0)   ///< \ref cvSetSignal \a mask parameter: LED 0 (for 3-color: red)
#define cvLED1             (1U << 1)   ///< \ref cvSetSignal \a mask parameter: LED 1 (for 3-color: green)
#define cvLED2             (1U << 2)   ///< \ref cvSetSignal \a mask parameter: LED 2 (for 3-color: blue)
#define cvLED3             (1U << 3)   ///< \ref cvSetSignal \a mask parameter: LED 3
#define cvLED4             (1U << 4)   ///< \ref cvSetSignal \a mask parameter: LED 4
#define cvLED5             (1U << 5)   ///< \ref cvSetSignal \a mask parameter: LED 5
#define cvLED6             (1U << 6)   ///< \ref cvSetSignal \a mask parameter: LED 6
#define cvLED7             (1U << 7)   ///< \ref cvSetSignal \a mask parameter: LED 7

// cvSetSignal: signal values
#define cvLEDon            (0xFF)      ///< \ref cvSetSignal \a signal parameter: pattern to turn any LED on
#define cvLEDoff           (0x00)      ///< \ref cvSetSignal \a signal parameter: pattern to turn any LED off


// cvGetSignal: mask values and return values
#define cvBUTTON0          (1U << 0)   ///< \ref cvGetSignal \a mask parameter: Push button 0
#define cvBUTTON1          (1U << 1)   ///< \ref cvGetSignal \a mask parameter: Push button 1
#define cvBUTTON2          (1U << 2)   ///< \ref cvGetSignal \a mask parameter: Push button 2
#define cvBUTTON3          (1U << 3)   ///< \ref cvGetSignal \a mask parameter: Push button 3
#define cvJOYup            (1U << 4)   ///< \ref cvGetSignal \a mask parameter: Joystick button: up
#define cvJOYdown          (1U << 5)   ///< \ref cvGetSignal \a mask parameter: Joystick button: down
#define cvJOYleft          (1U << 6)   ///< \ref cvGetSignal \a mask parameter: Joystick button: left
#define cvJOYright         (1U << 7)   ///< \ref cvGetSignal \a mask parameter: Joystick button: right
#define cvJOYselect        (1U << 8)   ///< \ref cvGetSignal \a mask parameter: Joystick button: select
#define cvJOYall           (cvJOYup     | \
                            cvJOYdown   | \
                            cvJOYleft   | \
                            cvJOYright  | \
                            cvJOYselect | ) ///< \ref cvGetSignal \a mask Joystick button: all


// cvSetValue / cvGetValue: id values
#define cvAIN0             (0U)        ///< \ref cvSetValue / \ref cvGetValue \a id parameter: Analog input value 0
#define cvAIN1             (1U)        ///< \ref cvSetValue / \ref cvGetValue \a id parameter: Analog input value 1
#define cvAIN2             (2U)        ///< \ref cvSetValue / \ref cvGetValue \a id parameter: Analog input value 2
#define cvAIN3             (3U)        ///< \ref cvSetValue / \ref cvGetValue \a id parameter: Analog input value 3
#define cvAOUT0            (3U)        ///< \ref cvSetValue / \ref cvGetValue \a id parameter: Analog output value 0


// cvSetXYZ / cvGetXZY: id values
#define cvMotionGyro       (0U)        ///< \ref cvSetXYZ / \ref cvGetXYZ \a id parameter: for Gyroscope
#define cvMotionAccelero   (1U)        ///< \ref cvSetXYZ / \ref cvGetXYZ \a id parameter: for Accelerometer
#define cvMotionMagneto    (2U)        ///< \ref cvSetXYZ / \ref cvGetXYZ \a id parameter: for Magnetometer


// cvPrint: levels
#define cvLevelNone        (0U)        ///< \ref cvPrint \a level parameter: None
#define cvLevelHeading     (1U)        ///< \ref cvPrint \a level parameter: Heading
#define cvLevelMessage     (2U)        ///< \ref cvPrint \a level parameter: Message
#define cvLevelError       (3U)        ///< \ref cvPrint \a level parameter: Error

/// 3-D vector value
typedef struct cvValueXYZ {
  int32_t   X;                         ///< X coordinate
  int32_t   Y;                         ///< X coordinate
  int32_t   Z;                         ///< X coordinate
} cvValueXYZ_t;

/// IPv4 Internet Address
typedef struct cvAddrIPv4 {
  uint8_t   addr[4];                   ///< IPv4 address value used in \ref cvSetIPv4 / \ref cvGetIPv4 
} cvAddrIPv4_t;

/// IPv6 Internet Address
typedef struct cvAddrIPv6 {
  uint8_t   addr[16];                  ///< IPv6 address value used in \ref cvSetIPv6 / \ref cvGetIPv6
} cvAddrIPv6_t;

#ifdef  __cplusplus
extern "C"
{
#endif

/// Initialize test input, output.
/// \return none
void cvInit (void);

/// Print formated string to test terminal.
/// \param[in]     level        level (cvLevel...).
/// \param[in]     format       formated string to print.
/// \param[in]     ...          optional arguments (depending on format string).
/// \return number of characters written or -1 in case of error.
int32_t cvPrint (uint32_t level, const char *format, ...);

/// Get character from test terminal.
/// \return character from terminal or -1 in case of no character or error.
int32_t cvGetChar (void);

/// Set signal output.
/// \param[in]     mask         bit mask of signals to set.
/// \param[in]     signal       signal value to set (LEDs bit 0..15,  Switches bit 16..31).
/// \return none
void cvSetSignal (uint32_t mask, uint32_t signal);

/// Get signal input.
/// \param[in]     mask         bit mask of signals to read.
/// \return signal value or highest bit set in case of error.
uint32_t cvGetSignal (uint32_t mask);

/// Set value output.
/// \param[in]     id           output identifier.
/// \param[in]     value        value to set.
/// \return none
void cvSetValue (uint32_t id, int32_t value);

/// Get value input.
/// \param[in]     id           input identifier.
/// \return  value retrieved from input.
int32_t cvGetValue (uint32_t id);

/// Set XYZ value output.
/// \param[in]     id           output identifier.
/// \param[in]     valueXYZ     XYZ data to set.
/// \return none
void cvSetXYZ (uint32_t id, cvValueXYZ_t valueXYZ);

/// Get XYZ value input.
/// \param[in]     id           input identifier.
/// \return  XYZ data retrieved from XYZ peripheral.
cvValueXYZ_t cvGetXYZ (uint32_t id);

/// Set IPv4 address output.
/// \param[in]     id           output identifier.
/// \param[in]     addrIPv4     pointer to IPv4 address.
/// \return none
void cvSetIPv4 (uint32_t id, cvAddrIPv4_t addrIPv4);

/// Get IPv4 address input.
/// \param[in]     id           input identifier.
/// \return IPv4 address retrieved from peripheral
cvAddrIPv4_t cvGetIPv4 (uint32_t id);

/// Set IPv6 address output.
/// \param[in]     id           output identifier.
/// \param[in]     addrIPv6     pointer to IPv6 address.
/// \return none
void cvSetIPv6 (uint32_t id, cvAddrIPv6_t addrIPv6);

/// Get IPv6 address from peripheral.
/// \param[in]     id           input identifier.
/// \return IPv6 address retrieved from peripheral
cvAddrIPv6_t cvGetIPv6 (uint32_t id);

#ifdef  __cplusplus
}
#endif

#endif /* __CMSIS_VIO_H */
