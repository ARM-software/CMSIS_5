
/** \addtogroup hal */
/** @{*/
/*
 * Copyright (c) 2006-2016, ARM Limited, All Rights Reserved
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DRIVER_COMMON_H
#define __DRIVER_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/****** This file has been deprecated since mbed-os-5.5 *****/

#define ARM_DRIVER_VERSION_MAJOR_MINOR(major,minor) (((major) << 8) | (minor))

/**
\brief Driver Version
*/
typedef struct _ARM_DRIVER_VERSION {
  uint16_t api;                         ///< API version
  uint16_t drv;                         ///< Driver version
} ARM_DRIVER_VERSION;

/* General return codes */
#define ARM_DRIVER_OK                 0 ///< Operation succeeded
#define ARM_DRIVER_ERROR             -1 ///< Unspecified error
#define ARM_DRIVER_ERROR_BUSY        -2 ///< Driver is busy
#define ARM_DRIVER_ERROR_TIMEOUT     -3 ///< Timeout occurred
#define ARM_DRIVER_ERROR_UNSUPPORTED -4 ///< Operation not supported
#define ARM_DRIVER_ERROR_PARAMETER   -5 ///< Parameter error
#define ARM_DRIVER_ERROR_SPECIFIC    -6 ///< Start of driver specific errors

/**
\brief General power states
*/
typedef enum _ARM_POWER_STATE {
  ARM_POWER_OFF,                        ///< Power off: no operation possible
  ARM_POWER_LOW,                        ///< Low Power mode: retain state, detect and signal wake-up events
  ARM_POWER_FULL                        ///< Power on: full operation at maximum performance
} ARM_POWER_STATE;

#endif /* __DRIVER_COMMON_H */

/** @}*/
