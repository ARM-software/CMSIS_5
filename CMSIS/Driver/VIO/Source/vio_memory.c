/******************************************************************************
 * @file     vio_memory.c
 * @brief    Virtual I/O implementation using memory only
 * @version  V1.0.0
 * @date     18. March 2020
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

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "cmsis_vio.h"

#include "RTE_Components.h"             // Component selection
#include CMSIS_device_header

// CV input, output definitions
#define CV_PRINT_MAX_SIZE       64U     // maximum size of print memory
#define CV_PRINTMEM_NUM          4U     // number of print memories
#ifndef CV_VALUE_NUM
#define CV_VALUE_NUM             3U     // number of values
#endif
#ifndef CV_VALUEXYZ_NUM
#define CV_VALUEXYZ_NUM          3U     // number of XYZ values
#endif
#ifndef CV_IPV4_ADDRESS_NUM
#define CV_IPV4_ADDRESS_NUM      2U     // number of IPv4 addresses
#endif
#ifndef CV_IPV6_ADDRESS_NUM
#define CV_IPV6_ADDRESS_NUM      2U     // number of IPv6 addresses
#endif

// CV input, output variables
__USED uint32_t     cvSignalIn;
__USED uint32_t     cvSignalOut;
__USED char         cvPrintMem[CV_PRINTMEM_NUM][CV_PRINT_MAX_SIZE];
__USED int32_t      cvValue   [CV_VALUE_NUM];
__USED cvValueXYZ_t cvValueXYZ[CV_VALUEXYZ_NUM];
__USED cvAddrIPv4_t cvAddrIPv4[CV_IPV4_ADDRESS_NUM];
__USED cvAddrIPv6_t cvAddrIPv6[CV_IPV6_ADDRESS_NUM];

// Initialize test input, output.
void cvInit (void) {

  cvSignalIn  = 0U;
  cvSignalOut = 0U;

  memset (cvPrintMem, 0, sizeof(cvPrintMem));
  memset (cvValue,    0, sizeof(cvValue));
  memset (cvValueXYZ, 0, sizeof(cvValueXYZ));
  memset (cvAddrIPv4, 0, sizeof(cvAddrIPv4));
  memset (cvAddrIPv6, 0, sizeof(cvAddrIPv6));
}

// Print formated string to test terminal.
int32_t cvPrint (uint32_t level, const char *format, ...) {
  va_list args;
  int32_t ret;

  if (level > cvLevelError) {
    return (-1);
  }

  if (level > CV_PRINTMEM_NUM) {
    return (-1);
  }

  va_start(args, format);

  ret = vsnprintf((char *)cvPrintMem[level], sizeof(cvPrintMem[level]), format, args);

  va_end(args);

  return (ret);
}

// Get character from test terminal.
int32_t cvGetChar (void) {
  int32_t ch = -1;

  return ch;
}

// Set signal output.
void cvSetSignal (uint32_t mask, uint32_t signal) {

  cvSignalOut &= ~mask;
  cvSignalOut |=  mask & signal;
}

// Get signal input.
uint32_t cvGetSignal (uint32_t mask) {
  uint32_t signal;

  signal = cvSignalIn;

  return (signal & mask);
}

// Set value output.
void cvSetValue (uint32_t id, int32_t value) {
  uint32_t index = id;

  if (index >= CV_VALUE_NUM) {
    return;                             /* return in case of out-of-range index */
  }

  cvValue[index] = value;
}

// Get value input.
int32_t cvGetValue (uint32_t id) {
  uint32_t index = id;
  int32_t  value = 0;

  if (index >= CV_VALUE_NUM) {
    return value;                       /* return default in case of out-of-range index */
  }

  value = cvValue[index];

  return value;
}

// Set XYZ value output.
void cvSetXYZ (uint32_t id, cvValueXYZ_t valueXYZ) {
  uint32_t index = id;

  if (index >= CV_VALUEXYZ_NUM) {
    return;                             /* return in case of out-of-range index */
  }

  cvValueXYZ[index] = valueXYZ;
}

// Get XYZ value input.
cvValueXYZ_t cvGetXYZ (uint32_t id) {
  uint32_t index = id;
  cvValueXYZ_t valueXYZ = {0, 0, 0};

  if (index >= CV_VALUEXYZ_NUM) {
    return valueXYZ;                    /* return default in case of out-of-range index */
  }

  valueXYZ = cvValueXYZ[index];

  return valueXYZ;
}

// Set IPv4 address output.
void cvSetIPv4 (uint32_t id, cvAddrIPv4_t addrIPv4) {
  uint32_t index = id;

  if (index >= CV_IPV4_ADDRESS_NUM) {
    return;                             /* return in case of out-of-range index */
  }

  cvAddrIPv4[index] = addrIPv4;
}

// Get IPv4 address input.
cvAddrIPv4_t cvGetIPv4 (uint32_t id) {
  uint32_t index = id;
  cvAddrIPv4_t addrIPv4 = {0U, 0U, 0U, 0U};

  if (index >= CV_IPV4_ADDRESS_NUM) {
    return addrIPv4;                    /* return default in case of out-of-range index */
  }

  addrIPv4 = cvAddrIPv4[index];

  return addrIPv4;
}

// Set IPv6 address output.
void cvSetIPv6 (uint32_t id, cvAddrIPv6_t addrIPv6) {
  uint32_t index = id;

  if (index >= CV_IPV6_ADDRESS_NUM) {
    return;                             /* return in case of out-of-range index */
  }

  cvAddrIPv6[index] = addrIPv6;
}

// Get IPv6 address input.
cvAddrIPv6_t cvGetIPv6 (uint32_t id) {
  uint32_t index = id;
  cvAddrIPv6_t addrIPv6 = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
                           0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};

  if (index >= CV_IPV6_ADDRESS_NUM) {
    return addrIPv6;                    /* return default in case of out-of-range index */
  }

  addrIPv6 = cvAddrIPv6[index];

  return addrIPv6;
}
