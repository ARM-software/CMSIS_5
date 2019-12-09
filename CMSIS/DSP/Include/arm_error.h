/******************************************************************************
 * @file     arm_error.h
 * @brief    Public header file for CMSIS DSP Library
 * @version  V1
 * @date     13. December 2019
 ******************************************************************************/
/*
 * Copyright (c) 2010-2019 Arm Limited or its affiliates. All rights reserved.
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

#ifndef _ARM_ERROR_H
#define _ARM_ERROR_H

#ifdef   __cplusplus
extern "C"
{
#endif

typedef enum
{
  ARM_ERROR_MATH      = -1,       /**< Mathematical error */
  ARM_ERROR_ALIGNMENT = -2        /**< Alignment error */
} arm_error;

/**
 * @brief  Error handler function used for post-mortem analysis and debug
 * @param[in]       error_code   Error code
 * @param[in, out]  error_desc   Error description
 * @return          none
 *
 */
void arm_error_handler(arm_error error_code, const char *error_desc);

#ifdef   __cplusplus
}
#endif


#endif /* _ARM_ERROR_H */

/**
 *
 * End of file.
 */
