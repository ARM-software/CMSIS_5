/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        coef.h
 * Description:  Parameters for the ML model and filter
 *
 * $Date:        16 March 2022
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2022 ARM Limited or its affiliates. All rights reserved.
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

#ifndef COEF_H
#define COEF_H

#define NUMTAPS 10
#define AUDIOBUFFER_LENGTH 512

extern const q15_t fir_coefs[NUMTAPS];

extern const q15_t coef_q15[98];
extern const q15_t intercept_q15 ;
extern const int coef_shift;
extern const int intercept_shift;
extern const q15_t window[400];

#endif
