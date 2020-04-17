/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_svm_polynomial_predict_f32.c
 * Description:  SVM Polynomial Classifier
 *
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2019 ARM Limited or its affiliates. All rights reserved.
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
#ifndef _SVMDEFH_
#define _SVMDEFH_

#include "arm_math.h"

typedef struct
{
  uint32_t        nbOfSupportVectors;     /**< Number of support vectors */
  uint32_t        vectorDimension;        /**< Dimension of vector space */
  float32_t       intercept;              /**< Intercept */
  const float32_t *dualCoefficients;      /**< Dual coefficients */
  const float32_t *supportVectors;        /**< Support vectors */
  const int32_t   *classes;               /**< The two SVM classes */
  int32_t         degree;                 /**< Polynomial degree */
  float32_t       coef0;                  /**< Polynomial constant */
  float32_t       gamma;                  /**< Gamma factor */
} arm_svm_polynomial_instance_f32;

void arm_svm_polynomial_predict_f32(const arm_svm_polynomial_instance_f32 *S, 
   const float32_t * in, 
   float32_t * pResult);


#endif
