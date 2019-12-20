/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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

#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

inline int validate(int8_t *act, const int8_t *ref, int size)
{
    int test_passed = true;
    int count = 0;
    int total = 0;

    for(int i = 0; i < size; ++i)
    {
      total++;
      if(act[i] != ref[i])
      {
        count++;
        printf("ERROR at pos %d: Act: %d Ref: %d\r\n", i, act[i], ref[i]);
        test_passed = false;
      }
      else
      {
        //printf("PASS at pos %d: %d\r\n", i, act[i]);
      }
    }

    if (!test_passed)
    {
      printf("%d of %d failed\r\n", count, total);
    }

    return test_passed;
}
