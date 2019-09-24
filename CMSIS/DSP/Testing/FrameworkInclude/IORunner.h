/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        IORunner.h
 * Description:  IORunner Header
 *
 * $Date:        20. June 2019
 * $Revision:    V1.0.0
 *
 * Target Processor: Cortex-M cores
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
#ifndef _IORUNNER_H_
#define _IORUNNER_H_

#include "Test.h"

namespace Client
{
  class IORunner:public Runner
  {
     public:
      IORunner(IO*,PatternMgr*);
      IORunner(IO*,PatternMgr*, Testing::RunningMode);
      ~IORunner();
      virtual Testing::TestStatus run(Suite *s);
      virtual Testing::TestStatus run(Group *g);
     private:
      IO *m_io;
      PatternMgr *m_mgr;
      Testing::RunningMode m_runningMode;

      Testing::cycles_t calibration = 0;
  };

}

#endif
