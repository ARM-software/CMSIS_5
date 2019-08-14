/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        FPGA.h
 * Description:  FPGA Header
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
#ifndef _FPGA_H_
#define _FPGA_H_
#include <string>

namespace Client
{

/*

FPGA driver. Used to read a C array describing how to drive the test.


*/

 struct offsetOrGen;

 class FPGA:public IO
  {
     public:
      FPGA(const char *testDesc,const char *patterns);
      ~FPGA();
      void ReadIdentification();
      void ReadTestIdentification();
      Testing::nbParameters_t ReadNbParameters();
      void DispStatus(Testing::TestStatus,Testing::errorID_t,unsigned long,Testing::cycles_t);
      void EndGroup();
      void ImportPattern(Testing::PatternID_t);
      void ReadPatternList();
      void ReadOutputList();
      void ReadParameterList(Testing::nbParameters_t);
      Testing::nbSamples_t GetPatternSize(Testing::PatternID_t);

      void ImportPattern_f64(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_f32(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_q31(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_q15(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_q7(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_u32(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_u16(Testing::PatternID_t,char*,Testing::nbSamples_t nb);
      void ImportPattern_u8(Testing::PatternID_t,char*,Testing::nbSamples_t nb);

      void DumpParams(std::vector<Testing::param_t>&);
      Testing::param_t* ImportParams(Testing::PatternID_t,Testing::nbParameterEntries_t &,Testing::ParameterKind &);
      bool hasParam();
      Testing::PatternID_t getParamID();

      void DumpPattern_f64(Testing::outputID_t,Testing::nbSamples_t nb, float64_t* data);
      void DumpPattern_f32(Testing::outputID_t,Testing::nbSamples_t nb, float32_t* data);
      void DumpPattern_q31(Testing::outputID_t,Testing::nbSamples_t nb, q31_t* data);
      void DumpPattern_q15(Testing::outputID_t,Testing::nbSamples_t nb, q15_t* data);
      void DumpPattern_q7(Testing::outputID_t,Testing::nbSamples_t nb, q7_t* data);
      void DumpPattern_u32(Testing::outputID_t,Testing::nbSamples_t nb, uint32_t* data);
      void DumpPattern_u16(Testing::outputID_t,Testing::nbSamples_t nb, uint16_t* data);
      void DumpPattern_u8(Testing::outputID_t,Testing::nbSamples_t nb, uint8_t* data);
      
      Testing::testID_t CurrentTestID();
     private:
      void recomputeTestDir();
      void DeleteParams();
      struct offsetOrGen getParameterDesc(Testing::PatternID_t id);
      // Get offset in C array of a pattern.
      unsigned long getPatternOffset(Testing::PatternID_t);
      // Get output path
      std::string getOutputPath(Testing::outputID_t id);
      // Get offset in C array of a parameter.
      unsigned long getParameterOffset(Testing::PatternID_t);
      // Read data from the driver C array.
      void read32(unsigned long *);
      void readStr(char *str);
      void readChar(char *);

      // Driver array
      const char *m_testDesc;

      // Pattern array
      const char *m_patterns;

      // Parameter array
      char *m_parameters;

      // Current position in the driver array
      const char *currentDesc;
      int currentKind;
      Testing::testID_t currentId;
      // Current param ID for the node
      Testing::PatternID_t currentParam;
      bool m_hasParam;

      // Association pattern ID to pattern offset in C array m_patterns
      std::vector<unsigned long> *patternOffsets;

      // Association pattern ID to pattern size.
      std::vector<Testing::nbSamples_t> *patternSizes;

      // Association parameter ID to parameter offset in C array m_parameters
      std::vector<struct offsetOrGen> *parameterOffsets;

      // Association parameter ID to parameter size.
      std::vector<Testing::nbSamples_t> *parameterSizes;

      // testDir is only used for output.
      // In a future version it will be removed.
      // Output will just use ID and post processing
      // script will recover the path
      std::string testDir;
      std::vector<std::string> *path;
      std::string currentPath;
      std::vector<std::string> *outputNames;

  };
}

#endif