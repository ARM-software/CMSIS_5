/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Semihosting.h
 * Description:  Semihosting Header
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
#ifndef _SEMIHOSTING_H_
#define _SEMIHOSTING_H_
#include <string>
#include <memory>
#include <stdio.h>
#include "arm_math.h"
#include "arm_math_f16.h"


namespace Client
{
/*

Semihosting driver. Used to read a text file describing how to drive the test.


*/

 struct pathOrGen;

 class Semihosting:public IO
  {
     public:
      Semihosting(std::string path,std::string patternRootPath
            ,std::string outputRootPath
            ,std::string parameterRootPath);
      ~Semihosting();
      virtual void ReadIdentification();
      virtual void ReadTestIdentification();
      virtual Testing::nbParameters_t ReadNbParameters();
      virtual void DispStatus(Testing::TestStatus,Testing::errorID_t,unsigned long,Testing::cycles_t);
      virtual void DispErrorDetails(const char* );
      virtual void EndGroup();
      virtual void ImportPattern(Testing::PatternID_t);
      virtual void ReadPatternList();
      virtual void ReadOutputList();
      virtual void ReadParameterList(Testing::nbParameters_t);
      virtual Testing::nbSamples_t GetPatternSize(Testing::PatternID_t);
      //Testing::nbSamples_t GetParameterSize(Testing::PatternID_t);
      
      virtual void ImportPattern_f64(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_f32(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
      virtual void ImportPattern_f16(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
#endif
      virtual void ImportPattern_q63(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_q31(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_q15(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_q7(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_u32(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_u16(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);
      virtual void ImportPattern_u8(Testing::PatternID_t,char*,Testing::nbSamples_t nb=0);

      virtual void DumpParams(std::vector<Testing::param_t>&);
      virtual Testing::param_t* ImportParams(Testing::PatternID_t,Testing::nbParameterEntries_t &,Testing::ParameterKind &);

      virtual bool hasParam();
      virtual Testing::PatternID_t getParamID();

      virtual void DumpPattern_f64(Testing::outputID_t,Testing::nbSamples_t nb, float64_t*);
      virtual void DumpPattern_f32(Testing::outputID_t,Testing::nbSamples_t nb, float32_t*);
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
      virtual void DumpPattern_f16(Testing::outputID_t,Testing::nbSamples_t nb, float16_t*);
#endif
      virtual void DumpPattern_q63(Testing::outputID_t,Testing::nbSamples_t nb, q63_t*);
      virtual void DumpPattern_q31(Testing::outputID_t,Testing::nbSamples_t nb, q31_t*);
      virtual void DumpPattern_q15(Testing::outputID_t,Testing::nbSamples_t nb, q15_t*);
      virtual void DumpPattern_q7(Testing::outputID_t,Testing::nbSamples_t nb, q7_t*);
      virtual void DumpPattern_u32(Testing::outputID_t,Testing::nbSamples_t nb, uint32_t*);
      virtual void DumpPattern_u16(Testing::outputID_t,Testing::nbSamples_t nb, uint16_t*);
      virtual void DumpPattern_u8(Testing::outputID_t,Testing::nbSamples_t nb, uint8_t*);
      
      virtual Testing::testID_t CurrentTestID();
     private:
      void DeleteParams();
      void recomputeTestDir();
      // Get the path to a pattern from a pattern ID
      std::string getPatternPath(Testing::PatternID_t id);
      // Get a path to an output file from an ouput category
      // (test ID will be used so same output ID
      // is giving different names for different tests)
      std::string getOutputPath(Testing::outputID_t id);
      // Get description of parameters from parameter ID
      struct pathOrGen getParameterDesc(Testing::PatternID_t id);
      //  Get file size from local path    
      Testing::nbSamples_t GetFileSize(std::string &path);
      // Driver filer controlling the tests.
      FILE*  infile;
      // Node description (group, suite or test)
      int currentKind;
      // Node ID
      Testing::testID_t currentId;
      // Current param ID for the node
      Testing::PatternID_t currentParam;
      bool m_hasParam;

      // Current path for the node
      // (It is not the full path but the path added by this node)
      std::string currentPath;

      // Contains the current test directory
      // (where to find pattenrs)
      std::string testDir;
      // Array of folder used to build the 
      // testDir parth
      std::vector<std::string> *path;
      // Root directory for patterns
      std::string patternRootPath;
      // Root directory for output
      std::string outputRootPath;
      // Root directory for parameters
      std::string parameterRootPath;
      // List of pattern filenames
      // Used to find a name from a pattern ID
      std::vector<std::string> *patternFilenames;
      // List of output names
      // Used to find a name from an output ID
      std::vector<std::string> *outputNames;
      // List of parameters descriptions
      // Used to find a path or generator from a parameter ID
      std::vector<struct pathOrGen> *parameterNames;
  };
}

#endif
