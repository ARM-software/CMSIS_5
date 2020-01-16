/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        IORunner.cpp
 * Description:  IORunner
 *
 *               Runner implementation for runner running on device 
 *               under test
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
#include "Test.h"

#include <string>
#include <cstddef>
#include <stdlib.h>
#include <stdio.h>
#include "IORunner.h"
#include "Error.h"
#include "Timing.h"
#include "arm_math.h"
#include "Calibrate.h"

namespace Client
{
  
      IORunner::IORunner(IO *io,PatternMgr *mgr,  Testing::RunningMode runningMode):m_io(io), m_mgr(mgr)
      {
        this->m_runningMode = runningMode;
        // Set running mode on PatternMgr.
        if (runningMode == Testing::kDumpOnly)
        {
          mgr->setDumpMode();
        }
        if (runningMode == Testing::kTestAndDump)
        {
          mgr->setTestAndDumpMode();
        }

        initCycleMeasurement();

/* 

For calibration :

Calibration means, in this context, removing the overhad of calling
a C++ function pointer from the cycle measurements.


*/
        Calibrate c((Testing::testID_t)0);
        Client::Suite *s=(Client::Suite *)&c;
        Client::test t = (Client::test)&Calibrate::empty;

        cycleMeasurementStart();
/* 

EXTBENCH is set when benchmarking is done through external traces
instead of using internal counters.

Currently the post-processing scripts are only supporting traces generated from
fast models.

*/
#ifdef EXTBENCH
        startSection();
#endif

/*

For calibration, we measure the time it takes to call 20 times an empty benchmark and compute
the average.
(20 is an arbitrary value.)

This overhead is removed from benchmarks in the Runner..

Calibration is removed from the python script when external trace is used for the cycles.
Indeed, in that case the calibration value can only be measured by parsing the trace.

Otherwise, the calibration is measured below.

*/
        for(int i=0;i < 20;i++)
        {
          if (!m_mgr->HasMemError())
          {
             (s->*t)();
          }
        }
#ifdef EXTBENCH
        stopSection();
#endif
#ifndef EXTBENCH
        calibration=getCycles() / 20;
#endif
        cycleMeasurementStop();

      }

      // Testing.
      // When false we are in dump mode and the failed assertion are ignored
      // (But exception is taken so assert should be at end of the test and not in the
      // middle )
      IORunner::IORunner(IO *io,PatternMgr *mgr):m_io(io), m_mgr(mgr)
      {
        this->m_runningMode = Testing::kTestOnly;
      }

      IORunner::~IORunner()
      {
        
      }

     
      /** Read driver data to control execution of a suite
      */
      Testing::TestStatus IORunner::run(Suite *s) 
      {
        Testing::TestStatus finalResult = Testing::kTestPassed;
        int nbTests = s->getNbTests();
        int failedTests=0;
        Testing::errorID_t error=0;
        unsigned long line = 0;
        char details[200];
        Testing::cycles_t cycles=0;
        Testing::nbParameters_t nbParams;

        // Read node identification (suite)
        m_io->ReadIdentification();
        // Read suite nb of parameters 
        nbParams = m_io->ReadNbParameters();

        // Read list of patterns
        m_io->ReadPatternList();
        // Read list of output
        m_io->ReadOutputList();
        // Read list of parameters
        m_io->ReadParameterList(nbParams);

        // Iterate on tests
        for(int i=1; i <= nbTests; i++)
        {
            test t = s->getTest(i);
            Testing::TestStatus result = Testing::kTestPassed;
            error = UNKNOWN_ERROR;
            line = 0;
            cycles = 0;
            details[0]='\0';
            Testing::param_t *paramData=NULL;
            Testing::nbParameterEntries_t entries=0;
            std::vector<Testing::param_t> params(nbParams);
            bool canExecute=true;
            int  dataIndex=0;
            Testing::ParameterKind paramKind;

            // Read test identification (test ID)
            m_io->ReadTestIdentification();
            
            
            if (m_io->hasParam())
            {
               Testing::PatternID_t paramID=m_io->getParamID();
               paramData = m_io->ImportParams(paramID,entries,paramKind);
               dataIndex = 0;
            }


            while(canExecute)
            {
              canExecute = false; 
              
              if (m_io->hasParam() && paramData)
              {
                // Load new params
                for(int j=0; j < nbParams ; j++)
                {
                  params[j] = paramData[nbParams*dataIndex+j];
                }
                // Update condition for new execution
                dataIndex += 1;
                canExecute = dataIndex < entries;
              }
              // Execute test
              try {     
                // Prepare memory for test
                // setUp will generally load patterns
                // and do specific initialization for the tests
                s->setUp(m_io->CurrentTestID(),params,m_mgr);
                
                   // Run the test
                cycleMeasurementStart();
#ifdef EXTBENCH
                startSection();
#endif
                if (!m_mgr->HasMemError())
                {
                    (s->*t)();
                }
#ifdef EXTBENCH
                stopSection();
#endif
#ifndef EXTBENCH
                cycles=getCycles()-calibration;
#endif
                cycleMeasurementStop();
              } 
              catch(Error &ex)
              {
                 cycleMeasurementStop();
                 // In dump only mode we ignore the tests 
                 // since the reference patterns are not loaded
                 // so tests will fail.
                 if (this->m_runningMode != Testing::kDumpOnly)
                 {
                    error = ex.errorID;
                    line = ex.lineNumber;
                    strcpy(details,ex.details);
                    result=Testing::kTestFailed;
                 }
              }
              catch (...) { 
                cycleMeasurementStop();
                // In dump only mode we ignore the tests 
                // since the reference patterns are not loaded
                // so tests will fail.
                if (this->m_runningMode != Testing::kDumpOnly)
                {
                  result = Testing::kTestFailed;
                  error = UNKNOWN_ERROR;
                  line = 0;
                }
              }
              try { 
                 // Clean memory after this test
                 // May dump output and do specific cleaning for a test
                 s->tearDown(m_io->CurrentTestID(),m_mgr);
              }
              catch(...)
              {
              
              }

              if (m_mgr->HasMemError())
              {
                /* We keep the current error if set.
                */
                if (result == Testing::kTestPassed)
                {
                  result = Testing::kTestFailed;
                  error = MEMORY_ALLOCATION_ERROR;
                  line = 0;
                }
              }
              
              // Free all memory of memory manager so that next test
              // is starting in a clean and controlled tests
              m_mgr->freeAll();
  
              // Dump test status to output
              m_io->DispStatus(result,error,line,cycles);
              m_io->DispErrorDetails(details);
              m_io->DumpParams(params);
            }
            if (paramData)
            {
                if (paramKind == Testing::kDynamicBuffer)
                {
                  free(paramData);
                }
                paramData = NULL;
            }

            if (result == Testing::kTestFailed)
            {
              failedTests ++;
              finalResult = Testing::kTestFailed;
            }
        }
        // Signal end of group processing to output
        m_io->EndGroup();
        return(finalResult);
     }

      /** Read driver data to control execution of a group
      */
      Testing::TestStatus IORunner::run(Group *g) 
      {
        int nbTests = g->getNbContainer();
        int failedTests=0;


        // Read Node identification
        m_io->ReadIdentification();
        

        Testing::TestStatus finalResult = Testing::kTestPassed;
        // Iterate on group elements
        for(int i=1; i <= nbTests; i++)
        {
            TestContainer *c = g->getContainer(i);
            if (c != NULL)
            {
                // Execute runner for this group
                Testing::TestStatus result = c->accept(this);

                if (result == Testing::kTestFailed)
                {
                   failedTests ++;
                   finalResult = Testing::kTestFailed;
                }
            }
            
        }
        // Signal to output that processing of this group has finished.
        m_io->EndGroup();
        return(finalResult);
      }


}
