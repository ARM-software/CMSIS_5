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
#include <cstdlib>
#include <cstdio>
#include "IORunner.h"
#include "Error.h"
#include "Timing.h"
#include "arm_math_types.h"
#include "Calibrate.h"

#ifdef CORTEXA
#define CALIBNB 1
#else
#define CALIBNB 20
#endif

using namespace std;

namespace Client
{
  
      IORunner::IORunner(IO *io,PatternMgr *mgr,  Testing::RunningMode runningMode):m_io(io), m_mgr(mgr)
      {
        volatile Testing::cycles_t current;

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
        calibration = 0;
        


/*

For calibration, we measure the time it takes to call 20 times an empty benchmark and compute
the average.
(20 is an arbitrary value.)

This overhead is removed from benchmarks in the Runner..

Calibration is removed from the python script when external trace is used for the cycles.
Indeed, in that case the calibration value can only be measured by parsing the trace.

Otherwise, the calibration is measured below.

*/

/*

We want to ensure that the calibration of the overhead of the
measurement is the same here and when we do the measurement later.

So to ensure the conditions are always the same, the instruction cache
and branch predictor are flushed.

*/
#ifdef CORTEXA
  __set_BPIALL(0);
  __DSB();
  __ISB();

  __set_ICIALLU(0);
  __DSB();
  __ISB();
#endif

/*

We always call the empty function once to ensure it is in the cache
because it is how the measurement is done.

*/
        if (!m_mgr->HasMemError())
        {
             (s->*t)();
        }

/*

We measure the cycles required for a measurement,
The cycleMeasurement starts, getCycles and cycleMeasurementStop
should not be in the cache.

So, for the overhead we always have the value corresponding to
the code not in cache.

While for the code itself we have the value for the code in cache.

*/

/* 

EXTBENCH is set when benchmarking is done through external traces
instead of using internal counters.

Currently the post-processing scripts are only supporting traces generated from
fast models.

*/
#if defined(EXTBENCH)  || defined(CACHEANALYSIS)
        startSection();
#endif
        
        for(int i=0;i < CALIBNB;i++)
        {
          cycleMeasurementStart();
          if (!m_mgr->HasMemError())
          {
             (s->*t)();
          }
          #ifndef EXTBENCH
             current = getCycles();
          #endif
          calibration += current;
          cycleMeasurementStop();
        }
#if defined(EXTBENCH)  || defined(CACHEANALYSIS)
        stopSection();
#endif

#ifndef EXTBENCH
        calibration=calibration / CALIBNB;
#endif
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
        volatile Testing::cycles_t cycles=0;
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
            unsigned long  dataIndex=0;
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
                for(unsigned long j=0; j < nbParams ; j++)
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
                
                // Run the test once to force the code to be in cache.
                // By default it is disabled in the suite.
#ifdef CORTEXA
  __set_BPIALL(0);
  __DSB();
  __ISB();

  __set_ICIALLU(0);
  __DSB();
  __ISB();
#endif

/* If cache analysis mode, we don't force the code to be in cache. */
#if !defined(CACHEANALYSIS)
                if (s->isForcedInCache())
                {
                   if (!m_mgr->HasMemError())
                   {
                      (s->*t)();
                   }
                }
#endif
                // Run the test
                cycleMeasurementStart();

#if defined(EXTBENCH) || defined(CACHEANALYSIS)
                startSection();
#endif
                if (!m_mgr->HasMemError())
                {
                    (s->*t)();
                }

#if defined(EXTBENCH) || defined(CACHEANALYSIS)
                stopSection();
#endif

#ifndef EXTBENCH
                cycles=getCycles();
                cycles=cycles-calibration;
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
