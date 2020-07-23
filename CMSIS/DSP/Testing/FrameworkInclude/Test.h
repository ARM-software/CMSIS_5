/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Test.h
 * Description:  Test Framework Header
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
#ifndef _TEST_H_
#define _TEST_H_

#include <cstdlib>
#include <vector>
#include <cstdio>
#include "arm_math_types.h"
#include "arm_math_types_f16.h"


// This special value means no limit on the number of samples.
// It is used when importing patterns and we want to read
// all the samples contained in the pattern.
#define MAX_NB_SAMPLES 0

// Pattern files are containing hexadecimal values.
// So we need to be able to convert some int into float without convertion
#define TOINT16(v) *((uint16_t*)&v)
#define TOINT32(v) *((uint32_t*)&v)
#define TOINT64(v) *((uint64_t*)&v)
// Or convert some  float into a uint32 or uint64 without convertion
#define TOTYP(TYP,v) *((TYP*)&v)
// So it is a cast and not a data conversion.
// (uint32_t)1.0 would give 1. We want the hexadecimal representation of the float.
// TOINT32(1.0) can be used


namespace Testing 
{
  enum TestStatus
  {
    kTestFailed=0,
    kTestPassed=1
  };

  /* In Dump only, reference patterns are never read.
     So tests are failing
     and we dump output.
     In this mode we are only interested in the output data and
     not the test status.

     In test only mode, no output is dumped.
   */
  enum RunningMode
  {
    kTestOnly=0,
    kDumpOnly=1,
    kTestAndDump=2
  };

 


  // test ID are ID of nodes in the tree of tests.
  // So a group ID, suite ID or test ID all have the same type
  // testID_t
  typedef unsigned long testID_t;
  typedef unsigned long outputID_t;
  typedef unsigned long PatternID_t;

  typedef unsigned long testIndex_t;
  typedef unsigned long nbSamples_t;
  typedef unsigned long errorID_t;



  typedef uint32_t cycles_t;
  typedef unsigned long nbMeasurements_t;

  // parameter value 
  // (always int since we need to be able to iterate on
  // different parameter values which are often dimensions of
  // input data)
  typedef int param_t;
  // Number of parameters for a given configuration
  typedef unsigned long nbParameters_t;
  // Number of parameter configurations
  typedef unsigned long nbParameterEntries_t;

  // To know if parameter array is malloc buffer or static buffer in C array
  enum ParameterKind
  {
    kStaticBuffer=0,
    kDynamicBuffer=1,
  };

}

namespace Client
{

 

/*

Client code 

*/



  class Suite;
  class Group;


  // Type of a test function
  // It is not a function pointer (because the function is
  // a method of a CPP class)
  typedef void (Suite::*test)();

/*

API of Memory managers used in the test framework

*/
  class Memory
{
  public:
    // Allocate a new buffer of size length
    // and generate a pointer to this buffer.
    // It does not imply that any malloc is done.
    // It depends on how the Memory manager is implemented.
    virtual char *NewBuffer(size_t length)=0;

    // Free all the memory allocated by the memory manager
    // and increment the memory generation number.
    virtual void FreeMemory()=0;

    // Memory allocation errors must be tracked during a test.
    // The runner should force the test status to FAILED
    // when a memory error occured.
    virtual bool HasMemError()=0;

    // When memory manager is supporting tail
    // then we can check that the tail of the buffer has not been 
    // corrupted.
    // The tail being the additional words after the end of the buffer allocated
    // by the memory manager so that there is some seperation between
    // successive buffers.
    // When memory manager is not supporting tail, this function should
    // always succeed.
    virtual bool IsTailEmpty(char *, size_t)=0;


    // Get the memory generation number
    unsigned long generation()
    {
      return(m_generation);
    }

  protected:
    unsigned long m_generation=0;
};

 
  // A runner is a class driving the tests
  // It can use information from driving files
  // or in the future could communicate with a process
  // on a host computer which would be the real driver of the
  // testing.
  // It is following the visitor pattern. IT is the reason for an accept
  // function in Group class.
  // Run is the visitor
  class Runner
  {
    public:
      virtual Testing::TestStatus run(Suite*) = 0;
      virtual Testing::TestStatus run(Group*) = 0;
  };

  // Abstract the IO needs of the test framework.
  // IO could be done from semihosting, socket, C array in memory etc ...
  class IO
  {
    public:

      /** Read the identification of a node from driver data.

          Update the node kind and node id and local folder.
          To be used for group and suite. Generally update
          the path to the folder by using this new local folder
          which is appended to the path.
      */
      virtual void ReadIdentification()=0;

      /** Read the identification of a node  driver data.

          Update the node kind and node id and local folder.
      */
      virtual void ReadTestIdentification()=0;

      /** Read the number of parameters for all the tests in a suite

          Used for benchmarking. Same functions executed with
          different initializations controlled by the parameters.

      */
      virtual Testing::nbParameters_t ReadNbParameters()=0;

      /** Dump the test status

          For format of output, refer to Python script.
          The format must be coherent with the Python script
          parsing the output.
      */
      virtual void DispStatus(Testing::TestStatus,Testing::errorID_t,unsigned long,Testing::cycles_t cycles)=0;
      

      /** Dump additional details for the error

          For instance, for SNR error, it may contain the SNR value.
      */
      virtual void DispErrorDetails(const char* )=0;

      /** Dump parameters for a test

          When a test is run several time with different
          parameters for benchmarking,
          the parameters are displayed after test status.
          Line should begin with b
      */
      virtual void DumpParams(std::vector<Testing::param_t>&)=0;


      /** Dump an end of group/suite to output

          Used by Python script parsing the output.

      */
      virtual void EndGroup()=0;

      /** Get the zize of a pattern in this suite.

          Pattern is identified with an ID.
          Using the local path and ID, the IO implementatiom should
          be able to access the pattern.

          The path do not have to be a file path. Just a way
          to identify patterns in a suite and know
          how to access them.

      */
      virtual Testing::nbSamples_t GetPatternSize(Testing::PatternID_t)=0;
      
      /** Get the size of a parameter pattern in this suite.

          Parameter is identified with an ID.
          Using the local path and ID, the IO implementatiom should
          be able to access the data.

          The path do not have to be a file path. Just a way
          to identify data in a suite and know
          how to access them.

      */
      //virtual Testing::nbSamples_t GetParameterSize(Testing::PatternID_t id)=0;

      /** Check if some parameters are controlling this test
      */
      virtual bool hasParam()=0;

      /** Get ID of parameter generator
      */
      virtual Testing::PatternID_t getParamID()=0;

      /** Import pattern.

          The nb field can be used to limit the number of samples read
          to a smaller value than the number of samples available in the 
          pattern.

      */
      virtual void ImportPattern_f64(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_f32(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
      virtual void ImportPattern_f16(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
#endif
      virtual void ImportPattern_q63(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_q31(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_q15(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_q7(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_u32(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_u16(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;
      virtual void ImportPattern_u8(Testing::PatternID_t,char*,Testing::nbSamples_t nb=MAX_NB_SAMPLES)=0;

      /** Import params.

          This is allocating memory.
          The runner should free it after use.

          It is not using the Memory manager since tests don't have access
          to the array of parameters.

          They receive parameters as a vector argument for the setUp fucntion.
      */
      virtual Testing::param_t* ImportParams(Testing::PatternID_t,Testing::nbParameterEntries_t &,Testing::ParameterKind &)=0;

      /** Dump pattern.

          The output ID (and test ID) must be used to uniquely identify
          the dump.


      */
      virtual void DumpPattern_f64(Testing::outputID_t,Testing::nbSamples_t nb, float64_t*)=0;
      virtual void DumpPattern_f32(Testing::outputID_t,Testing::nbSamples_t nb, float32_t*)=0;
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
      virtual void DumpPattern_f16(Testing::outputID_t,Testing::nbSamples_t nb, float16_t*)=0;
#endif
      virtual void DumpPattern_q63(Testing::outputID_t,Testing::nbSamples_t nb, q63_t*)=0;
      virtual void DumpPattern_q31(Testing::outputID_t,Testing::nbSamples_t nb, q31_t*)=0;
      virtual void DumpPattern_q15(Testing::outputID_t,Testing::nbSamples_t nb, q15_t*)=0;
      virtual void DumpPattern_q7(Testing::outputID_t,Testing::nbSamples_t nb, q7_t*)=0;
      virtual void DumpPattern_u32(Testing::outputID_t,Testing::nbSamples_t nb, uint32_t*)=0;
      virtual void DumpPattern_u16(Testing::outputID_t,Testing::nbSamples_t nb, uint16_t*)=0;
      virtual void DumpPattern_u8(Testing::outputID_t,Testing::nbSamples_t nb, uint8_t*)=0;

      /** Import list of patterns from the driver 
          for current suite.

          This list is used to identify a pattern from its pattern ID.
          The information of this list (local to the suite) is
          combined with the path to identify patterns in other part of the class.

      */
      virtual void ReadPatternList()=0;

      /** Import list of output from the driver 
          for current suite.

          This list is used to identify an output from its pattern ID (and current test ID)
          The information of this list (local to the suite) is
          combined with the path and current test ID 
          to identify output in other part of the class.

      */
      virtual void ReadOutputList()=0;

      /** Import list of parameters from the driver 
          for current suite.

          This list is used to control a functions with different parameters
          for benchmarking purpose.

          A parameter can be a file of parameters or a generator
          of parameters (cartesian product of lists only).

      */
      virtual void ReadParameterList(Testing::nbParameters_t)=0;

      /** Get current node ID
          group, suite or test. A group of test is considered as a test hence
          the name of the function.

      */
      virtual Testing::testID_t CurrentTestID()=0;
  };


// A pattern manager is making the link between
// IO and the Memory manager.
// It knows how to import patterns into memory or dump
// memory into outputs (output which may be different from a file)
// The running mode is controlling if dumping is disabled or not.
// But cna also be used by the runner to know if test results must be ignored or not.
// Pattern manager is used by the tests
// In current version load and dump functions are visible to any body.
// In theory they should only be visible to Patterns
class PatternMgr
{
public:
    PatternMgr(IO*,Memory*);

    /** In those loading APIs, nb samples is coming from the pattern read.
        maxSamples is coming from the test.

        A test does not know what is the length of a pattern since the test
        has no visiblity on how the pattern is imported (file, serial port, include files
        etc ...).

        So the test is specifying the max sampls it needs.
        The pattern is specifying its lengths.

        Those functions are importing at most what is needed and what is available.

    */
    float64_t *load_f64(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
    float32_t *load_f32(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
    float16_t *load_f16(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
#endif
    q63_t *load_q63(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
    q31_t *load_q31(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
    q15_t *load_q15(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
    q7_t *load_q7(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);

    uint32_t *load_u32(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
    uint16_t *load_u16(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);
    uint8_t *load_u8(Testing::PatternID_t,Testing::nbSamples_t&,Testing::nbSamples_t maxSamples=MAX_NB_SAMPLES);

    /** Create a local pattern.

        Generally it is used as output of a test and has no
        correspondance to a pattern in the suite.

    */
    float64_t *local_f64(Testing::nbSamples_t);
    float32_t *local_f32(Testing::nbSamples_t);
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
    float16_t *local_f16(Testing::nbSamples_t);
#endif
    q63_t *local_q63(Testing::nbSamples_t);
    q31_t *local_q31(Testing::nbSamples_t);
    q15_t *local_q15(Testing::nbSamples_t);
    q7_t *local_q7(Testing::nbSamples_t);

    uint32_t *local_u32(Testing::nbSamples_t);
    uint16_t *local_u16(Testing::nbSamples_t);
    uint8_t *local_u8(Testing::nbSamples_t);

    /**  Dump a pattern

    */
    void dumpPattern_f64(Testing::outputID_t,Testing::nbSamples_t,float64_t*);
    void dumpPattern_f32(Testing::outputID_t,Testing::nbSamples_t,float32_t*);
#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
    void dumpPattern_f16(Testing::outputID_t,Testing::nbSamples_t,float16_t*);
#endif
    
    void dumpPattern_q63(Testing::outputID_t,Testing::nbSamples_t,q63_t*);
    void dumpPattern_q31(Testing::outputID_t,Testing::nbSamples_t,q31_t*);
    void dumpPattern_q15(Testing::outputID_t,Testing::nbSamples_t,q15_t*);
    void dumpPattern_q7(Testing::outputID_t,Testing::nbSamples_t,q7_t*);

    void dumpPattern_u32(Testing::outputID_t,Testing::nbSamples_t,uint32_t*);
    void dumpPattern_u16(Testing::outputID_t,Testing::nbSamples_t,uint16_t*);
    void dumpPattern_u8(Testing::outputID_t,Testing::nbSamples_t,uint8_t*);

    /** Free all allocated patterns.

        Just wrapper around the memory manager free function.

    */
    void freeAll();

    /** MeMory manager generation

    */
    unsigned long generation()
    {
      return(m_mem->generation());
    }

    // Memory allocation errors must be tracked during a test.
    // The runner should force the test status to FAILED
    // when a memory error occured.
    bool HasMemError()
    {
      return(m_mem->HasMemError());
    }

    // Set by the runner when in dump mode
    void setDumpMode()
    {
      this->m_runningMode = Testing::kDumpOnly;
    }

    void setTestAndDumpMode()
    {
      this->m_runningMode = Testing::kTestAndDump;
    }

    Testing::RunningMode runningMode()
    {
      return(this->m_runningMode);
    }

    bool IsTailEmpty(char *ptr, size_t length)
    {
        return(m_mem->IsTailEmpty(ptr,length));
    }

private:
    IO *m_io;
    Memory *m_mem;
    Testing::RunningMode m_runningMode=Testing::kTestOnly;
    
};

  // TestContainer which is a node of the tree of tests
  class TestContainer
  {
    public:
      TestContainer(Testing::testID_t);
      // Used for implementing the visitor pattern.
      // The visitor pattern is allowing to implement
      // different Runner for a tree of tests.
      virtual Testing::TestStatus accept(Runner* v) = 0;
    protected:
      // Node ID (test ID)
      Testing::testID_t m_containerID;
  };

  // A suite object
  // It contains a list of tests
  // Methods to get a test from the test ID
  // Initialization and cleanup (setUp and tearDown) to be called
  // between each test.
  // Those functions are used by the Runner to execute the tests
  class Suite:public TestContainer
  {
    public:
      Suite(Testing::testID_t);
  
      // Prepare memory for a test
      // (Load input and reference patterns)
      virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>&,PatternMgr *mgr)=0;

      // Clean memory after a test
      // Free all memory
      // DUmp outputs
      virtual void tearDown(Testing::testID_t,PatternMgr *mgr)=0;

      // Add a test to be run.
      void addTest(Testing::testID_t,test aTest);

      // Get a test from its index. Used by runner when iterating
      // on all the tests. Index is not the test ID.
      // It is the index in internal list of tests
      test getTest(Testing::testIndex_t);

      // Get number of test in this suite.
      // The suite is only containing the active tests
      // (deprecated tests are never generated by python scripts)
      int getNbTests();

      // Used for implementing the visitor pattern.
      // The visitor pattern is allowing to implement
      // different Runner for a tree of tests.
      virtual Testing::TestStatus accept(Runner* v) override
      {
         return(v->run(this));
      }

      // Check if, for benchmark, we want to run the code once
      // before benchmarking it, to force it to be in the I-cache.
      bool isForcedInCache();

      // Change the status of the forceInCache mode.
      void setForceInCache(bool);

    private:
        bool m_forcedInCache=false;
        // List of tests
        std::vector<test> m_tests;
        // List of tests IDs (since they are not contiguous
        // due to deprecation feature in python scripts)
        std::vector<Testing::testID_t> m_testIds;
  };

 
 
  // A group
  // It is possible to add subgroups to a group
  // and get a subgroup from its ID.
  class Group:public TestContainer
  {
    public:
      Group(Testing::testID_t);
  
      // Add a group or suite to this group.
      void addContainer(TestContainer*);

      // Get a container from its index. (index is not the node ID)
      TestContainer *getContainer(Testing::testIndex_t);

      // Get number of containers
      int getNbContainer();

      virtual Testing::TestStatus accept(Runner* v) override
      {
         return(v->run(this));
      }
     
    public:
        std::vector<TestContainer*> m_groups;

  };


    
}


#endif
