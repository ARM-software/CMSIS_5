/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Test.cpp
 * Description:  Generic test framework code
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
#include <cstdio>

int testIndex(Testing::testIndex_t i)
{
    return(i-1);
}

namespace Client
{

  TestContainer::TestContainer(Testing::testID_t id):m_containerID(id)
  {

  }
/* Client */

  Suite::Suite(Testing::testID_t id):
     TestContainer(id),
     m_tests(std::vector<test>()),
     m_testIds(std::vector<Testing::testID_t>())
  {

  }

  void Suite::addTest(Testing::testID_t id,test aTest)
  {
    m_tests.push_back(aTest);
    m_testIds.push_back(id);
  }

  test Suite::getTest(Testing::testIndex_t id)
  {
     return(m_tests[testIndex(id)]);
  }

  int Suite::getNbTests()
  {
    return(m_tests.size());
  }

  bool Suite::isForcedInCache()
  {
      return(m_forcedInCache);
  }
      
  void Suite::setForceInCache(bool status)
  {
      m_forcedInCache = status;
  }
 


  Group::Group(Testing::testID_t id):
     TestContainer(id),
     m_groups(std::vector<TestContainer*>())
  {

  }
  
  void Group::addContainer(TestContainer *s)
  {
      m_groups.push_back(s);
  }

  TestContainer *Group::getContainer(Testing::testIndex_t id)
  {
     return(m_groups[testIndex(id)]);
  }

  int Group::getNbContainer()
  {
    return(m_groups.size());
  }
     


 
}

