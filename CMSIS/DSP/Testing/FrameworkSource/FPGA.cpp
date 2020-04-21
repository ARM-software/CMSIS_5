/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        FPGA.cpp
 * Description:  FPGA
 *
 *               IO implementation for constrained platforms where
 *               inputs are contained in a header files and output is
 *               only stdout.
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
#include "FPGA.h"
#include <stdio.h>
#include <string.h>
#include "Generators.h"
#include "arm_math.h"
#include "arm_math_f16.h"

namespace Client
{
    struct offsetOrGen
    {
        int kind;
        unsigned long offset;
        Testing::param_t *data;
        Testing::nbSamples_t nbInputSamples;
        Testing::nbSamples_t nbOutputSamples;
        int dimensions;
    };

    FPGA::FPGA(const char *testDesc,const char *patterns)
    {
      this->m_testDesc=testDesc;
      this->m_patterns=patterns;

      this->currentDesc=testDesc;
      this->path=new std::vector<std::string>();

      this->patternOffsets=new std::vector<unsigned long>();
      this->patternSizes=new std::vector<unsigned long>();

      this->parameterOffsets=new std::vector<struct offsetOrGen>();
      this->parameterSizes=new std::vector<unsigned long>();

      this->outputNames=new std::vector<std::string>();

    }

    void FPGA::DeleteParams()
      {
        for (std::vector<struct offsetOrGen>::iterator it = this->parameterOffsets->begin() ; it != this->parameterOffsets->end(); ++it)
         {
           if (it->kind==1)
           {
             if (it->data)
             {
                free(it->data);
                it->data = NULL;
             }
           }
         }
      }

    FPGA::~FPGA()
    {
      delete(this->path);

      delete(this->patternOffsets);
      delete(this->patternSizes);
      this->DeleteParams();
      delete(this->parameterOffsets);
      delete(this->parameterSizes);

      delete(this->outputNames);
    }

    /** Read word 64 from C array

    */

    /** Read word 32 from C array

    */
    void FPGA::read32(unsigned long *r)
    {
       unsigned char a,b,c,d;
       unsigned long v;
       a = *this->currentDesc++;
       b = *this->currentDesc++;
       c = *this->currentDesc++;
       d = *this->currentDesc++;
       //printf("%d %d %d %d\n",a,b,c,d);

       v = a | (b << 8) | (c << 16) | (d << 24);
       *r = v;
    }

    /** Read null terminated C string C array

    */
    void FPGA::readStr(char *str)
    {
       char *p = str;
       while(*this->currentDesc != 0)
       {
        *p++ = *this->currentDesc++;
       }
       *p++ = 0;
       this->currentDesc++;
    }

    void FPGA::readChar(char *c)
    {
        *c = *this->currentDesc;
        this->currentDesc++;
    }

    /** Get output path from output ID

    */
    std::string FPGA::getOutputPath(Testing::outputID_t id)
    {
        char fmt[256];

        std::string tmp;
        tmp += this->testDir;
        sprintf(fmt,"/%s_%ld.txt",(*this->outputNames)[id].c_str(),this->currentId);
        tmp += std::string(fmt);
        //printf("%s\n",tmp.c_str());
        
        return(tmp); 
    }

    /** Read the number of parameters for all the tests in a suite

          Used for benchmarking. Same functions executed with
          different initializations controlled by the parameters.

    */
    Testing::nbParameters_t FPGA::ReadNbParameters()
    {
         unsigned long nb;
         this->read32(&nb);

         return(nb);
    }

    void FPGA::ReadTestIdentification()
    {
        char tmp[255];
        unsigned long kind;
        unsigned long theId;
        char hasPath;
        char hasParamID;
        Testing::PatternID_t paramID;
        //printf("Read ident\n");

        this->read32(&kind);
        this->read32(&theId);

        this->readChar(&hasParamID);
        this->m_hasParam=false;
        if (hasParamID == 'y')
        {
           this->m_hasParam=true;
           this->read32(&paramID);
           this->currentParam=paramID;
        }

        this->readChar(&hasPath);
        if (hasPath == 'y')
        {
            this->readStr(tmp);
            //printf("-->%s\n",tmp);
            currentPath.assign(tmp);
        }

        this->currentKind=kind;
        this->currentId=theId;
        switch(kind)
        {
          case 1:
             printf("S: t \n");
             break;
          case 2:
             printf("S: s %ld\n",this->currentId);
             break;
          case 3:
             printf("S: g %ld\n",this->currentId);
             break;
          default:
             printf("S: u\n");
        }

        
        //printf("End read ident\n\n");
    }

    void FPGA::recomputeTestDir()
      {
        this->testDir = ".";
        int start = 1;
        std::vector<std::string>::const_iterator iter;
        for (iter = this->path->begin(); iter != this->path->end(); ++iter)
        {
             if (start)
             {
                this->testDir = *iter;
                start =0;
             }
             else
             {
               if (!(*iter).empty())
               {
                  this->testDir += "/" + *iter;
               }
             }
        }
      }

    void FPGA::ReadIdentification()
    {
       this->ReadTestIdentification();
       this->path->push_back(currentPath);
       this->recomputeTestDir();
    }

    /** There is only stdout available for "FPGA".
        So status output and data output are interleaved.

        Status is starting with "S: "

    */
    void FPGA::DispStatus(Testing::TestStatus status
      ,Testing::errorID_t error
      ,unsigned long lineNb
      ,Testing::cycles_t cycles)
    {
        if (status == Testing::kTestFailed)
        {
            printf("S: %ld %ld %ld 0 N\n",this->currentId,error,lineNb);
        }
        else
        {
#ifdef EXTBENCH
            printf("S: %ld 0 0 t Y\n",this->currentId);
#else
            printf("S: %ld 0 0 %u Y\n",this->currentId, cycles);
#endif
        }
    }

    void FPGA::DispErrorDetails(const char* details)
    {
          printf("E: %s\n",details);
    }

    void FPGA::EndGroup()
    {
       printf("S: p\n");
       this->path->pop_back();
    }


    /** Read pattern list

        Different from semihosting.
        We read offset and sizes for the patterns
        rather than file names.

    */
    void FPGA::ReadPatternList()
    {

        unsigned long offset,nb;
        unsigned long nbPatterns;
        this->read32(&nbPatterns);
        this->patternOffsets->clear();
        this->patternSizes->clear();
        std::string tmpstr;

        for(int i=0;i<nbPatterns;i++)
        {
           this->read32(&offset);
           this->read32(&nb);
           this->patternOffsets->push_back(offset);
           this->patternSizes->push_back(nb);
        }

    }

    /** Read parameters list

        Different from semihosting.
        We read offset and sizes for the parameters
        rather than file names.

    */
    void FPGA::ReadParameterList(Testing::nbParameters_t nbParams)
    {

        unsigned long offset,nb;
        unsigned long nbValues;
        char paramKind;

        this->read32(&nbValues);

        this->DeleteParams();
        this->parameterOffsets->clear();
        this->parameterSizes->clear();
        std::string tmpstr;

        for(int i=0;i<nbValues;i++)
        {
           this->readChar(&paramKind);
           struct offsetOrGen gen;
           if (paramKind == 'p')
           {
             gen.kind=0;
             this->read32(&offset);
             this->read32(&nb);
             gen.offset=offset;

             gen.kind=0;
             gen.nbInputSamples=nb;
             gen.dimensions = nbParams;
           }
           else
           {
              unsigned long kind,nbInputSamples,nbOutputSamples,dimensions,sample;
              Testing::param_t *p,*current;

              // Generator kind
              this->read32(&kind);

              this->read32(&nbInputSamples);
              this->read32(&nbOutputSamples);
              this->read32(&dimensions);

              p=(Testing::param_t*)malloc(sizeof(Testing::param_t)*(nbInputSamples));
              current=p;
              for(int i=0;i < nbInputSamples; i ++)
              {
                
                this->read32(&sample);
                *current++ = (Testing::param_t)sample;
              }


              gen.kind=1;
              gen.data=p;
              gen.nbInputSamples = nbInputSamples;
              gen.nbOutputSamples = nbOutputSamples;
              gen.dimensions = dimensions;
              
           }
           this->parameterOffsets->push_back(gen);
           this->parameterSizes->push_back(nb);
        }

    }

    void FPGA::ReadOutputList()
      {
        char tmp[256];
        unsigned long nbOutputs;
        this->read32(&nbOutputs);
        this->outputNames->clear();
        std::string tmpstr;

        for(int i=0;i<nbOutputs;i++)
        {
           this->readStr(tmp);
           tmpstr.assign(tmp);
           this->outputNames->push_back(tmpstr);
        }
      }

    unsigned long FPGA::getPatternOffset(Testing::PatternID_t id)
    {
        return((*this->patternOffsets)[id]);
    }

    Testing::nbSamples_t FPGA::GetPatternSize(Testing::PatternID_t id)
    {
      return((Testing::nbSamples_t)((*this->patternSizes)[id]));
    }

    unsigned long FPGA::getParameterOffset(Testing::PatternID_t id)
    {
        return((*this->parameterOffsets)[id].offset);
    }

    struct offsetOrGen FPGA::getParameterDesc(Testing::PatternID_t id)
    {

        return((*this->parameterOffsets)[id]);
        
    }


    void FPGA::DumpParams(std::vector<Testing::param_t>& params)
      {
           bool begin=true;
           printf("b ");
           for(std::vector<Testing::param_t>::iterator it = params.begin(); it != params.end(); ++it) 
           {
              if (!begin)
              {
                printf(",");
              }
              printf("%d",*it);
              begin=false;
           }
           printf("\n");
      }

    Testing::param_t* FPGA::ImportParams(Testing::PatternID_t id,Testing::nbParameterEntries_t &nbEntries,Testing::ParameterKind &paramKind)
    {
        nbEntries=0;
        unsigned long offset;


        Testing::nbSamples_t len;
        struct offsetOrGen gen = this->getParameterDesc(id);

        if (gen.kind == 0)
        {
           offset=gen.offset;
           paramKind=Testing::kStaticBuffer;

           nbEntries = gen.nbInputSamples / gen.dimensions;
   
           const char *patternStart = this->m_patterns + offset;
   
           return((Testing::param_t*)patternStart);
        }
        else
        {
          Testing::param_t* result;
          // Output samples is number of parameter line
          len=gen.nbOutputSamples * gen.dimensions;
          paramKind=Testing::kDynamicBuffer;

          result=(Testing::param_t*)malloc(len*sizeof(Testing::param_t));

          switch(gen.dimensions)
          {
            case 1:
              generate1(result,gen.data,nbEntries);
            break;
            case 2:
              generate2(result,gen.data,nbEntries);
            break;
            case 3:
              generate3(result,gen.data,nbEntries);
            break;
            case 4:
              generate4(result,gen.data,nbEntries);
            break;
            default:
              generate1(result,gen.data,nbEntries);
            break;
          }
 
          return(result);
        }
    }

    bool FPGA::hasParam()
    {
         return(this->m_hasParam);
    }

    Testing::PatternID_t FPGA::getParamID()
    {
         return(this->currentParam);
    } 



    void FPGA::ImportPattern_f64(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const float64_t *src = (const float64_t*)patternStart;
        float64_t *dst = (float64_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }
    }

    void FPGA::ImportPattern_f32(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const float32_t *src = (const float32_t*)patternStart;
        float32_t *dst = (float32_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }

    }

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
    void FPGA::ImportPattern_f16(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const float16_t *src = (const float16_t*)patternStart;
        float16_t *dst = (float16_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }

    }
#endif

    void FPGA::ImportPattern_q63(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const q63_t *src = (const q63_t*)patternStart;
        q63_t *dst = (q63_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }
    }

    void FPGA::ImportPattern_q31(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const q31_t *src = (const q31_t*)patternStart;
        q31_t *dst = (q31_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }
    }

    void FPGA::ImportPattern_q15(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const q15_t *src = (const q15_t*)patternStart;
        q15_t *dst = (q15_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }
    }

    void FPGA::ImportPattern_q7(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const q7_t *src = (const q7_t*)patternStart;
        q7_t *dst = (q7_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }
    }

    void FPGA::ImportPattern_u32(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const uint32_t *src = (const uint32_t*)patternStart;
        uint32_t *dst = (uint32_t*)p;

        if (dst)
        {
            for(i=0; i < nb; i++)
            {
                *dst++ = *src++;
            }
        }
    }

    void FPGA::ImportPattern_u16(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const uint16_t *src = (const uint16_t*)patternStart;
        uint16_t *dst = (uint16_t*)p;

        if (dst)
        {
           for(i=0; i < nb; i++)
           {
               *dst++ = *src++;
           }
        }
    }

    void FPGA::ImportPattern_u8(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
    {
        unsigned long offset,i;

        offset=this->getPatternOffset(id);

        const char *patternStart = this->m_patterns + offset;
        const uint8_t *src = (const uint8_t*)patternStart;
        uint8_t *dst = (uint8_t*)p;

        if (dst)
        {
            for(i=0; i < nb; i++)
            {
                *dst++ = *src++;
            }
        }
    }
    
    /** Dump patterns.

        There is only stdout available for "FPGA".
        So status output and data output are interleaved.

        Data is starting with "D: "

    */
    void FPGA::DumpPattern_f64(Testing::outputID_t id,Testing::nbSamples_t nb, float64_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
           printf("D: %s\n",fileName.c_str());
           Testing::nbSamples_t i=0;
           uint64_t t;
           float64_t v;
           for(i=0; i < nb; i++)
           {
              v = data[i];
              t = TOINT64(v);
              printf("D: 0x%016llx\n",t);
           }
           printf("D: END\n");
        }

    }
    void FPGA::DumpPattern_f32(Testing::outputID_t id,Testing::nbSamples_t nb, float32_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
            printf("D: %s\n",fileName.c_str());
            Testing::nbSamples_t i=0;
            uint32_t t;
            float32_t v;
            for(i=0; i < nb; i++)
            {
               v = data[i];
               t = TOINT32(v);
               printf("D: 0x%08x\n",t);
            }
            printf("D: END\n");
        }
    }

#if !defined( __CC_ARM ) && defined(ARM_FLOAT16_SUPPORTED)
    void FPGA::DumpPattern_f16(Testing::outputID_t id,Testing::nbSamples_t nb, float16_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
            printf("D: %s\n",fileName.c_str());
            Testing::nbSamples_t i=0;
            uint16_t t;
            float16_t v;
            for(i=0; i < nb; i++)
            {
               v = data[i];
               t = TOINT16(v);
               printf("D: 0x0000%04x\n",t);
            }
            printf("D: END\n");
        }
    }
#endif
    
    void FPGA::DumpPattern_q63(Testing::outputID_t id,Testing::nbSamples_t nb, q63_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
           printf("D: %s\n",fileName.c_str());
           Testing::nbSamples_t i=0;
           uint64_t t;
           q63_t v;
           for(i=0; i < nb; i++)
           {
              v = data[i];
              t = (uint64_t)v;
              printf("D: 0x%016llx\n",t);
           }
           printf("D: END\n");
        }
    }

    void FPGA::DumpPattern_q31(Testing::outputID_t id,Testing::nbSamples_t nb, q31_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
           printf("D: %s\n",fileName.c_str());
           Testing::nbSamples_t i=0;
           uint32_t t;
           q31_t v;
           for(i=0; i < nb; i++)
           {
              v = data[i];
              t = (uint32_t)v;
              printf("D: 0x%08x\n",t);
           }
           printf("D: END\n");
        }
    }

    void FPGA::DumpPattern_q15(Testing::outputID_t id,Testing::nbSamples_t nb, q15_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
           printf("D: %s\n",fileName.c_str());
           Testing::nbSamples_t i=0;
           uint32_t t;
           q15_t v;
           for(i=0; i < nb; i++)
           {
              v = data[i];
              t = (uint32_t)v;
              printf("D: 0x%08x\n",t);
           }
           printf("D: END\n");
        }
    }

    void FPGA::DumpPattern_q7(Testing::outputID_t id,Testing::nbSamples_t nb, q7_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
            printf("D: %s\n",fileName.c_str());
            Testing::nbSamples_t i=0;
            uint32_t t;
            q7_t v;
            for(i=0; i < nb; i++)
            {
               v = data[i];
               t = (uint32_t)v;
               printf("D: 0x%08x\n",t);
            }
            printf("D: END\n");
        }
    }

    void FPGA::DumpPattern_u32(Testing::outputID_t id,Testing::nbSamples_t nb, uint32_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
           printf("D: %s\n",fileName.c_str());
           Testing::nbSamples_t i=0;
           uint32_t t;
           uint32_t v;
           for(i=0; i < nb; i++)
           {
              v = data[i];
              t = (uint32_t)v;
              printf("D: 0x%08x\n",t);
           }
           printf("D: END\n");
        }
    }

    void FPGA::DumpPattern_u16(Testing::outputID_t id,Testing::nbSamples_t nb, uint16_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
            printf("D: %s\n",fileName.c_str());
            Testing::nbSamples_t i=0;
            uint32_t t;
            uint16_t v;
            for(i=0; i < nb; i++)
            {
               v = data[i];
               t = (uint32_t)v;
               printf("D: 0x%08x\n",t);
            }
            printf("D: END\n");
        }
    }

    void FPGA::DumpPattern_u8(Testing::outputID_t id,Testing::nbSamples_t nb, uint8_t* data)
    {
        std::string fileName = this->getOutputPath(id); 
        if (data)
        {
            printf("D: %s\n",fileName.c_str());
            Testing::nbSamples_t i=0;
            uint32_t t;
            uint8_t v;
            for(i=0; i < nb; i++)
            {
               v = data[i];
               t = (uint32_t)v;
               printf("D: 0x%08x\n",t);
            }
            printf("D: END\n");
        }
    }

    Testing::testID_t FPGA::CurrentTestID()
    {
        return(this->currentId);
    }
    
}

