/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        Semihosting.cpp
 * Description:  Semihosting io
 *
 *               IO for a platform supporting semihosting.
 *               (Several input and output files)
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "Generators.h"
#include "Semihosting.h"


namespace Client
{
  
      struct pathOrGen {
        int kind;
        std::string path;
        Testing::param_t *data;
        Testing::nbSamples_t nbInputSamples;
        Testing::nbSamples_t nbOutputSamples;
        int dimensions;
      };

      Semihosting::Semihosting(std::string path,std::string patternRootPath,std::string outputRootPath,std::string parameterRootPath)
      {
        // Open the driver file
        this->infile=fopen(path.c_str(), "r");
        this->path=new std::vector<std::string>();
        this->patternRootPath=patternRootPath;
        this->outputRootPath=outputRootPath;
        this->parameterRootPath=parameterRootPath;
        this->patternFilenames=new std::vector<std::string>();
        this->outputNames=new std::vector<std::string>();
        this->parameterNames=new std::vector<struct pathOrGen>();
        this->m_hasParam = false;
      }

      void Semihosting::DeleteParams()
      {
        for (std::vector<struct pathOrGen>::iterator it = this->parameterNames->begin() ; it != this->parameterNames->end(); ++it)
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

      Semihosting::~Semihosting()
      {
         fclose(this->infile);
         delete(this->path);
         delete(this->patternFilenames);
         delete(this->outputNames);
         this->DeleteParams();
         delete(this->parameterNames);
      }

      /**
         Read the list of patterns from the driver file.

         This list is for the current suite.
    
      */
      void Semihosting::ReadPatternList()
      {
        char tmp[256];
        int nbPatterns;
        fscanf(this->infile,"%d\n",&nbPatterns);
        // Reset the list for the current suite
        this->patternFilenames->clear();
        std::string tmpstr;

        for(int i=0;i<nbPatterns;i++)
        {
           fgets(tmp,256,this->infile);
           // Remove end of line
            if (tmp[strlen(tmp)-1] == '\n')
            {
              tmp[strlen(tmp)-1]=0;
            }
           tmpstr.assign(tmp);
           this->patternFilenames->push_back(tmpstr);
        }
      }

      /**
         Read the list of parameters from the driver file.

         This list is for the current suite.
    
      */
      void Semihosting::ReadParameterList(Testing::nbParameters_t nbParams)
      {
        char tmp[256];
        char paramKind;
        std::string tmpstr;

        // It is the number of samples in the file.
        // Not the number of parameters controlling the function
        int nbValues;
        fscanf(this->infile,"%d\n",&nbValues);
        
        // Reset the list for the current suite
        this->DeleteParams();
        this->parameterNames->clear();

        for(int i=0;i<nbValues;i++)
        {
           fscanf(this->infile,"%c\n",&paramKind);
           struct pathOrGen gen;

           if (paramKind == 'p')
           {
              fgets(tmp,256,this->infile);
              // Remove end of line
              if (tmp[strlen(tmp)-1] == '\n')
              {
                 tmp[strlen(tmp)-1]=0;
              }
              tmpstr.assign(tmp);

              std::string tmp;
              tmp += this->parameterRootPath;
              tmp += this->testDir;
              tmp += "/";
              tmp += tmpstr;
        

              gen.kind=0;
              gen.path=tmp;

              gen.nbInputSamples = this->GetFileSize(tmp);
              gen.dimensions = nbParams;

           }
           // Generator
           // Generator kind (only 1 = cartesian product generator)
           // Number of samples generated when run
           // Number of dimensions
           // For each dimension
           //   Length
           //   Samples
           else
           {
              int kind,nbInputSamples,nbOutputSamples,dimensions,sample;
              Testing::param_t *p,*current;
             
              // Generator kind. Not yet used since there is only one kind of generator
              fscanf(this->infile,"%d\n",&kind);
              // Input data in config file
              fscanf(this->infile,"%d\n",&nbInputSamples);
              
              // Number of output combinations
              // And each output has dimensions parameters
              fscanf(this->infile,"%d\n",&nbOutputSamples);
              fscanf(this->infile,"%d\n",&dimensions);

              p=(Testing::param_t*)malloc(sizeof(Testing::param_t)*(nbInputSamples));
              current=p;
              for(int i=0;i < nbInputSamples; i ++)
              {
                
                fscanf(this->infile,"%d\n",&sample);
                *current++ = (Testing::param_t)sample;
              }

              gen.kind=1;
              gen.data=p;
              gen.nbInputSamples = nbInputSamples;
              gen.nbOutputSamples = nbOutputSamples;
              gen.dimensions = dimensions;
           }
           this->parameterNames->push_back(gen);

        }
      }

      /**
         Read the list of output from the driver file.

         This list is for the current suite.
    
      */
      void Semihosting::ReadOutputList()
      {
        char tmp[256];
        int nbOutputs;
        fscanf(this->infile,"%d\n",&nbOutputs);
        // Reset the list for the current suite
        this->outputNames->clear();
        std::string tmpstr;

        for(int i=0;i<nbOutputs;i++)
        {
           fgets(tmp,256,this->infile);
           // Remove end of line
            if (tmp[strlen(tmp)-1] == '\n')
            {
              tmp[strlen(tmp)-1]=0;
            }
           tmpstr.assign(tmp);
           this->outputNames->push_back(tmpstr);
        }
      }


      /** Read the number of parameters for all the tests in a suite

          Used for benchmarking. Same functions executed with
          different initializations controlled by the parameters.

          It is not the number of parameters in a file
          but the number of arguments (parameters) to control a function.

      */
      Testing::nbParameters_t Semihosting::ReadNbParameters()
      {
         unsigned long nb;
         fscanf(this->infile,"%ld\n",&nb);

         return(nb);
      }

      void Semihosting::recomputeTestDir()
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

      
      void Semihosting::ReadTestIdentification()
      {
    
        char tmp[255];
        int kind;
        Testing::testID_t theId;
        char hasPath;
        char hasParamID;
        Testing::PatternID_t paramID;


        fscanf(this->infile,"%d %ld\n",&kind,&theId);

        fscanf(this->infile,"%c\n",&hasParamID);
        this->m_hasParam=false;
        if (hasParamID == 'y')
        {
           this->m_hasParam=true;
           fscanf(this->infile,"%ld\n",&paramID);
           this->currentParam=paramID;
        }

        fscanf(this->infile,"%c\n",&hasPath);
        if (hasPath == 'y')
        {
            fgets(tmp,256,this->infile);
            // Remove end of line
            if (tmp[strlen(tmp)-1] == '\n')
            {
              tmp[strlen(tmp)-1]=0;
            }
            currentPath.assign(tmp);
        }

        this->currentKind=kind;
        this->currentId=theId;
        switch(kind)
        {
          case 1:
             printf("t \n");
             break;
          case 2:
             printf("s %ld\n",this->currentId);
             break;
          case 3:
             printf("g %ld\n",this->currentId);
             break;
          default:
             printf("u\n");
        }

      }

      Testing::testID_t Semihosting::CurrentTestID()
      {
         return(this->currentId);
      }

      /**
         Read identification of a group or suite.

         The difference with a test node is that the current folder
         can be changed by a group or suite.
    
      */
      void Semihosting::ReadIdentification()
      {
        this->ReadTestIdentification();
        this->path->push_back(currentPath);
        this->recomputeTestDir();
      }

      /**
         Dump the test status into the output (stdout)

      */
      void Semihosting::DispStatus(Testing::TestStatus status
        ,Testing::errorID_t error
        ,unsigned long lineNb
        ,Testing::cycles_t cycles)
      {
        if (status == Testing::kTestFailed)
        {
            printf("%ld %ld %ld 0 N\n",this->currentId,error,lineNb);
        }
        else
        {
#ifdef EXTBENCH
            printf("%ld 0 0 t Y\n",this->currentId);
#else
            printf("%ld 0 0 %u Y\n",this->currentId,cycles);
#endif
        }
      }

      void Semihosting::DispErrorDetails(const char* details)
      {
          printf("E: %s\n",details);
      }

      /**
           Signal end of group

           (Used by scripts parsing the output to display the results)
      */
      void Semihosting::EndGroup()
      {
        printf("p\n");
        this->path->pop_back();
      }

      /**
           Get pattern path.


      */
      std::string Semihosting::getPatternPath(Testing::PatternID_t id)
      {
        std::string tmp;
        tmp += this->patternRootPath;
        tmp += this->testDir;
        tmp += "/";
        tmp += (*this->patternFilenames)[id];
        
        return(tmp); 
      }

      /**
           Get parameter path.


      */
      struct pathOrGen Semihosting::getParameterDesc(Testing::PatternID_t id)
      {

        

        return((*this->parameterNames)[id]);
        
      }

      /**
           Get output path.

           The test ID (currentId) is used in the name
      */
      std::string Semihosting::getOutputPath(Testing::outputID_t id)
      {
        char fmt[256];

        std::string tmp;
        tmp += this->outputRootPath;
        tmp += this->testDir;
        sprintf(fmt,"/%s_%ld.txt",(*this->outputNames)[id].c_str(),this->currentId);
        tmp += std::string(fmt);
        //printf("%s\n",tmp.c_str());
        
        return(tmp); 
      }

      Testing::nbSamples_t Semihosting::GetPatternSize(Testing::PatternID_t id)
      {
           char tmp[256];
           Testing::nbSamples_t len;
           std::string fileName = this->getPatternPath(id);

           FILE *pattern=fopen(fileName.c_str(), "r");

           if (pattern==NULL)
           {
             return(0);
           }

           // Ignore word size format
           fgets(tmp,256,pattern);
        
           // Get nb of samples
           fgets(tmp,256,pattern);

           fclose(pattern);


           len=atoi(tmp);
           return(len);

      }

      Testing::nbSamples_t Semihosting::GetFileSize(std::string &filepath)
      {
           char tmp[256];
           Testing::nbSamples_t len;
           
  
           FILE *params=fopen(filepath.c_str(), "r");
           if (params==NULL)
           {
             return(0);
           }
           // Get nb of samples
           fgets(tmp,256,params);
           fclose(params);

           len=atoi(tmp);
           return(len);

      }

      void Semihosting::DumpParams(std::vector<Testing::param_t>& params)
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

      Testing::param_t* Semihosting::ImportParams(Testing::PatternID_t id,Testing::nbParameterEntries_t &nbEntries,Testing::ParameterKind &paramKind)
      {
          nbEntries = 0;

          char tmp[256];
          
          Testing::param_t *p;
          uint32_t val;

          Testing::nbSamples_t len;
          struct pathOrGen gen = this->getParameterDesc(id);

          if (gen.kind == 0)
          {
             char *result=NULL;
             paramKind=Testing::kDynamicBuffer;
             FILE *params=fopen(gen.path.c_str(), "r");
             
             if (params==NULL)
             {
                return(NULL);
             }
             // Get nb of samples
             fgets(tmp,256,params);
             
   
             len=gen.nbInputSamples;
             result=(char*)malloc(len*sizeof(Testing::param_t));
             p = (Testing::param_t*)result;
             nbEntries = len / gen.dimensions;
   
             for(uint32_t i=0; i < len; i++)
             {
               fscanf(params,"%d\n",&val);
               *p++ = val;
             }
   
   
             fclose(params);
             return((Testing::param_t*)result);
          }
          else
          {

             Testing::param_t* result;
             paramKind=Testing::kDynamicBuffer;
             // Output samples is number of parameter line
             len=gen.nbOutputSamples * gen.dimensions;

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

      bool Semihosting::hasParam()
      {
         return(this->m_hasParam);
      }

      Testing::PatternID_t Semihosting::getParamID()
      {
         return(this->currentParam);
      } 

      void Semihosting::ImportPattern_f64(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint64_t val;
          float64_t *ptr=(float64_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          // word size format is used when generating include files with python scripts
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
             for(i=0;i<len;i++)
             {
               // Ignore comment
                fgets(tmp,256,pattern);
                fscanf(pattern,"0x%16llx\n",&val);
                *ptr = TOTYP(float64_t,val);
                ptr++;
             }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_f32(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          float32_t *ptr=(float32_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          //printf(":::: %s\n",fileName.c_str());

          if (ptr)
          {
             for(i=0;i<len;i++)
             {
                // Ignore comment
                fgets(tmp,256,pattern);
                fscanf(pattern,"0x%08X\n",&val);
                //printf(":::: %08X %f\n",val, TOTYP(float32_t,val));
                *ptr = TOTYP(float32_t,val);
                ptr++;
             }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_q63(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint64_t val;
          q63_t *ptr=(q63_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
             for(i=0;i<len;i++)
             {
               // Ignore comment
                fgets(tmp,256,pattern);
                fscanf(pattern,"0x%016llX\n",&val);
                *ptr = TOTYP(q63_t,val);
                ptr++;
             }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_q31(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          q31_t *ptr=(q31_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
             for(i=0;i<len;i++)
             {
               // Ignore comment
                fgets(tmp,256,pattern);
                fscanf(pattern,"0x%08X\n",&val);
                *ptr = TOTYP(q31_t,val);
                ptr++;
             }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_q15(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          q15_t *ptr=(q15_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
             for(i=0;i<len;i++)
             {
               // Ignore comment
                fgets(tmp,256,pattern);
                fscanf(pattern,"0x%08X\n",&val);
                *ptr = TOTYP(q15_t,val);
                ptr++;
             }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_q7(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          q7_t *ptr=(q7_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
              for(i=0;i<len;i++)
              {
                // Ignore comment
                 fgets(tmp,256,pattern);
                 fscanf(pattern,"0x%08X\n",&val);
                 *ptr = TOTYP(q7_t,val);
                 ptr++;
              }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_u32(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          uint32_t *ptr=(uint32_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
              for(i=0;i<len;i++)
              {
                // Ignore comment
                 fgets(tmp,256,pattern);
                 fscanf(pattern,"0x%08X\n",&val);
                 *ptr = val;
                 ptr++;
              }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_u16(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          uint16_t *ptr=(uint16_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }

          if (ptr)
          {
              for(i=0;i<len;i++)
              {
                // Ignore comment
                 fgets(tmp,256,pattern);
                 fscanf(pattern,"0x%08X\n",&val);
                 *ptr = (uint16_t)val;
                 ptr++;
              }
          }

          fclose(pattern);
          
      }

      void Semihosting::ImportPattern_u8(Testing::PatternID_t id,char* p,Testing::nbSamples_t nb)
      {
          char tmp[256];
          Testing::nbSamples_t len;
          Testing::nbSamples_t i=0;

          uint32_t val;
          uint8_t *ptr=(uint8_t*)p;

          std::string fileName = this->getPatternPath(id);
          FILE *pattern=fopen(fileName.c_str(), "r");
          // Ignore word size format
          fgets(tmp,256,pattern);
          // Get nb of samples
          fgets(tmp,256,pattern);
          len=atoi(tmp);

          if ((nb != MAX_NB_SAMPLES) && (nb < len))
          {
             len = nb;
          }
          
          if (ptr)
          {
              for(i=0;i<len;i++)
              {
                // Ignore comment
                 fgets(tmp,256,pattern);
                 fscanf(pattern,"0x%08X\n",&val);
                 *ptr = (uint8_t)val;
                 ptr++;
              }
          }

          fclose(pattern);
          
      }

      void Semihosting::DumpPattern_f64(Testing::outputID_t id,Testing::nbSamples_t nb, float64_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint64_t t;
                float64_t v;
                for(i=0; i < nb; i++)
                {
                   v = data[i];
                   t = TOINT64(v);
                   fprintf(f,"0x%016llx\n",t);
                }
                fclose(f);
            }

      }
      void Semihosting::DumpPattern_f32(Testing::outputID_t id,Testing::nbSamples_t nb, float32_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
               FILE *f = fopen(fileName.c_str(),"w");
               Testing::nbSamples_t i=0;
               uint32_t t;
               float32_t v;
               for(i=0; i < nb; i++)
               {
                  v = data[i];
                  t = TOINT32(v);
                  fprintf(f,"0x%08x\n",t);
               }
               fclose(f);
            }
      }

      void Semihosting::DumpPattern_q63(Testing::outputID_t id,Testing::nbSamples_t nb, q63_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint64_t t;
                for(i=0; i < nb; i++)
                {
                   t = (uint64_t)data[i];
                   fprintf(f,"0x%016llx\n",t);
                }
                fclose(f);
            }
      }

      void Semihosting::DumpPattern_q31(Testing::outputID_t id,Testing::nbSamples_t nb, q31_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint32_t t;
                for(i=0; i < nb; i++)
                {
                   t = (uint32_t)data[i];
                   fprintf(f,"0x%08x\n",t);
                }
                fclose(f);
            }
      }
      void Semihosting::DumpPattern_q15(Testing::outputID_t id,Testing::nbSamples_t nb, q15_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint32_t t;
                for(i=0; i < nb; i++)
                {
                   t = (uint32_t)data[i];
                   fprintf(f,"0x%08x\n",t);
                }
                fclose(f);
            }
      }
      void Semihosting::DumpPattern_q7(Testing::outputID_t id,Testing::nbSamples_t nb, q7_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint32_t t;
                for(i=0; i < nb; i++)
                {
                   t = (uint32_t)data[i];
                   fprintf(f,"0x%08x\n",t);
                }
                fclose(f);
            }
      }
      void Semihosting::DumpPattern_u32(Testing::outputID_t id,Testing::nbSamples_t nb, uint32_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint32_t t;
                for(i=0; i < nb; i++)
                {
                   t = (uint32_t)data[i];
                   fprintf(f,"0x%08x\n",t);
                }
                fclose(f);
            }
      }
      void Semihosting::DumpPattern_u16(Testing::outputID_t id,Testing::nbSamples_t nb, uint16_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
                FILE *f = fopen(fileName.c_str(),"w");
                Testing::nbSamples_t i=0;
                uint32_t t;
                for(i=0; i < nb; i++)
                {
                   t = (uint32_t)data[i];
                   fprintf(f,"0x%08x\n",t);
                }
                fclose(f);
            }
      }
      void Semihosting::DumpPattern_u8(Testing::outputID_t id,Testing::nbSamples_t nb, uint8_t* data)
      {
            std::string fileName = this->getOutputPath(id);
            if (data)
            {
               FILE *f = fopen(fileName.c_str(),"w");
               Testing::nbSamples_t i=0;
               uint32_t t;
               for(i=0; i < nb; i++)
               {
                  t = (uint32_t)data[i];
                  fprintf(f,"0x%08x\n",t);
               }
               fclose(f);
            }
      }
      
    


}
