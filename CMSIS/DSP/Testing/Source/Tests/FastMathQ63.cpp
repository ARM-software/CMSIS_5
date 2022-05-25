#include "FastMathQ63.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#include "arm_common_tables.h"
#include "dsp/utils.h"

#define SNR_THRESHOLD 100
/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR ((q63_t)0)

    void FastMathQ63::test_norm_64_to_32u()
    {
        
        const uint64_t *inp  = inputU64.ptr();
        int32_t *outValp  = outputVals.ptr();
        int16_t *outNormp  = outputNorms.ptr();
        unsigned long i;

        for(i=0; i < refVal.nbSamples(); i++)
        {
          int32_t val;
          int32_t norm;
          
          arm_norm_64_to_32u(inp[i],&val,&norm);
          outValp[i]=val;
          outNormp[i]=norm;
        }

        ASSERT_EQ(refVal,outputVals);
        ASSERT_EQ(refNorm,outputNorms);

    }

    void FastMathQ63::test_div_int64_to_int32()
    {
        const int64_t *denp  = inputS64.ptr();
        const int32_t *nump  = inputS32.ptr();

        int32_t *outValp  = outputVals.ptr();
        unsigned long i;

        for(i=0; i < refVal.nbSamples(); i++)
        {
          int32_t val;
          
          val = arm_div_int64_to_int32(denp[i],nump[i]);
          outValp[i]=val;

        }

        ASSERT_EQ(refVal,outputVals);
    }

  
    void FastMathQ63::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        (void)paramsArgs;
        switch(id)
        {
            case FastMathQ63::TEST_NORM_64_TO_32U_1:
            {
               inputU64.reload(FastMathQ63::NORMINPUT1_U64_ID,mgr);
               
               refVal.reload(FastMathQ63::NORM_REF_VALS_S32_ID,mgr);
               refNorm.reload(FastMathQ63::NORM_REF_S16_ID,mgr);
               
               outputVals.create(refVal.nbSamples(),FastMathQ63::OUT_S32_ID,mgr);
               outputNorms.create(refNorm.nbSamples(),FastMathQ63::NORMS_S16_ID,mgr);

            }
            break;

            case FastMathQ63::TEST_DIV_INT64_TO_INT32_2:
            {
               inputS64.reload(FastMathQ63::DIV_DEN_INPUT1_S64_ID,mgr);
               inputS32.reload(FastMathQ63::DIV_NUM_INPUT1_S32_ID,mgr);

               refVal.reload(FastMathQ63::DIV_REF_S32_ID,mgr);
               
               outputVals.create(refVal.nbSamples(),FastMathQ63::OUT_S32_ID,mgr);

            }
            break;

        }
        
    }

    void FastMathQ63::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
      //output.dump(mgr);
      
    }
