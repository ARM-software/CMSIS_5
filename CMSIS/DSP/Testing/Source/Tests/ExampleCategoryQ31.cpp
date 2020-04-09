#include "ExampleCategoryQ31.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 100

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)2)
#define ABS_ERROR_Q63 ((q63_t)(1<<16))




    void ExampleCategoryQ31::test_op_q31()
    {
        const q31_t *inp1=input1.ptr();
        const q31_t *inp2=input2.ptr();
        q31_t *refp=ref.ptr();
        q31_t *outp=output.ptr();

        arm_add_q31(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 


 
    void ExampleCategoryQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case ExampleCategoryQ31::TEST_OP_Q31_1:
             ref.reload(ExampleCategoryQ31::REF_OUT_Q31_ID,mgr);
          break;

 

       }
      

       input1.reload(ExampleCategoryQ31::INPUT1_Q31_ID,mgr,nb);
       input2.reload(ExampleCategoryQ31::INPUT2_Q31_ID,mgr,nb);

       output.create(ref.nbSamples(),ExampleCategoryQ31::OUT_Q31_ID,mgr);
    }

    void ExampleCategoryQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
