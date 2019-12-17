#include "ExampleCategoryQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 70

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)2)
#define ABS_ERROR_Q63 ((q63_t)(1<<16))



    void ExampleCategoryQ15::test_op_q15()
    {
        const q15_t *inp1=input1.ptr();
        const q15_t *inp2=input2.ptr();
        q15_t *refp=ref.ptr();
        q15_t *outp=output.ptr();

        arm_add_q15(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 


 
    void ExampleCategoryQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case ExampleCategoryQ15::TEST_OP_Q15_1:
             ref.reload(ExampleCategoryQ15::REF_OUT_Q15_ID,mgr);
          break;

 

       }
      

       input1.reload(ExampleCategoryQ15::INPUT1_Q15_ID,mgr,nb);
       input2.reload(ExampleCategoryQ15::INPUT2_Q15_ID,mgr,nb);

       output.create(ref.nbSamples(),ExampleCategoryQ15::OUT_Q15_ID,mgr);
    }

    void ExampleCategoryQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
