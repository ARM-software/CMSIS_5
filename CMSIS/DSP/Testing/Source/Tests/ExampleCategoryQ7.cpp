#include "ExampleCategoryQ7.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 20

#define ABS_ERROR_Q7 ((q7_t)2)
#define ABS_ERROR_Q31 ((q31_t)(1<<15))



    void ExampleCategoryQ7::test_op_q7()
    {
        const q7_t *inp1=input1.ptr();
        const q7_t *inp2=input2.ptr();
        q7_t *outp=output.ptr();

        arm_add_q7(inp1,inp2,outp,input1.nbSamples());

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(q7_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);

    } 


 
    void ExampleCategoryQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;
       
       switch(id)
       {
        case ExampleCategoryQ7::TEST_OP_Q7_1:
             ref.reload(ExampleCategoryQ7::REF_OUT_Q7_ID,mgr);
          break;

 

       }
      

       input1.reload(ExampleCategoryQ7::INPUT1_Q7_ID,mgr,nb);
       input2.reload(ExampleCategoryQ7::INPUT2_Q7_ID,mgr,nb);

       output.create(ref.nbSamples(),ExampleCategoryQ7::OUT_Q7_ID,mgr);
    }

    void ExampleCategoryQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
