#include "InterpolationTestsQ7.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 20

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q7 ((q7_t)2)



    void InterpolationTestsQ7::test_linear_interp_q7()
    {
       const q31_t *inp = input.ptr();
       q7_t *outp = output.ptr();

       int nb;
       for(nb = 0; nb < input.nbSamples(); nb++)
       {
          outp[nb] = arm_linear_interp_q7(y.ptr(),inp[nb],y.nbSamples());
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);

    } 

 
    void InterpolationTestsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case InterpolationTestsQ7::TEST_LINEAR_INTERP_Q7_1:
          input.reload(InterpolationTestsQ7::INPUT_Q31_ID,mgr,nb);
          y.reload(InterpolationTestsQ7::YVAL_Q7_ID,mgr,nb);
          ref.reload(InterpolationTestsQ7::REF_LINEAR_Q7_ID,mgr,nb);

          break;

       }
      


       output.create(ref.nbSamples(),InterpolationTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
    }

    void InterpolationTestsQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
