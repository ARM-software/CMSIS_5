#include "InterpolationTestsQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 70

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)2)



    void InterpolationTestsQ15::test_linear_interp_q15()
    {
       const q31_t *inp = input.ptr();
       q15_t *outp = output.ptr();

       int nb;
       for(nb = 0; nb < input.nbSamples(); nb++)
       {
          outp[nb] = arm_linear_interp_q15(y.ptr(),inp[nb],y.nbSamples());
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

 
    void InterpolationTestsQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case InterpolationTestsQ15::TEST_LINEAR_INTERP_Q15_1:
          input.reload(InterpolationTestsQ15::INPUT_Q31_ID,mgr,nb);
          y.reload(InterpolationTestsQ15::YVAL_Q15_ID,mgr,nb);
          ref.reload(InterpolationTestsQ15::REF_LINEAR_Q15_ID,mgr,nb);

          break;

       }
      


       output.create(ref.nbSamples(),InterpolationTestsQ15::OUT_SAMPLES_Q15_ID,mgr);
    }

    void InterpolationTestsQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
