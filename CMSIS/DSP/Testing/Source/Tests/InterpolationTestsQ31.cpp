#include "InterpolationTestsQ31.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 100

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)5)



    void InterpolationTestsQ31::test_linear_interp_q31()
    {
       const q31_t *inp = input.ptr();
       q31_t *outp = output.ptr();

       int nb;
       for(nb = 0; nb < input.nbSamples(); nb++)
       {
          outp[nb] = arm_linear_interp_q31(y.ptr(),inp[nb],y.nbSamples());
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

 
    void InterpolationTestsQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case InterpolationTestsQ31::TEST_LINEAR_INTERP_Q31_1:
          input.reload(InterpolationTestsQ31::INPUT_Q31_ID,mgr,nb);
          y.reload(InterpolationTestsQ31::YVAL_Q31_ID,mgr,nb);
          ref.reload(InterpolationTestsQ31::REF_LINEAR_Q31_ID,mgr,nb);

          break;

       }
      


       output.create(ref.nbSamples(),InterpolationTestsQ31::OUT_SAMPLES_Q31_ID,mgr);
    }

    void InterpolationTestsQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
