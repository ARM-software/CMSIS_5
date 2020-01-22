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

    void InterpolationTestsQ7::test_bilinear_interp_q7()
    {
       const q31_t *inp = input.ptr();
       q7_t *outp = output.ptr();
       q31_t x,y;
       int nb;
       for(nb = 0; nb < input.nbSamples(); nb += 2)
       {
          x = inp[nb];
          y = inp[nb+1];
          *outp++=arm_bilinear_interp_q7(&SBI,x,y);
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q7);

    } 

 
    void InterpolationTestsQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       const int16_t *pConfig;
       
       switch(id)
       {
        case InterpolationTestsQ7::TEST_LINEAR_INTERP_Q7_1:
          input.reload(InterpolationTestsQ7::INPUT_Q31_ID,mgr,nb);
          y.reload(InterpolationTestsQ7::YVAL_Q7_ID,mgr,nb);
          ref.reload(InterpolationTestsQ7::REF_LINEAR_Q7_ID,mgr,nb);

          break;

        case InterpolationTestsQ7::TEST_BILINEAR_INTERP_Q7_2:
          input.reload(InterpolationTestsQ7::INPUTBI_Q31_ID,mgr,nb);
          config.reload(InterpolationTestsQ7::CONFIGBI_S16_ID,mgr,nb);
          y.reload(InterpolationTestsQ7::YVALBI_Q7_ID,mgr,nb);
          ref.reload(InterpolationTestsQ7::REF_BILINEAR_Q7_ID,mgr,nb);

          pConfig = config.ptr();

          SBI.numRows = pConfig[1];
          SBI.numCols = pConfig[0];
          
          SBI.pData = y.ptr();
         
          break;

       }
      


       output.create(ref.nbSamples(),InterpolationTestsQ7::OUT_SAMPLES_Q7_ID,mgr);
    }

    void InterpolationTestsQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
