#include "InterpolationTestsF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (8.0e-5)



    void InterpolationTestsF32::test_linear_interp_f32()
    {
       const float32_t *inp = input.ptr();
       float32_t *outp = output.ptr();

       int nb;
       for(nb = 0; nb < input.nbSamples(); nb++)
       {
          outp[nb] = arm_linear_interp_f32(&S,inp[nb]);
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

 
    void InterpolationTestsF32::test_bilinear_interp_f32()
    {
       const float32_t *inp = input.ptr();
       float32_t *outp = output.ptr();
       float32_t x,y;
       int nb;
       for(nb = 0; nb < input.nbSamples(); nb += 2)
       {
          x = inp[nb];
          y = inp[nb+1];
          *outp++=arm_bilinear_interp_f32(&SBI,x,y);
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

       ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

 
    void InterpolationTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       const int16_t *pConfig;
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case InterpolationTestsF32::TEST_LINEAR_INTERP_F32_1:
          input.reload(InterpolationTestsF32::INPUT_F32_ID,mgr,nb);
          y.reload(InterpolationTestsF32::YVAL_F32_ID,mgr,nb);
          ref.reload(InterpolationTestsF32::REF_LINEAR_F32_ID,mgr,nb);

           
          S.nValues=y.nbSamples();           /**< nValues */
          /* Those values must be coherent with the ones in the 
          Python script generating the patterns */
          S.x1=0.0;               /**< x1 */
          S.xSpacing=1.0;         /**< xSpacing */
          S.pYData=y.ptr();          /**< pointer to the table of Y values */
          break;

         case InterpolationTestsF32::TEST_BILINEAR_INTERP_F32_2:
          input.reload(InterpolationTestsF32::INPUTBI_F32_ID,mgr,nb);
          config.reload(InterpolationTestsF32::CONFIGBI_S16_ID,mgr,nb);
          y.reload(InterpolationTestsF32::YVALBI_F32_ID,mgr,nb);
          ref.reload(InterpolationTestsF32::REF_BILINEAR_F32_ID,mgr,nb);

          pConfig = config.ptr();

          SBI.numRows = pConfig[1];
          SBI.numCols = pConfig[0];
          
          SBI.pData = y.ptr();
         
          break;
       }
      


       output.create(ref.nbSamples(),InterpolationTestsF32::OUT_SAMPLES_F32_ID,mgr);
    }

    void InterpolationTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
