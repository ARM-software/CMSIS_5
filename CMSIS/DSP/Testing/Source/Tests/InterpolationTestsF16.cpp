#include "InterpolationTestsF16.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 55

/* 

Reference patterns are generated with
a double precision computation.

*/

#define REL_ERROR (5.0e-3)
#define ABS_ERROR (5.0e-3)



    void InterpolationTestsF16::test_linear_interp_f16()
    {
       const float16_t *inp = input.ptr();
       float16_t *outp = output.ptr();

       unsigned long nb;
       for(nb = 0; nb < input.nbSamples(); nb++)
       {
          outp[nb] = arm_linear_interp_f16(&S,inp[nb]);
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

       ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

 
    void InterpolationTestsF16::test_bilinear_interp_f16()
    {
       const float16_t *inp = input.ptr();
       float16_t *outp = output.ptr();
       float16_t x,y;
       unsigned long nb;
       for(nb = 0; nb < input.nbSamples(); nb += 2)
       {
          x = inp[nb];
          y = inp[nb+1];
          *outp++=arm_bilinear_interp_f16(&SBI,x,y);
       }

       ASSERT_EMPTY_TAIL(output);

       ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

       ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

 #if 0
    void InterpolationTestsF16::test_spline_square_f16()
    {
       const float16_t *inpX = inputX.ptr();
       const float16_t *inpY = inputY.ptr();
       const float16_t *outX = outputX.ptr();
       float16_t *outp = output.ptr();
       float16_t *buf = buffer.ptr();       // ((2*4-1)*sizeof(float16_t))
       float16_t *coef = splineCoefs.ptr(); // ((3*(4-1))*sizeof(float16_t))

       arm_spline_instance_f16 S;
       arm_spline_init_f16(&S, ARM_SPLINE_PARABOLIC_RUNOUT, inpX, inpY, 4, coef, buf);
       arm_spline_f16(&S, outX, outp, 20);

       ASSERT_EMPTY_TAIL(buffer);
       ASSERT_EMPTY_TAIL(splineCoefs);
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);
    } 

    void InterpolationTestsF16::test_spline_sine_f16()
    {
       const float16_t *inpX = inputX.ptr();
       const float16_t *inpY = inputY.ptr();
       const float16_t *outX = outputX.ptr();
       float16_t *outp = output.ptr();
       float16_t *buf = buffer.ptr(); // ((2*9-1)*sizeof(float16_t))
       float16_t *coef = splineCoefs.ptr(); // ((3*(9-1))*sizeof(float16_t))

       arm_spline_instance_f16 S;
       arm_spline_init_f16(&S, ARM_SPLINE_NATURAL, inpX, inpY, 9, coef, buf);
       arm_spline_f16(&S, outX, outp, 33);

       ASSERT_EMPTY_TAIL(buffer);
       ASSERT_EMPTY_TAIL(splineCoefs);
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);
    } 

    void InterpolationTestsF16::test_spline_ramp_f16()
    {
       const float16_t *inpX = inputX.ptr();
       const float16_t *inpY = inputY.ptr();
       const float16_t *outX = outputX.ptr();
       float16_t *outp = output.ptr();
       float16_t *buf = buffer.ptr(); // ((2*3-1)*sizeof(float16_t))
       float16_t *coef = splineCoefs.ptr(); // ((3*(3-1))*sizeof(float16_t))

       arm_spline_instance_f16 S;
       arm_spline_init_f16(&S, ARM_SPLINE_PARABOLIC_RUNOUT, inpX, inpY, 3, coef, buf);
       arm_spline_f16(&S, outX, outp, 30);

       ASSERT_EMPTY_TAIL(buffer);
       ASSERT_EMPTY_TAIL(splineCoefs);
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);
    } 
#endif

    void InterpolationTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       const int16_t *pConfig;
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;

       
       switch(id)
       {
        case InterpolationTestsF16::TEST_LINEAR_INTERP_F16_1:
          input.reload(InterpolationTestsF16::INPUT_F16_ID,mgr,nb);
          y.reload(InterpolationTestsF16::YVAL_F16_ID,mgr,nb);
          ref.reload(InterpolationTestsF16::REF_LINEAR_F16_ID,mgr,nb);

           
          S.nValues=y.nbSamples();           /**< nValues */
          /* Those values must be coherent with the ones in the 
          Python script generating the patterns */
          S.x1=0.0;               /**< x1 */
          S.xSpacing=1.0;         /**< xSpacing */
          S.pYData=y.ptr();          /**< pointer to the table of Y values */
          break;

         case InterpolationTestsF16::TEST_BILINEAR_INTERP_F16_2:
          input.reload(InterpolationTestsF16::INPUTBI_F16_ID,mgr,nb);
          config.reload(InterpolationTestsF16::CONFIGBI_S16_ID,mgr,nb);
          y.reload(InterpolationTestsF16::YVALBI_F16_ID,mgr,nb);
          ref.reload(InterpolationTestsF16::REF_BILINEAR_F16_ID,mgr,nb);

          pConfig = config.ptr();

          SBI.numRows = pConfig[1];
          SBI.numCols = pConfig[0];
          
          SBI.pData = y.ptr();
         
          break;
#if 0
          case TEST_SPLINE_SQUARE_F16_3:
             inputX.reload(InterpolationTestsF16::INPUT_SPLINE_SQU_X_F16_ID,mgr,4);
             inputY.reload(InterpolationTestsF16::INPUT_SPLINE_SQU_Y_F16_ID,mgr,4);
             outputX.reload(InterpolationTestsF16::OUTPUT_SPLINE_SQU_X_F16_ID,mgr,20);
             ref.reload(InterpolationTestsF16::REF_SPLINE_SQU_F16_ID,mgr,20);
             splineCoefs.create(3*(4-1),InterpolationTestsF16::COEFS_SPLINE_F16_ID,mgr);
             
             buffer.create(2*4-1,InterpolationTestsF16::TEMP_SPLINE_F16_ID,mgr);
             output.create(20,InterpolationTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;

          case TEST_SPLINE_SINE_F16_4:
             inputX.reload(InterpolationTestsF16::INPUT_SPLINE_SIN_X_F16_ID,mgr,9);
             inputY.reload(InterpolationTestsF16::INPUT_SPLINE_SIN_Y_F16_ID,mgr,9);
             outputX.reload(InterpolationTestsF16::OUTPUT_SPLINE_SIN_X_F16_ID,mgr,33);
             ref.reload(InterpolationTestsF16::REF_SPLINE_SIN_F16_ID,mgr,33);
             splineCoefs.create(3*(9-1),InterpolationTestsF16::COEFS_SPLINE_F16_ID,mgr);
             
             buffer.create(2*9-1,InterpolationTestsF16::TEMP_SPLINE_F16_ID,mgr);
             output.create(33,InterpolationTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;

          case TEST_SPLINE_RAMP_F16_5:
             inputX.reload(InterpolationTestsF16::INPUT_SPLINE_RAM_X_F16_ID,mgr,3);
             inputY.reload(InterpolationTestsF16::INPUT_SPLINE_RAM_Y_F16_ID,mgr,3);
             outputX.reload(InterpolationTestsF16::OUTPUT_SPLINE_RAM_X_F16_ID,mgr,30);
             ref.reload(InterpolationTestsF16::REF_SPLINE_RAM_F16_ID,mgr,30);
             splineCoefs.create(3*(3-1),InterpolationTestsF16::COEFS_SPLINE_F16_ID,mgr);
             
             buffer.create(2*3-1,InterpolationTestsF16::TEMP_SPLINE_F16_ID,mgr);
             output.create(30,InterpolationTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
#endif
       }
      


       output.create(ref.nbSamples(),InterpolationTestsF16::OUT_SAMPLES_F16_ID,mgr);
    }

    void InterpolationTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
