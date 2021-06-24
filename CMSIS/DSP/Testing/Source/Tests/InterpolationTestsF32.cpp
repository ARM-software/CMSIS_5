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

       unsigned long nb;
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
       unsigned long nb;
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

 
    void InterpolationTestsF32::test_spline_square_f32()
    {
       const float32_t *inpX = inputX.ptr();
       const float32_t *inpY = inputY.ptr();
       const float32_t *outX = outputX.ptr();
       float32_t *outp = output.ptr();
       float32_t *buf = buffer.ptr();       // ((2*4-1)*sizeof(float32_t))
       float32_t *coef = splineCoefs.ptr(); // ((3*(4-1))*sizeof(float32_t))

       arm_spline_instance_f32 S;
       arm_spline_init_f32(&S, ARM_SPLINE_PARABOLIC_RUNOUT, inpX, inpY, 4, coef, buf);
       arm_spline_f32(&S, outX, outp, 20);

       ASSERT_EMPTY_TAIL(buffer);
       ASSERT_EMPTY_TAIL(splineCoefs);
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
    } 

    void InterpolationTestsF32::test_spline_sine_f32()
    {
       const float32_t *inpX = inputX.ptr();
       const float32_t *inpY = inputY.ptr();
       const float32_t *outX = outputX.ptr();
       float32_t *outp = output.ptr();
       float32_t *buf = buffer.ptr(); // ((2*9-1)*sizeof(float32_t))
       float32_t *coef = splineCoefs.ptr(); // ((3*(9-1))*sizeof(float32_t))

       arm_spline_instance_f32 S;
       arm_spline_init_f32(&S, ARM_SPLINE_NATURAL, inpX, inpY, 9, coef, buf);
       arm_spline_f32(&S, outX, outp, 33);

       ASSERT_EMPTY_TAIL(buffer);
       ASSERT_EMPTY_TAIL(splineCoefs);
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
    } 

    void InterpolationTestsF32::test_spline_ramp_f32()
    {
       const float32_t *inpX = inputX.ptr();
       const float32_t *inpY = inputY.ptr();
       const float32_t *outX = outputX.ptr();
       float32_t *outp = output.ptr();
       float32_t *buf = buffer.ptr(); // ((2*3-1)*sizeof(float32_t))
       float32_t *coef = splineCoefs.ptr(); // ((3*(3-1))*sizeof(float32_t))

       arm_spline_instance_f32 S;
       arm_spline_init_f32(&S, ARM_SPLINE_PARABOLIC_RUNOUT, inpX, inpY, 3, coef, buf);
       arm_spline_f32(&S, outX, outp, 30);

       ASSERT_EMPTY_TAIL(buffer);
       ASSERT_EMPTY_TAIL(splineCoefs);
       ASSERT_EMPTY_TAIL(output);
       ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
    } 


    void InterpolationTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       const int16_t *pConfig;
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;

       
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

          case TEST_SPLINE_SQUARE_F32_3:
             inputX.reload(InterpolationTestsF32::INPUT_SPLINE_SQU_X_F32_ID,mgr,4);
             inputY.reload(InterpolationTestsF32::INPUT_SPLINE_SQU_Y_F32_ID,mgr,4);
             outputX.reload(InterpolationTestsF32::OUTPUT_SPLINE_SQU_X_F32_ID,mgr,20);
             ref.reload(InterpolationTestsF32::REF_SPLINE_SQU_F32_ID,mgr,20);
             splineCoefs.create(3*(4-1),InterpolationTestsF32::COEFS_SPLINE_F32_ID,mgr);
             
             buffer.create(2*4-1,InterpolationTestsF32::TEMP_SPLINE_F32_ID,mgr);
             output.create(20,InterpolationTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;

          case TEST_SPLINE_SINE_F32_4:
             inputX.reload(InterpolationTestsF32::INPUT_SPLINE_SIN_X_F32_ID,mgr,9);
             inputY.reload(InterpolationTestsF32::INPUT_SPLINE_SIN_Y_F32_ID,mgr,9);
             outputX.reload(InterpolationTestsF32::OUTPUT_SPLINE_SIN_X_F32_ID,mgr,33);
             ref.reload(InterpolationTestsF32::REF_SPLINE_SIN_F32_ID,mgr,33);
             splineCoefs.create(3*(9-1),InterpolationTestsF32::COEFS_SPLINE_F32_ID,mgr);
             
             buffer.create(2*9-1,InterpolationTestsF32::TEMP_SPLINE_F32_ID,mgr);
             output.create(33,InterpolationTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;

          case TEST_SPLINE_RAMP_F32_5:
             inputX.reload(InterpolationTestsF32::INPUT_SPLINE_RAM_X_F32_ID,mgr,3);
             inputY.reload(InterpolationTestsF32::INPUT_SPLINE_RAM_Y_F32_ID,mgr,3);
             outputX.reload(InterpolationTestsF32::OUTPUT_SPLINE_RAM_X_F32_ID,mgr,30);
             ref.reload(InterpolationTestsF32::REF_SPLINE_RAM_F32_ID,mgr,30);
             splineCoefs.create(3*(3-1),InterpolationTestsF32::COEFS_SPLINE_F32_ID,mgr);
             
             buffer.create(2*3-1,InterpolationTestsF32::TEMP_SPLINE_F32_ID,mgr);
             output.create(30,InterpolationTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
       }
      


       output.create(ref.nbSamples(),InterpolationTestsF32::OUT_SAMPLES_F32_ID,mgr);
    }

    void InterpolationTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
