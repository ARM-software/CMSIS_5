#include "ComplexTestsF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

#define REL_ERROR (7.0e-6)

    void ComplexTestsF32::test_cmplx_conj_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();


        arm_cmplx_conj_f32(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);


    } 


    void ComplexTestsF32::test_cmplx_dot_prod_f32()
    {
        float32_t re,im;

        const float32_t *inp1=input1.ptr();
        const float32_t *inp2=input2.ptr();
        float32_t *outp=output.ptr();

        arm_cmplx_dot_prod_f32(inp1,inp2,input1.nbSamples() >> 1,&re,&im);

        outp[0] = re;
        outp[1] = im;

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

        ASSERT_EMPTY_TAIL(output);
    } 

    void ComplexTestsF32::test_cmplx_mag_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_cmplx_mag_f32(inp1,outp,input1.nbSamples() >> 1 );
        
        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF32::test_cmplx_mag_squared_f32()
    {
        const float32_t *inp1=input1.ptr();
        float32_t *outp=output.ptr();

        arm_cmplx_mag_squared_f32(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF32::test_cmplx_mult_cmplx_f32()
    {
        const float32_t *inp1=input1.ptr();
        const float32_t *inp2=input2.ptr();
        float32_t *outp=output.ptr();

        arm_cmplx_mult_cmplx_f32(inp1,inp2,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF32::test_cmplx_mult_real_f32()
    {
        const float32_t *inp1=input1.ptr();
        const float32_t *inp2=input2.ptr();
        float32_t *outp=output.ptr();

        arm_cmplx_mult_real_f32(inp1,inp2,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 
 
    void ComplexTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;

       
       switch(id)
       {
        case ComplexTestsF32::TEST_CMPLX_CONJ_F32_1:
          nb = 3;
          ref.reload(ComplexTestsF32::REF_CONJ_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_CONJ_F32_2:
          nb = 8;
          ref.reload(ComplexTestsF32::REF_CONJ_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_CONJ_F32_3:
          nb = 11;
          ref.reload(ComplexTestsF32::REF_CONJ_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_DOT_PROD_F32_4:
          nb = 3;
          ref.reload(ComplexTestsF32::REF_DOT_PROD_3_F32_ID,mgr);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;

        case ComplexTestsF32::TEST_CMPLX_DOT_PROD_F32_5:
          nb = 8;
          ref.reload(ComplexTestsF32::REF_DOT_PROD_4N_F32_ID,mgr);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;

        case ComplexTestsF32::TEST_CMPLX_DOT_PROD_F32_6:
          nb = 11;
          ref.reload(ComplexTestsF32::REF_DOT_PROD_4N1_F32_ID,mgr);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MAG_F32_7:
          nb = 3;
          ref.reload(ComplexTestsF32::REF_MAG_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MAG_F32_8:
          nb = 8;
          ref.reload(ComplexTestsF32::REF_MAG_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MAG_F32_9:
          nb = 11;
          ref.reload(ComplexTestsF32::REF_MAG_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MAG_SQUARED_F32_10:
          nb = 3;
          ref.reload(ComplexTestsF32::REF_MAG_SQUARED_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MAG_SQUARED_F32_11:
          nb = 8;
          ref.reload(ComplexTestsF32::REF_MAG_SQUARED_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MAG_SQUARED_F32_12:
          nb = 11;
          ref.reload(ComplexTestsF32::REF_MAG_SQUARED_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MULT_CMPLX_F32_13:
          nb = 3;
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_CMPLX_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MULT_CMPLX_F32_14:
          nb = 8;
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_CMPLX_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MULT_CMPLX_F32_15:
          nb = 11;
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_CMPLX_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MULT_REAL_F32_16:
          nb = 3;
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_REAL_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT3_F32_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MULT_REAL_F32_17:
          nb = 8;
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_REAL_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT3_F32_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;
        case ComplexTestsF32::TEST_CMPLX_MULT_REAL_F32_18:
          nb = 11;
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_REAL_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT3_F32_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
          break;

        case ComplexTestsF32::TEST_CMPLX_CONJ_F32_19:
          ref.reload(ComplexTestsF32::REF_CONJ_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
        break;

        case ComplexTestsF32::TEST_CMPLX_DOT_PROD_F32_20:
          ref.reload(ComplexTestsF32::REF_DOT_PROD_LONG_F32_ID,mgr);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
        break;
        
        case ComplexTestsF32::TEST_CMPLX_MAG_F32_21:
          ref.reload(ComplexTestsF32::REF_MAG_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
        break;
        
        case ComplexTestsF32::TEST_CMPLX_MAG_SQUARED_F32_22:
          ref.reload(ComplexTestsF32::REF_MAG_SQUARED_F32_ID,mgr,nb);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
        break;
        
        case ComplexTestsF32::TEST_CMPLX_MULT_CMPLX_F32_23:
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_CMPLX_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT2_F32_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
        break;
        
        case ComplexTestsF32::TEST_CMPLX_MULT_REAL_F32_24:
          ref.reload(ComplexTestsF32::REF_CMPLX_MULT_REAL_F32_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF32::INPUT1_F32_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF32::INPUT3_F32_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF32::OUT_SAMPLES_F32_ID,mgr);
        break;
        
       }
      
    }

    void ComplexTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
