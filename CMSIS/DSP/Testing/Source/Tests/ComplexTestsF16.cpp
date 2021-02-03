#include "ComplexTestsF16.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 39

#define REL_ERROR (6.0e-2)

    void ComplexTestsF16::test_cmplx_conj_f16()
    {
        const float16_t *inp1=input1.ptr();
        float16_t *outp=output.ptr();


        arm_cmplx_conj_f16(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);


    } 


    void ComplexTestsF16::test_cmplx_dot_prod_f16()
    {
        float16_t re,im;

        const float16_t *inp1=input1.ptr();
        const float16_t *inp2=input2.ptr();
        float16_t *outp=output.ptr();

        arm_cmplx_dot_prod_f16(inp1,inp2,input1.nbSamples() >> 1,&re,&im);

        outp[0] = re;
        outp[1] = im;

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

        ASSERT_EMPTY_TAIL(output);
    } 

    void ComplexTestsF16::test_cmplx_mag_f16()
    {
        const float16_t *inp1=input1.ptr();
        float16_t *outp=output.ptr();

        arm_cmplx_mag_f16(inp1,outp,input1.nbSamples() >> 1 );
        
        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF16::test_cmplx_mag_squared_f16()
    {
        const float16_t *inp1=input1.ptr();
        float16_t *outp=output.ptr();

        arm_cmplx_mag_squared_f16(inp1,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF16::test_cmplx_mult_cmplx_f16()
    {
        const float16_t *inp1=input1.ptr();
        const float16_t *inp2=input2.ptr();
        float16_t *outp=output.ptr();

        arm_cmplx_mult_cmplx_f16(inp1,inp2,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void ComplexTestsF16::test_cmplx_mult_real_f16()
    {
        const float16_t *inp1=input1.ptr();
        const float16_t *inp2=input2.ptr();
        float16_t *outp=output.ptr();

        arm_cmplx_mult_real_f16(inp1,inp2,outp,input1.nbSamples() >> 1 );

        ASSERT_EMPTY_TAIL(output);
        

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 
 
    void ComplexTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;

       
       switch(id)
       {
        case ComplexTestsF16::TEST_CMPLX_CONJ_F16_1:
          nb = 7;
          ref.reload(ComplexTestsF16::REF_CONJ_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_CONJ_F16_2:
          nb = 16;
          ref.reload(ComplexTestsF16::REF_CONJ_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_CONJ_F16_3:
          nb = 23;
          ref.reload(ComplexTestsF16::REF_CONJ_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_DOT_PROD_F16_4:
          nb = 7;
          ref.reload(ComplexTestsF16::REF_DOT_PROD_3_F16_ID,mgr);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;

        case ComplexTestsF16::TEST_CMPLX_DOT_PROD_F16_5:
          nb = 16;
          ref.reload(ComplexTestsF16::REF_DOT_PROD_4N_F16_ID,mgr);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;

        case ComplexTestsF16::TEST_CMPLX_DOT_PROD_F16_6:
          nb = 23;
          ref.reload(ComplexTestsF16::REF_DOT_PROD_4N1_F16_ID,mgr);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MAG_F16_7:
          nb = 7;
          ref.reload(ComplexTestsF16::REF_MAG_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MAG_F16_8:
          nb = 16;
          ref.reload(ComplexTestsF16::REF_MAG_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MAG_F16_9:
          nb = 23;
          ref.reload(ComplexTestsF16::REF_MAG_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MAG_SQUARED_F16_10:
          nb = 7;
          ref.reload(ComplexTestsF16::REF_MAG_SQUARED_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MAG_SQUARED_F16_11:
          nb = 16;
          ref.reload(ComplexTestsF16::REF_MAG_SQUARED_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MAG_SQUARED_F16_12:
          nb = 23;
          ref.reload(ComplexTestsF16::REF_MAG_SQUARED_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MULT_CMPLX_F16_13:
          nb = 7;
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_CMPLX_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MULT_CMPLX_F16_14:
          nb = 16;
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_CMPLX_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MULT_CMPLX_F16_15:
          nb = 23;
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_CMPLX_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MULT_REAL_F16_16:
          nb = 7;
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_REAL_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT3_F16_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MULT_REAL_F16_17:
          nb = 16;
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_REAL_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT3_F16_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;
        case ComplexTestsF16::TEST_CMPLX_MULT_REAL_F16_18:
          nb = 23;
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_REAL_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT3_F16_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
          break;

        case ComplexTestsF16::TEST_CMPLX_CONJ_F16_19:
          ref.reload(ComplexTestsF16::REF_CONJ_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
        break;

        case ComplexTestsF16::TEST_CMPLX_DOT_PROD_F16_20:
          ref.reload(ComplexTestsF16::REF_DOT_PROD_LONG_F16_ID,mgr);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
        break;
        
        case ComplexTestsF16::TEST_CMPLX_MAG_F16_21:
          ref.reload(ComplexTestsF16::REF_MAG_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
        break;
        
        case ComplexTestsF16::TEST_CMPLX_MAG_SQUARED_F16_22:
          ref.reload(ComplexTestsF16::REF_MAG_SQUARED_F16_ID,mgr,nb);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
        break;
        
        case ComplexTestsF16::TEST_CMPLX_MULT_CMPLX_F16_23:
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_CMPLX_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT2_F16_ID,mgr,nb << 1);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
        break;
        
        case ComplexTestsF16::TEST_CMPLX_MULT_REAL_F16_24:
          ref.reload(ComplexTestsF16::REF_CMPLX_MULT_REAL_F16_ID,mgr,nb << 1);
          input1.reload(ComplexTestsF16::INPUT1_F16_ID,mgr,nb << 1);
          input2.reload(ComplexTestsF16::INPUT3_F16_ID,mgr,nb);

          output.create(ref.nbSamples(),ComplexTestsF16::OUT_SAMPLES_F16_ID,mgr);
        break;
        
       }
      
    }

    void ComplexTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
