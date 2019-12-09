#include "IIRTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>

#define SNR_THRESHOLD 100
#define REL_ERROR (1.2e-3)

#define GET_F32_PTR()             \
float32_t *inp=input.ptr(); \
float32_t *cfs=coefs.ptr(); \
float32_t *stp=state.ptr(); \
float32_t *outp=output.ptr();     \
float32_t *refp=ref.ptr();

//arm_iir_init_f32(S, iirType, order, nbCascaded, simdFlag, debugFlag, pState, pCoeffs )

    void IIRTestsF32::test_iir_1st_df1()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        arm_iir_init_f32(&S, ARM_IIR_DF1, 1, 1, 0, 0, stp, cfs );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_2nd_df1()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

#if defined(ARM_MATH_MVEF)
        /* MVE simd only */
        float32_t coefssimd[32];
        memcpy(coefssimd, cfs, 5*sizeof(float32_t) );
        arm_iir_init_f32(&S, ARM_IIR_DF1, 2, 1, 0, 0, stp, coefssimd );
#else
        arm_iir_init_f32(&S, ARM_IIR_DF1, 2, 1, 0, 0, stp, cfs );
#endif
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_3rd_df1()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        arm_iir_init_f32(&S, ARM_IIR_DF1, 3, 1, 0, 0, stp, cfs );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_5th_df1()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        arm_iir_init_f32(&S, ARM_IIR_DF1, 5, 1, 0, 0, stp, cfs );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_1st_df1_simd()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 1, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[24];
        memcpy(coefssimd, cfs, 3*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 1, 1, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_2nd_df1_simd()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;
        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 2, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[32];
        memcpy(coefssimd, cfs, 5*sizeof(float32_t) );
        arm_iir_init_f32(&S, ARM_IIR_DF1, 2, 1, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_3rd_df1_simd()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 3, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[40];
        memcpy(coefssimd, cfs, 7*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 3, 1, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_5th_df1_simd()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 3, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[56];
        memcpy(coefssimd, cfs, 11*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 5, 1, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 


    void IIRTestsF32::test_iir_1st_df1_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        arm_iir_init_f32(&S, ARM_IIR_DF1, 1, 2, 0, 0, stp, cfs );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_2nd_df1_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;
#if defined(ARM_MATH_MVEF)
        /* MVE simd only */
        float32_t coefssimd[32*2];
        memcpy(coefssimd, cfs, 2*5*sizeof(float32_t) );
        arm_iir_init_f32(&S, ARM_IIR_DF1, 2, 2, 0, 0, stp, coefssimd );
#else
        arm_iir_init_f32(&S, ARM_IIR_DF1, 2, 2, 0, 0, stp, cfs );
#endif
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_3rd_df1_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        arm_iir_init_f32(&S, ARM_IIR_DF1, 3, 2, 0, 0, stp, cfs );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_5th_df1_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        arm_iir_init_f32(&S, ARM_IIR_DF1, 5, 2, 0, 0, stp, cfs );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_1st_df1_simd_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 1, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[24*2];
        memcpy(coefssimd, cfs, 3*2*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 1, 2, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_2nd_df1_simd_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 2, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[32*2];
        memcpy(coefssimd, cfs, 5*2*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 2, 2, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_3rd_df1_simd_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 3, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[40*2];
        memcpy(coefssimd, cfs, 7*2*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 3, 2, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void IIRTestsF32::test_iir_5th_df1_simd_casc()
    {
        GET_F32_PTR();

        arm_iir_instance_f32 S;

        //uint32_t stateSize, coeffSize, stateAlign, coeffAlign;
        //arm_iir_req_f32(ARM_IIR_DF1, 3, 1, 1, &stateSize, &coeffSize, &stateAlign, &coeffAlign);
        //uint32_t cfSize = coeffSize/sizeof(float32_t);
        float32_t coefssimd[56*2];
        memcpy(coefssimd, cfs, 11*2*sizeof(float32_t) );

        arm_iir_init_f32(&S, ARM_IIR_DF1, 5, 2, 1, 0, stp, coefssimd );
        arm_iir_f32(&S,inp,outp,11);

        ASSERT_EMPTY_TAIL(output);
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 


    void IIRTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       
       switch(id)
       {
            case IIRTestsF32::TEST_IIR_1ST_DF1_1:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_1ST_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_1ST_F32_ID, mgr, 3);
            ref.reload(IIRTestsF32::REF_IIR_1ST_F32_ID, mgr, nb);
            state.create(2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_2ND_DF1_2:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_2ND_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_2ND_F32_ID, mgr, 5);
            ref.reload(IIRTestsF32::REF_IIR_2ND_F32_ID, mgr, nb);
            state.create(4,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_3RD_DF1_3:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_3RD_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_3RD_F32_ID, mgr, 7);
            ref.reload(IIRTestsF32::REF_IIR_3RD_F32_ID, mgr, nb);
            state.create(6,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_5TH_DF1_4:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_5TH_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_5TH_F32_ID, mgr, 11);
            ref.reload(IIRTestsF32::REF_IIR_5TH_F32_ID, mgr, nb);
            state.create(10,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_1ST_DF1_SIMD_5:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_1ST_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_1ST_F32_ID, mgr, 3);
            ref.reload(IIRTestsF32::REF_IIR_1ST_F32_ID, mgr, nb);
            state.create(2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_2ND_DF1_SIMD_6:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_2ND_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_2ND_F32_ID, mgr, 5);
            ref.reload(IIRTestsF32::REF_IIR_2ND_F32_ID, mgr, nb);
            state.create(4,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_3RD_DF1_SIMD_7:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_3RD_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_3RD_F32_ID, mgr, 7);
            ref.reload(IIRTestsF32::REF_IIR_3RD_F32_ID, mgr, nb);
            state.create(6,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_5TH_DF1_SIMD_8:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_5TH_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_5TH_F32_ID, mgr, 11);
            ref.reload(IIRTestsF32::REF_IIR_5TH_F32_ID, mgr, nb);
            state.create(10,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_1ST_DF1_CASC_9:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_1ST_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_1ST_CASC_F32_ID, mgr, 3*2);
            ref.reload(IIRTestsF32::REF_IIR_1ST_CASC_F32_ID, mgr, nb);
            state.create(2*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_2ND_DF1_CASC_10:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_2ND_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_2ND_CASC_F32_ID, mgr, 5*2);
            ref.reload(IIRTestsF32::REF_IIR_2ND_CASC_F32_ID, mgr, nb);
            state.create(4*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_3RD_DF1_CASC_11:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_3RD_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_3RD_CASC_F32_ID, mgr, 7*2);
            ref.reload(IIRTestsF32::REF_IIR_3RD_CASC_F32_ID, mgr, nb);
            state.create(6*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_5TH_DF1_CASC_12:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_5TH_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_5TH_CASC_F32_ID, mgr, 11*2);
            ref.reload(IIRTestsF32::REF_IIR_5TH_CASC_F32_ID, mgr, nb);
            state.create(10*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_1ST_DF1_SIMD_CASC_13:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_1ST_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_1ST_CASC_F32_ID, mgr, 3*2);
            ref.reload(IIRTestsF32::REF_IIR_1ST_CASC_F32_ID, mgr, nb);
            state.create(2*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_2ND_DF1_SIMD_CASC_14:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_2ND_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_2ND_CASC_F32_ID, mgr, 5*2);
            ref.reload(IIRTestsF32::REF_IIR_2ND_CASC_F32_ID, mgr, nb);
            state.create(4*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_3RD_DF1_SIMD_CASC_15:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_3RD_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_3RD_CASC_F32_ID, mgr, 7*2);
            ref.reload(IIRTestsF32::REF_IIR_3RD_CASC_F32_ID, mgr, nb);
            state.create(6*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;

            case IIRTestsF32::TEST_IIR_5TH_DF1_SIMD_CASC_16:
            nb = 11;
            input.reload(IIRTestsF32::INPUT_IIR_5TH_F32_ID, mgr, nb);
            coefs.reload(IIRTestsF32::COEFS_IIR_5TH_CASC_F32_ID, mgr, 11*2);
            ref.reload(IIRTestsF32::REF_IIR_5TH_CASC_F32_ID, mgr, nb);
            state.create(10*2,IIRTestsF32::STATE_F32_ID,mgr);
            break;
       }

       output.create(ref.nbSamples(), IIRTestsF32::OUT_SAMPLES_F32_ID, mgr);
    }

    void IIRTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }

