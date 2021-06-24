#include "BIQUADF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 98

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.2e-3)


    void BIQUADF32::test_biquad_cascade_df1_ref()
    {


        float32_t *statep = state.ptr();
        float32_t *debugstatep = debugstate.ptr();

        const float32_t *coefsp = coefs.ptr();
        
        const float32_t *inputp = inputs.ptr();
        float32_t *outp = output.ptr();

        #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        arm_biquad_mod_coef_f32 *coefsmodp = (arm_biquad_mod_coef_f32*)vecCoefs.ptr();
        #endif

        int blockSize;

        

        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           blockSize = inputs.nbSamples() >> 1;

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
           arm_biquad_cascade_df1_mve_init_f32(&this->Sdf1,3,coefsp,coefsmodp,statep);
#else
           arm_biquad_cascade_df1_init_f32(&this->Sdf1,3,coefsp,statep);
#endif

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df1_f32(&this->Sdf1,inputp,outp,blockSize);
        
           memcpy(debugstatep,statep,3*4*sizeof(float32_t));
           debugstatep += 3*4;

           outp += blockSize;
           
           inputp += blockSize;
           arm_biquad_cascade_df1_f32(&this->Sdf1,inputp,outp,blockSize);
           outp += blockSize;
          
           memcpy(debugstatep,statep,3*4*sizeof(float32_t));
           debugstatep += 3*4;

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

    void BIQUADF32::test_biquad_cascade_df2T_ref()
    {


        float32_t *statep = state.ptr();

#if !defined(ARM_MATH_NEON) 
        const float32_t *coefsp = coefs.ptr();
#else
        float32_t *coefsp = coefs.ptr();
#endif
        
        const float32_t *inputp = inputs.ptr();
        float32_t *outp = output.ptr();

        int blockSize;

        

        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           blockSize = inputs.nbSamples() >> 1;

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
#if !defined(ARM_MATH_NEON) 
           arm_biquad_cascade_df2T_init_f32(&this->Sdf2T,3,coefsp,statep);
#else
           float32_t *vecCoefsPtr = vecCoefs.ptr();

           arm_biquad_cascade_df2T_init_f32(&this->Sdf2T,
                    3,
                    vecCoefsPtr,
                    statep);

           // Those Neon coefs must be computed from original coefs
           arm_biquad_cascade_df2T_compute_coefs_f32(&this->Sdf2T,3,coefsp);
#endif

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df2T_f32(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;

           
           inputp += blockSize;
           arm_biquad_cascade_df2T_f32(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;


           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

    void BIQUADF32::test_biquad_cascade_df1_rand()
    {


        float32_t *statep = state.ptr();

        const float32_t *coefsp = coefs.ptr();
        const int16_t *configsp = configs.ptr();
        
        const float32_t *inputp = inputs.ptr();
        float32_t *outp = output.ptr();

        #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        arm_biquad_mod_coef_f32 *coefsmodp = (arm_biquad_mod_coef_f32*)vecCoefs.ptr();
        #endif

        int blockSize;
        int numStages;
        unsigned long i;

        

        for(i=0;i < configs.nbSamples(); i+=2)
        {
        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           
           numStages = configsp[0];
           blockSize = configsp[1];

           configsp += 2;

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
           arm_biquad_cascade_df1_mve_init_f32(&this->Sdf1,numStages,coefsp,coefsmodp,statep);
#else
           arm_biquad_cascade_df1_init_f32(&this->Sdf1,numStages,coefsp,statep);
#endif


           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df1_f32(&this->Sdf1,inputp,outp,blockSize);

           inputp += blockSize;
           outp += blockSize;
           coefsp += numStages * 5;

           
           
        }
           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

    void BIQUADF32::test_biquad_cascade_df2T_rand()
    {


        float32_t *statep = state.ptr();
        const int16_t *configsp = configs.ptr();

#if !defined(ARM_MATH_NEON) 
        const float32_t *coefsp = coefs.ptr();
#else
        float32_t *coefsp = coefs.ptr();
#endif
        
        const float32_t *inputp = inputs.ptr();
        float32_t *outp = output.ptr();

        int blockSize;
        int numStages;

        unsigned long i;

        

        for(i=0;i < configs.nbSamples(); i+=2)
        {

        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           numStages = configsp[0];
           blockSize = configsp[1];

           configsp += 2;

          

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
#if !defined(ARM_MATH_NEON) 
           arm_biquad_cascade_df2T_init_f32(&this->Sdf2T,numStages,coefsp,statep);
#else
           float32_t *vecCoefsPtr = vecCoefs.ptr();

           arm_biquad_cascade_df2T_init_f32(&this->Sdf2T,
                    numStages,
                    vecCoefsPtr,
                    statep);

           // Those Neon coefs must be computed from original coefs
           arm_biquad_cascade_df2T_compute_coefs_f32(&this->Sdf2T,numStages,coefsp);
#endif
           coefsp += numStages * 5;

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df2T_f32(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;
           inputp += blockSize;
           
        }

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

    void BIQUADF32::test_biquad_cascade_stereo_df2T_rand()
    {


        float32_t *statep = state.ptr();
        const int16_t *configsp = configs.ptr();

        const float32_t *coefsp = coefs.ptr();

        
        const float32_t *inputp = inputs.ptr();
        float32_t *outp = output.ptr();

        int blockSize;
        int numStages;

        unsigned long i;

        

        for(i=0;i < configs.nbSamples(); i+=2)
        {

        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           numStages = configsp[0];
           blockSize = configsp[1];

           configsp += 2;

          

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
           arm_biquad_cascade_stereo_df2T_init_f32(&this->SStereodf2T,numStages,coefsp,statep);

           coefsp += numStages * 5;

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_stereo_df2T_f32(&this->SStereodf2T,inputp,outp,blockSize);
           outp += 2*blockSize;
           inputp += 2*blockSize;
           
        }

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

    void BIQUADF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;
       switch(id)
       {
        case BIQUADF32::TEST_BIQUAD_CASCADE_DF1_REF_1:
            debugstate.create(2*64,BIQUADF32::STATE_F32_ID,mgr);

            inputs.reload(BIQUADF32::BIQUADINPUTS_F32_ID,mgr);
            coefs.reload(BIQUADF32::BIQUADCOEFS_F32_ID,mgr);
            ref.reload(BIQUADF32::BIQUADREFS_F32_ID,mgr);
            #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            /* Max num stages is 47 in Python script */
            vecCoefs.create(32*47,BIQUADF32::OUT_F32_ID,mgr);
            #endif
        break;

        case BIQUADF32::TEST_BIQUAD_CASCADE_DF2T_REF_2:
           vecCoefs.create(64,BIQUADF32::OUT_F32_ID,mgr);

           inputs.reload(BIQUADF32::BIQUADINPUTS_F32_ID,mgr);
           coefs.reload(BIQUADF32::BIQUADCOEFS_F32_ID,mgr);
           ref.reload(BIQUADF32::BIQUADREFS_F32_ID,mgr);
        break;

        case BIQUADF32::TEST_BIQUAD_CASCADE_DF1_RAND_3:

            inputs.reload(BIQUADF32::ALLBIQUADINPUTS_F32_ID,mgr);
            coefs.reload(BIQUADF32::ALLBIQUADCOEFS_F32_ID,mgr);
            ref.reload(BIQUADF32::ALLBIQUADREFS_F32_ID,mgr);
            configs.reload(BIQUADF32::ALLBIQUADCONFIGS_S16_ID,mgr);
            #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            /* Max num stages is 47 in Python script */
            vecCoefs.create(32*47,BIQUADF32::OUT_F32_ID,mgr);
            #endif
        break;

        case BIQUADF32::TEST_BIQUAD_CASCADE_DF2T_RAND_4:
           vecCoefs.create(512,BIQUADF32::OUT_F32_ID,mgr);

           inputs.reload(BIQUADF32::ALLBIQUADINPUTS_F32_ID,mgr);
           coefs.reload(BIQUADF32::ALLBIQUADCOEFS_F32_ID,mgr);
           ref.reload(BIQUADF32::ALLBIQUADREFS_F32_ID,mgr);
           configs.reload(BIQUADF32::ALLBIQUADCONFIGS_S16_ID,mgr);
        break;

        case BIQUADF32::TEST_BIQUAD_CASCADE_STEREO_DF2T_RAND_5:

           inputs.reload(BIQUADF32::ALLBIQUADSTEREOINPUTS_F32_ID,mgr);
           coefs.reload(BIQUADF32::ALLBIQUADCOEFS_F32_ID,mgr);
           ref.reload(BIQUADF32::ALLBIQUADSTEREOREFS_F32_ID,mgr);
           configs.reload(BIQUADF32::ALLBIQUADCONFIGS_S16_ID,mgr);
        break;

       }
      

       

       output.create(ref.nbSamples(),BIQUADF32::OUT_F32_ID,mgr);
      
       state.create(128,BIQUADF32::STATE_F32_ID,mgr);

       
    }

    void BIQUADF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
        switch(id)
        {
            case BIQUADF32::TEST_BIQUAD_CASCADE_DF1_REF_1:
               debugstate.dump(mgr);
            break;
        }
    }
