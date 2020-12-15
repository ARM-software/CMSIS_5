#include "BIQUADF16.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 27

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (5.0e-2)
#define ABS_ERROR (1.0e-1)

    void BIQUADF16::test_biquad_cascade_df1_ref()
    {


        float16_t *statep = state.ptr();
        float16_t *debugstatep = debugstate.ptr();

        const float16_t *coefsp = coefs.ptr();
        
        const float16_t *inputp = inputs.ptr();
        float16_t *outp = output.ptr();

        #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        arm_biquad_mod_coef_f16 *coefsmodp = (arm_biquad_mod_coef_f16*)vecCoefs.ptr();
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
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
           arm_biquad_cascade_df1_mve_init_f16(&this->Sdf1,3,coefsp,coefsmodp,statep);
#else
           arm_biquad_cascade_df1_init_f16(&this->Sdf1,3,coefsp,statep);
#endif

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df1_f16(&this->Sdf1,inputp,outp,blockSize);
        
           memcpy(debugstatep,statep,3*4*sizeof(float16_t));
           debugstatep += 3*4;

           outp += blockSize;
           
           inputp += blockSize;
           arm_biquad_cascade_df1_f16(&this->Sdf1,inputp,outp,blockSize);
           outp += blockSize;
          
           memcpy(debugstatep,statep,3*4*sizeof(float16_t));
           debugstatep += 3*4;

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

           ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);
  

    } 

    void BIQUADF16::test_biquad_cascade_df2T_ref()
    {


        float16_t *statep = state.ptr();


        float16_t *coefsp = coefs.ptr();
        
        const float16_t *inputp = inputs.ptr();
        float16_t *outp = output.ptr();

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

           arm_biquad_cascade_df2T_init_f16(&this->Sdf2T,3,coefsp,statep);


          
           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df2T_f16(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;

           
           inputp += blockSize;
           arm_biquad_cascade_df2T_f16(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;


           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

           ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);
  

    } 

    void BIQUADF16::test_biquad_cascade_df1_rand()
    {


        float16_t *statep = state.ptr();

        const float16_t *coefsp = coefs.ptr();
        const int16_t *configsp = configs.ptr();
        
        const float16_t *inputp = inputs.ptr();
        float16_t *outp = output.ptr();

        #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        arm_biquad_mod_coef_f16 *coefsmodp = (arm_biquad_mod_coef_f16*)vecCoefs.ptr();
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
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
           arm_biquad_cascade_df1_mve_init_f16(&this->Sdf1,numStages,coefsp,coefsmodp,statep);
#else
           arm_biquad_cascade_df1_init_f16(&this->Sdf1,numStages,coefsp,statep);
#endif


           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df1_f16(&this->Sdf1,inputp,outp,blockSize);

           inputp += blockSize;
           outp += blockSize;
           coefsp += numStages * 5;

           
           
        }

           ASSERT_EMPTY_TAIL(output);
           ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);
           ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);
  

    } 

    void BIQUADF16::test_biquad_cascade_df2T_rand()
    {


        float16_t *statep = state.ptr();
        const int16_t *configsp = configs.ptr();


        float16_t *coefsp = coefs.ptr();
        
        const float16_t *inputp = inputs.ptr();
        float16_t *outp = output.ptr();

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

           arm_biquad_cascade_df2T_init_f16(&this->Sdf2T,numStages,coefsp,statep);

           coefsp += numStages * 5;

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df2T_f16(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;
           inputp += blockSize;
           
        }

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

           ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);
  

    } 

    void BIQUADF16::test_biquad_cascade_stereo_df2T_rand()
    {


        float16_t *statep = state.ptr();
        const int16_t *configsp = configs.ptr();

        const float16_t *coefsp = coefs.ptr();

        
        const float16_t *inputp = inputs.ptr();
        float16_t *outp = output.ptr();

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
           arm_biquad_cascade_stereo_df2T_init_f16(&this->SStereodf2T,numStages,coefsp,statep);

           coefsp += numStages * 5;

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_stereo_df2T_f16(&this->SStereodf2T,inputp,outp,blockSize);
           outp += 2*blockSize;
           inputp += 2*blockSize;
           
        }

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

           ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);
  

    } 

    void BIQUADF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;
       switch(id)
       {
        case BIQUADF16::TEST_BIQUAD_CASCADE_DF1_REF_1:
            debugstate.create(2*64,BIQUADF16::STATE_F16_ID,mgr);

            inputs.reload(BIQUADF16::BIQUADINPUTS_F16_ID,mgr);
            coefs.reload(BIQUADF16::BIQUADCOEFS_F16_ID,mgr);
            ref.reload(BIQUADF16::BIQUADREFS_F16_ID,mgr);
            #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
            /* Max num stages is 47 in Python script */
            vecCoefs.create(96*47,BIQUADF16::OUT_F16_ID,mgr);
            #endif
        break;

        case BIQUADF16::TEST_BIQUAD_CASCADE_DF2T_REF_2:
           vecCoefs.create(64,BIQUADF16::OUT_F16_ID,mgr);

           inputs.reload(BIQUADF16::BIQUADINPUTS_F16_ID,mgr);
           coefs.reload(BIQUADF16::BIQUADCOEFS_F16_ID,mgr);
           ref.reload(BIQUADF16::BIQUADREFS_F16_ID,mgr);
        break;

        case BIQUADF16::TEST_BIQUAD_CASCADE_DF1_RAND_3:

            inputs.reload(BIQUADF16::ALLBIQUADINPUTS_F16_ID,mgr);
            coefs.reload(BIQUADF16::ALLBIQUADCOEFS_F16_ID,mgr);
            ref.reload(BIQUADF16::ALLBIQUADREFS_F16_ID,mgr);
            configs.reload(BIQUADF16::ALLBIQUADCONFIGS_S16_ID,mgr);
            #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
            /* Max num stages is 47 in Python script */
            vecCoefs.create(96*47,BIQUADF16::OUT_F16_ID,mgr);
            #endif
        break;

        case BIQUADF16::TEST_BIQUAD_CASCADE_DF2T_RAND_4:
           vecCoefs.create(512,BIQUADF16::OUT_F16_ID,mgr);

           inputs.reload(BIQUADF16::ALLBIQUADINPUTS_F16_ID,mgr);
           coefs.reload(BIQUADF16::ALLBIQUADCOEFS_F16_ID,mgr);
           ref.reload(BIQUADF16::ALLBIQUADREFS_F16_ID,mgr);
           configs.reload(BIQUADF16::ALLBIQUADCONFIGS_S16_ID,mgr);
        break;

        case BIQUADF16::TEST_BIQUAD_CASCADE_STEREO_DF2T_RAND_5:

           inputs.reload(BIQUADF16::ALLBIQUADSTEREOINPUTS_F16_ID,mgr);
           coefs.reload(BIQUADF16::ALLBIQUADCOEFS_F16_ID,mgr);
           ref.reload(BIQUADF16::ALLBIQUADSTEREOREFS_F16_ID,mgr);
           configs.reload(BIQUADF16::ALLBIQUADCONFIGS_S16_ID,mgr);
        break;

       }
      

       

       output.create(ref.nbSamples(),BIQUADF16::OUT_F16_ID,mgr);
      
       state.create(128,BIQUADF16::STATE_F16_ID,mgr);

       
    }

    void BIQUADF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
        switch(id)
        {
            case BIQUADF16::TEST_BIQUAD_CASCADE_DF1_REF_1:
               debugstate.dump(mgr);
            break;
        }
    }
