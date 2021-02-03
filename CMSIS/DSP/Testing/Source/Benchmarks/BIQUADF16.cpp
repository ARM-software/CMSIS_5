#include "BIQUADF16.h"
#include "Error.h"

   
    void BIQUADF16::test_biquad_cascade_df1_f16()
    {
       arm_biquad_cascade_df1_f16(&instBiquadDf1, this->pSrc, this->pDst, this->nbSamples);
    } 

    void BIQUADF16::test_biquad_cascade_df2T_f16()
    {
       arm_biquad_cascade_df2T_f16(&instBiquadDf2T, this->pSrc, this->pDst, this->nbSamples);
    } 

  
    void BIQUADF16::test_biquad_cascade_stereo_df2T_f16()
    {
       arm_biquad_cascade_stereo_df2T_f16(&instStereo, this->pSrc, this->pDst, this->nbSamples);
    } 


    
    void BIQUADF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       
       std::vector<Testing::param_t>::iterator it = params.begin();
       this->numStages = *it++;
       this->nbSamples = *it;

       
       

       switch(id)
       {
           case TEST_BIQUAD_CASCADE_DF1_F16_1:
                  samples.reload(BIQUADF16::SAMPLES1_F16_ID,mgr,this->nbSamples);
                  output.create(this->nbSamples,BIQUADF16::OUT_SAMPLES_F16_ID,mgr);
                  coefs.reload(BIQUADF16::COEFS1_F16_ID,mgr,this->numStages * 5);
                  state.create(4*this->numStages,BIQUADF16::STATE_F16_ID,mgr);

                  arm_biquad_cascade_df1_init_f16(&instBiquadDf1,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());

           break;

           case TEST_BIQUAD_CASCADE_DF2T_F16_2:
               samples.reload(BIQUADF16::SAMPLES1_F16_ID,mgr,this->nbSamples);
               output.create(this->nbSamples,BIQUADF16::OUT_SAMPLES_F16_ID,mgr);
               coefs.reload(BIQUADF16::COEFS1_F16_ID,mgr,this->numStages * 5);
               state.create(2*this->numStages,BIQUADF16::STATE_F16_ID,mgr);


#if defined(ARM_MATH_NEON)
               // For Neon, neonCoefs is the coef array and is bigger
               neonCoefs.create(8*this->numStages,BIQUADF16::STATE_F16_ID,mgr);

               arm_biquad_cascade_df2T_init_f16(&instBiquadDf2T,
                    this->numStages,
                    neonCoefs.ptr(),
                    state.ptr());

               // Those Neon coefs must be computed from original coefs
               arm_biquad_cascade_df2T_compute_coefs_f16(&instBiquadDf2T,this->numStages,coefs.ptr());
#else
                  
              // For cortex-M, coefs is the coef array
              arm_biquad_cascade_df2T_init_f16(&instBiquadDf2T,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());

                
#endif
           break;

           case TEST_BIQUAD_CASCADE_STEREO_DF2T_F16_3:
                  samples.reload(BIQUADF16::SAMPLES1_F16_ID,mgr,2*this->nbSamples);
                  output.create(2*this->nbSamples,BIQUADF16::OUT_SAMPLES_F16_ID,mgr);
                  coefs.reload(BIQUADF16::COEFS1_F16_ID,mgr,this->numStages * 5);
                  state.create(4*this->numStages,BIQUADF16::STATE_F16_ID,mgr);

                  arm_biquad_cascade_stereo_df2T_init_f16(&instStereo,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());
           break;
       }
       
       this->pSrc=samples.ptr();
       this->pDst=output.ptr();

    }

    void BIQUADF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        (void)mgr;
    }
