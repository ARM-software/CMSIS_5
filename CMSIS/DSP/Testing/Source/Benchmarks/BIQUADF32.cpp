#include "BIQUADF32.h"
#include "Error.h"

   
    void BIQUADF32::test_biquad_cascade_df1_f32()
    {
       
       
       const float32_t *pSrc=samples.ptr();
       float32_t *pDst=output.ptr();


       arm_biquad_cascade_df1_f32(&instBiquadDf1, pSrc, pDst, this->nbSamples);
       
        
    } 

    void BIQUADF32::test_biquad_cascade_df2T_f32()
    {
       const float32_t *pSrc=samples.ptr();
       float32_t *pDst=output.ptr();


       arm_biquad_cascade_df2T_f32(&instBiquadDf2T, pSrc, pDst, this->nbSamples);
        
    } 

  
    void BIQUADF32::test_biquad_cascade_stereo_df2T_f32()
    {
       
       const float32_t *pSrc=samples.ptr();
       float32_t *pDst=output.ptr();


       arm_biquad_cascade_stereo_df2T_f32(&instStereo, pSrc, pDst, this->nbSamples);
        
    } 


    
    void BIQUADF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       
       std::vector<Testing::param_t>::iterator it = params.begin();
       this->numStages = *it++;
       this->nbSamples = *it;

       
       

       switch(id)
       {
           case TEST_BIQUAD_CASCADE_DF1_F32_1:
                  samples.reload(BIQUADF32::SAMPLES1_F32_ID,mgr,this->nbSamples);
                  output.create(this->nbSamples,BIQUADF32::OUT_SAMPLES_F32_ID,mgr);
                  coefs.reload(BIQUADF32::COEFS1_F32_ID,mgr,this->numStages * 5);
                  state.create(4*this->numStages,BIQUADF32::STATE_F32_ID,mgr);

                  arm_biquad_cascade_df1_init_f32(&instBiquadDf1,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());

           break;

           case TEST_BIQUAD_CASCADE_DF2T_F32_2:
                  samples.reload(BIQUADF32::SAMPLES1_F32_ID,mgr,this->nbSamples);
                  output.create(this->nbSamples,BIQUADF32::OUT_SAMPLES_F32_ID,mgr);
                  coefs.reload(BIQUADF32::COEFS1_F32_ID,mgr,this->numStages * 5);
                  state.create(2*this->numStages,BIQUADF32::STATE_F32_ID,mgr);

                  arm_biquad_cascade_df2T_init_f32(&instBiquadDf2T,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());
           break;

           case TEST_BIQUAD_CASCADE_STEREO_DF2T_F32_3:
                  samples.reload(BIQUADF32::SAMPLES1_F32_ID,mgr,2*this->nbSamples);
                  output.create(2*this->nbSamples,BIQUADF32::OUT_SAMPLES_F32_ID,mgr);
                  coefs.reload(BIQUADF32::COEFS1_F32_ID,mgr,this->numStages * 5);
                  state.create(4*this->numStages,BIQUADF32::STATE_F32_ID,mgr);

                  arm_biquad_cascade_stereo_df2T_init_f32(&instStereo,
                    this->numStages,
                    coefs.ptr(),
                    state.ptr());
           break;
       }
       
    }

    void BIQUADF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
