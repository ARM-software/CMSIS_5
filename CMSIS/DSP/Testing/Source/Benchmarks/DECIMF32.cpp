#include "DECIMF32.h"
#include "Error.h"

   
    void DECIMF32::test_fir_decimate_f32()
    {
       arm_fir_decimate_f32(&instDecim,this->pSrc,this->pDst,this->nbSamples);
    } 

 
   
   
    void DECIMF32::test_fir_interpolate_f32()
    {
       arm_fir_interpolate_f32(&instInterpol,this->pSrc,this->pDst,this->nbSamples);
    } 
    
    void DECIMF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it++;
       

       samples.reload(DECIMF32::SAMPLES1_F32_ID,mgr,this->nbSamples);
       coefs.reload(DECIMF32::COEFS1_F32_ID,mgr,this->nbTaps);

       output.create(this->nbSamples,DECIMF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
           case TEST_FIR_DECIMATE_F32_1:
              this->decimationFactor = *it;

              state.create(this->nbSamples + this->nbTaps - 1,DECIMF32::STATE_F32_ID,mgr);

              arm_fir_decimate_init_f32(&instDecim,
                 this->nbTaps,
                 this->decimationFactor,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           break;


           case TEST_FIR_INTERPOLATE_F32_2:
           {
              this->interpolationFactor = *it;
              int phase = this->nbTaps / this->interpolationFactor;

              state.create(this->nbSamples + phase - 1,DECIMF32::STATE_F32_ID,mgr);

              arm_fir_interpolate_init_f32(&instInterpol,
                 this->interpolationFactor,
                 this->nbTaps,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           }
           break;

          
       }

       this->pSrc=samples.ptr();
       this->pDst=output.ptr();
       
    }

    void DECIMF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
