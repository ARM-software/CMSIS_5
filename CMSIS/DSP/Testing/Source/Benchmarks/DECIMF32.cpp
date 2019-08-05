#include "DECIMF32.h"
#include "Error.h"

   
    void DECIMF32::test_fir_decimate_f32()
    {
       
       const float32_t *pSrc=samples.ptr();
       float32_t *pDst=output.ptr();

       arm_fir_decimate_f32(&instDecim,pSrc,pDst,this->nbSamples);
 
        
    } 

 
   
   
    void DECIMF32::test_fir_interpolate_f32()
    {
       
       const float32_t *pSrc=samples.ptr();
       float32_t *pDst=output.ptr();

       arm_fir_interpolate_f32(&instInterpol,pSrc,pDst,this->nbSamples);
 
        
    } 
    
    void DECIMF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbTaps = *it++;
       this->nbSamples = *it++;
       

       samples.reload(DECIMF32::SAMPLES1_F32_ID,mgr,this->nbSamples);
       coefs.reload(DECIMF32::COEFS1_F32_ID,mgr,this->nbTaps);

       state.create(this->nbSamples + this->nbTaps - 1,DECIMF32::STATE_F32_ID,mgr);
       output.create(this->nbSamples,DECIMF32::OUT_SAMPLES_F32_ID,mgr);

       switch(id)
       {
           case TEST_FIR_DECIMATE_F32_1:
              this->decimationFactor = *it;
              arm_fir_decimate_init_f32(&instDecim,
                 this->nbTaps,
                 this->decimationFactor,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           break;


           case TEST_FIR_INTERPOLATE_F32_2:
              this->interpolationFactor = *it;
              arm_fir_interpolate_init_f32(&instInterpol,
                 this->interpolationFactor,
                 this->nbTaps,
                 coefs.ptr(),
                 state.ptr(),
                 this->nbSamples);
           break;

          
       }
       
    }

    void DECIMF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
