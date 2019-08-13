#include "SupportBarF32.h"
#include "Error.h"

   
    void SupportBarF32::test_barycenter_f32()
    {
      arm_barycenter_f32(this->inp, this->coefsp,
            this->outp, 
            this->nbVectors, 
            this->vecDim);
    } 

   

    void SupportBarF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbVectors = *it++;
       this->vecDim = *it;

       switch(id)
       {
           case TEST_BARYCENTER_F32_1:
              input.reload(SupportBarF32::SAMPLES_F32_ID,mgr,this->nbVectors*this->vecDim);
              coefs.reload(SupportBarF32::COEFS_F32_ID,mgr,this->nbVectors);
              output.create(this->vecDim,SupportBarF32::OUT_SAMPLES_F32_ID,mgr);

              this->inp = input.ptr();
              this->coefsp = coefs.ptr();
              this->outp = output.ptr();
           break;

       }

       
    }

    void SupportBarF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
