#include "SupportBarF16.h"
#include "Error.h"

   
    void SupportBarF16::test_barycenter_f16()
    {
      arm_barycenter_f16(this->inp, this->coefsp,
            this->outp, 
            this->nbVectors, 
            this->vecDim);
    } 

   

    void SupportBarF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbVectors = *it++;
       this->vecDim = *it;

       switch(id)
       {
           case TEST_BARYCENTER_F16_1:
              input.reload(SupportBarF16::SAMPLES_F16_ID,mgr,this->nbVectors*this->vecDim);
              coefs.reload(SupportBarF16::COEFS_F16_ID,mgr,this->nbVectors);
              output.create(this->vecDim,SupportBarF16::OUT_SAMPLES_F16_ID,mgr);

              this->inp = input.ptr();
              this->coefsp = coefs.ptr();
              this->outp = output.ptr();
           break;

       }

       
    }

    void SupportBarF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
      
    }
