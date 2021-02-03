#include "TransformF16.h"
#include "Error.h"

    void TransformF16::test_cfft_f16()
    { 
       arm_cfft_f16(&(this->cfftInstance), this->pDst, this->ifft,this->bitRev);
    } 

    void TransformF16::test_rfft_f16()
    { 
       arm_rfft_fast_f16(&this->rfftFastInstance, this->pTmp, this->pDst, this->ifft);
    } 

    void TransformF16::test_cfft_radix4_f16()
    { 
       arm_cfft_radix4_f16(&this->cfftRadix4Instance,this->pDst);
    } 

    void TransformF16::test_cfft_radix2_f16()
    { 
       arm_cfft_radix2_f16(&this->cfftRadix2Instance,this->pDst);
    } 


    void TransformF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it++;
       this->ifft = *it++;
       this->bitRev = *it;
      
       switch(id)
       {
          case TEST_CFFT_F16_1:
            samples.reload(TransformF16::INPUTC_F16_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF16::OUT_F16_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            status=arm_cfft_init_f16(&cfftInstance,this->nbSamples);
            memcpy(this->pDst,this->pSrc,2*sizeof(float16_t)*this->nbSamples);
          break;

          case TEST_RFFT_F16_2:
          {
            // Factor 2 for irfft
            samples.reload(TransformF16::INPUTR_F16_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF16::OUT_F16_ID,mgr);
            tmp.create(2*this->nbSamples,TransformF16::TMP_F16_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();
            this->pTmp=tmp.ptr();

            memcpy(this->pTmp,this->pSrc,sizeof(float16_t)*this->nbSamples); 


            arm_rfft_fast_init_f16(&this->rfftFastInstance, this->nbSamples);
          }
          break;

          case TEST_CFFT_RADIX4_F16_3:
            samples.reload(TransformF16::INPUTC_F16_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF16::OUT_F16_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(float16_t)*this->nbSamples);

            arm_cfft_radix4_init_f16(&this->cfftRadix4Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);

          break;

          case TEST_CFFT_RADIX2_F16_4:
            samples.reload(TransformF16::INPUTC_F16_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF16::OUT_F16_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(float16_t)*this->nbSamples);

            arm_cfft_radix2_init_f16(&this->cfftRadix2Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);
          break;

       }


       

    }

    void TransformF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
      (void)id;
      (void)mgr;
    }
