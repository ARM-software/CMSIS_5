#include "TransformQ31.h"
#include "Error.h"

    void TransformQ31::test_cfft_q31()
    { 
       arm_cfft_q31(&this->cfftInstance, this->pDst, this->ifft,this->bitRev);
    } 

    void TransformQ31::test_rfft_q31()
    { 
       arm_rfft_q31(&this->rfftInstance, this->pSrc, this->pDst);
    } 

    void TransformQ31::test_dct4_q31()
    { 
        arm_dct4_q31(
          &this->dct4Instance,
          this->pState,
          this->pDst);
    } 

    void TransformQ31::test_cfft_radix4_q31()
    { 
       arm_cfft_radix4_q31(&this->cfftRadix4Instance,this->pDst);
    } 

    void TransformQ31::test_cfft_radix2_q31()
    { 
       arm_cfft_radix2_q31(&this->cfftRadix2Instance,this->pDst);
    } 


    void TransformQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {

       float32_t normalize;

       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it++;
       this->ifft = *it++;
       this->bitRev = *it;
      
       switch(id)
       {
          case TEST_CFFT_Q31_1:
            samples.reload(TransformQ31::INPUTC_Q31_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformQ31::OUT_Q31_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            arm_cfft_init_q31(&this->cfftInstance,this->nbSamples);
            memcpy(this->pDst,this->pSrc,2*sizeof(q31_t)*this->nbSamples);
          break;

          case TEST_RFFT_Q31_2:
            samples.reload(TransformQ31::INPUTR_Q31_ID,mgr,this->nbSamples);
            output.create(this->nbSamples,TransformQ31::OUT_Q31_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            arm_rfft_init_q31(&this->rfftInstance, this->nbSamples,this->ifft,this->bitRev);
          break;

          case TEST_DCT4_Q31_3:
            samples.reload(TransformQ31::INPUTR_Q31_ID,mgr,this->nbSamples);
            output.create(this->nbSamples,TransformQ31::OUT_Q31_ID,mgr);
            state.create(2*this->nbSamples,TransformQ31::STATE_Q31_ID,mgr);
            

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();
            this->pState=state.ptr();

            normalize = sqrt((2.0f/(float32_t)this->nbSamples));      

            memcpy(this->pDst,this->pSrc,sizeof(q31_t)*this->nbSamples); 

            arm_dct4_init_q31(
               &this->dct4Instance,
               &this->rfftInstance,
               &this->cfftRadix4Instance,
               this->nbSamples,
               this->nbSamples/2,
               normalize);
          break;

          case TEST_CFFT_RADIX4_Q31_4:
            samples.reload(TransformQ31::INPUTC_Q31_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformQ31::OUT_Q31_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(q31_t)*this->nbSamples);

            arm_cfft_radix4_init_q31(&this->cfftRadix4Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);

          break;

          case TEST_CFFT_RADIX2_Q31_5:
            samples.reload(TransformQ31::INPUTC_Q31_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformQ31::OUT_Q31_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(q31_t)*this->nbSamples);

            arm_cfft_radix2_init_q31(&this->cfftRadix2Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);
          break;

       }


       

    }

    void TransformQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
