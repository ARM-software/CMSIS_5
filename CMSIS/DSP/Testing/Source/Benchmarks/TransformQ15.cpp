#include "TransformQ15.h"
#include "Error.h"


    void TransformQ15::test_cfft_q15()
    { 
       arm_cfft_q15(&this->cfftInstance, this->pDst, this->ifft,this->bitRev);
    } 

    void TransformQ15::test_rfft_q15()
    { 
       arm_rfft_q15(&this->rfftInstance, this->pSrc, this->pDst);
    } 

    void TransformQ15::test_dct4_q15()
    { 
        arm_dct4_q15(
          &this->dct4Instance,
          this->pState,
          this->pDst);
    } 

    void TransformQ15::test_cfft_radix4_q15()
    { 
       arm_cfft_radix4_q15(&this->cfftRadix4Instance,this->pDst);
    } 

    void TransformQ15::test_cfft_radix2_q15()
    { 
       arm_cfft_radix2_q15(&this->cfftRadix2Instance,this->pDst);
    } 


    void TransformQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {

       float32_t normalize;

       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it++;
       this->ifft = *it++;
       this->bitRev = *it;
      
       switch(id)
       {
          case TEST_CFFT_Q15_1:
            samples.reload(TransformQ15::INPUTC_Q15_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformQ15::OUT_Q15_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            arm_cfft_init_q15(&this->cfftInstance,this->nbSamples);
            memcpy(this->pDst,this->pSrc,2*sizeof(q15_t)*this->nbSamples);
          break;

          case TEST_RFFT_Q15_2:
            samples.reload(TransformQ15::INPUTR_Q15_ID,mgr,this->nbSamples);
            output.create(this->nbSamples,TransformQ15::OUT_Q15_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            arm_rfft_init_q15(&this->rfftInstance, this->nbSamples, this->ifft, this->bitRev);
          break;

          case TEST_DCT4_Q15_3:
            samples.reload(TransformQ15::INPUTR_Q15_ID,mgr,this->nbSamples);
            output.create(this->nbSamples,TransformQ15::OUT_Q15_ID,mgr);
            state.create(2*this->nbSamples,TransformQ15::STATE_Q15_ID,mgr);
            

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();
            this->pState=state.ptr();

            normalize = sqrt((2.0f/(float32_t)this->nbSamples));      

            memcpy(this->pDst,this->pSrc,sizeof(q15_t)*this->nbSamples); 

            arm_dct4_init_q15(
               &this->dct4Instance,
               &this->rfftInstance,
               &this->cfftRadix4Instance,
               this->nbSamples,
               this->nbSamples/2,
               normalize);
          break;

          case TEST_CFFT_RADIX4_Q15_4:
            samples.reload(TransformQ15::INPUTC_Q15_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformQ15::OUT_Q15_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(q15_t)*this->nbSamples);

            arm_cfft_radix4_init_q15(&this->cfftRadix4Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);

          break;

          case TEST_CFFT_RADIX2_Q15_5:
            samples.reload(TransformQ15::INPUTC_Q15_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformQ15::OUT_Q15_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(q15_t)*this->nbSamples);

            arm_cfft_radix2_init_q15(&this->cfftRadix2Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);
          break;

       }


       

    }

    void TransformQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
