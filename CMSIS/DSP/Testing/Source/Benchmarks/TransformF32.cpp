#include "TransformF32.h"
#include "Error.h"
#include "arm_math.h"
#include "arm_const_structs.h"

const arm_cfft_instance_f32 *arm_cfft_get_instance_f32(uint16_t fftLen)
{
switch (fftLen) {
     case 16:
       return(&arm_cfft_sR_f32_len16);
     case 32:
       return(&arm_cfft_sR_f32_len32);
     case 64:
       return(&arm_cfft_sR_f32_len64);
     case 128:
       return(&arm_cfft_sR_f32_len128);
     case 256:
       return(&arm_cfft_sR_f32_len256);
     case 512:
       return(&arm_cfft_sR_f32_len512);
     case 1024:
       return(&arm_cfft_sR_f32_len1024);
     case 2048:
       return(&arm_cfft_sR_f32_len2048);
     case 4096:
       return(&arm_cfft_sR_f32_len4096);
   }
   return(NULL);
}
   
    void TransformF32::test_cfft_f32()
    { 
       arm_cfft_f32(this->cfftInstance, this->pDst, this->ifft,this->bitRev);
    } 

    void TransformF32::test_rfft_f32()
    { 
       arm_rfft_fast_f32(&this->rfftFastInstance, this->pSrc, this->pDst, this->ifft);
    } 

    void TransformF32::test_dct4_f32()
    { 
        arm_dct4_f32(
          &this->dct4Instance,
          this->pState,
          this->pDst);
    } 

    void TransformF32::test_cfft_radix4_f32()
    { 
       arm_cfft_radix4_f32(&this->cfftRadix4Instance,this->pDst);
    } 

    void TransformF32::test_cfft_radix2_f32()
    { 
       arm_cfft_radix2_f32(&this->cfftRadix2Instance,this->pDst);
    } 


    void TransformF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {

       float32_t normalize;

       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nbSamples = *it++;
       this->ifft = *it++;
       this->bitRev = *it;
      
       switch(id)
       {
          case TEST_CFFT_F32_1:
            samples.reload(TransformF32::INPUTC_F32_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF32::OUT_F32_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            this->cfftInstance=arm_cfft_get_instance_f32(this->nbSamples);
            memcpy(this->pDst,this->pSrc,2*sizeof(float32_t)*this->nbSamples);
          break;

          case TEST_RFFT_F32_2:
            samples.reload(TransformF32::INPUTR_F32_ID,mgr,this->nbSamples);
            output.create(this->nbSamples,TransformF32::OUT_F32_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            arm_rfft_fast_init_f32(&this->rfftFastInstance, this->nbSamples);
          break;

          case TEST_DCT4_F32_3:
            samples.reload(TransformF32::INPUTR_F32_ID,mgr,this->nbSamples);
            output.create(this->nbSamples,TransformF32::OUT_F32_ID,mgr);
            state.create(2*this->nbSamples,TransformF32::STATE_F32_ID,mgr);
            

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();
            this->pState=state.ptr();

            normalize = sqrt((2.0f/(float32_t)this->nbSamples));      

            memcpy(this->pDst,this->pSrc,sizeof(float32_t)*this->nbSamples); 

            arm_dct4_init_f32(
               &this->dct4Instance,
               &this->rfftInstance,
               &this->cfftRadix4Instance,
               this->nbSamples,
               this->nbSamples/2,
               normalize);
          break;

          case TEST_CFFT_RADIX4_F32_4:
            samples.reload(TransformF32::INPUTC_F32_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF32::OUT_F32_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(float32_t)*this->nbSamples);

            arm_cfft_radix4_init_f32(&this->cfftRadix4Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);

          break;

          case TEST_CFFT_RADIX2_F32_5:
            samples.reload(TransformF32::INPUTC_F32_ID,mgr,2*this->nbSamples);
            output.create(2*this->nbSamples,TransformF32::OUT_F32_ID,mgr);

            this->pSrc=samples.ptr();
            this->pDst=output.ptr();

            
            memcpy(this->pDst,this->pSrc,2*sizeof(float32_t)*this->nbSamples);

            arm_cfft_radix2_init_f32(&this->cfftRadix2Instance,
                this->nbSamples,
                this->ifft,
                this->bitRev);
          break;

       }


       

    }

    void TransformF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
