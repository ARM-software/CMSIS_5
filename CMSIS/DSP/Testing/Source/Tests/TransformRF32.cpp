#include "TransformRF32.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "Test.h"


#define SNR_THRESHOLD 120



    void TransformRF32::test_rfft_f32()
    {
       float32_t *inp = input.ptr();

       float32_t *tmp = inputchanged.ptr();

       float32_t *outp = outputfft.ptr();

       memcpy(tmp,inp,sizeof(float32_t)*input.nbSamples());
   
        arm_rfft_fast_f32(
             &this->instRfftF32,
             tmp,
             outp,
             this->ifft);
          
        ASSERT_SNR(outputfft,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);
        
    } 

  
    void TransformRF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       

       switch(id)
       {

          case TransformRF32::TEST_RFFT_F32_1:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_32_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_32_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,32);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_17:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_32_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_32_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,32);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_2:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_64_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_64_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,64);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_18:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_64_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_64_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,64);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_3:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_128_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_128_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,128);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_19:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_128_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_128_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,128);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_4:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_256_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_256_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,256);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_20:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_256_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_256_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,256);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_5:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_512_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_512_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,512);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_21:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_512_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_512_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,512);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_6:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_1024_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_1024_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_22:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_1024_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_1024_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_7:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_2048_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_2048_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_23:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_2048_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_2048_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_8:

            input.reload(TransformRF32::INPUTS_RFFT_NOISY_4096_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_NOISY_4096_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_24:

            input.reload(TransformRF32::INPUTS_RIFFT_NOISY_4096_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_NOISY_4096_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          /* STEP FUNCTIONS */

          case TransformRF32::TEST_RFFT_F32_9:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_32_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_32_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,32);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_25:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_32_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_32_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,32);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_10:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_64_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_64_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,64);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_26:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_64_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_64_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,64);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_11:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_128_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_128_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,128);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);
            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_27:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_128_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_128_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,128);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_12:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_256_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_256_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,256);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_28:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_256_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_256_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,256);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_13:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_512_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_512_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,512);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_29:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_512_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_512_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,512);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_14:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_1024_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_1024_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_30:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_1024_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_1024_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_15:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_2048_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_2048_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_31:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_2048_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_2048_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF32::TEST_RFFT_F32_16:

            input.reload(TransformRF32::INPUTS_RFFT_STEP_4096_F32_ID,mgr);
            ref.reload(  TransformRF32::REF_RFFT_STEP_4096_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF32::TEST_RFFT_F32_32:

            input.reload(TransformRF32::INPUTS_RIFFT_STEP_4096_F32_ID,mgr);
            ref.reload(  TransformRF32::INPUTS_RFFT_STEP_4096_F32_ID,mgr);

            arm_rfft_fast_init_f32(&this->instRfftF32 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF32::TEMP_F32_ID,mgr);

            this->ifft=1;

          break;



       }

       
      outputfft.create(ref.nbSamples(),TransformRF32::OUTPUT_RFFT_F32_ID,mgr);
       

    }

    void TransformRF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        outputfft.dump(mgr);
    }
