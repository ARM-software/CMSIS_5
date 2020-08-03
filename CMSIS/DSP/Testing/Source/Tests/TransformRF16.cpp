#include "TransformRF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define SNR_THRESHOLD 58



    void TransformRF16::test_rfft_f16()
    {
       float16_t *inp = input.ptr();

       float16_t *tmp = inputchanged.ptr();

       float16_t *outp = outputfft.ptr();

       memcpy(tmp,inp,sizeof(float16_t)*input.nbSamples());
   
        arm_rfft_fast_f16(
             &this->instRfftF16,
             tmp,
             outp,
             this->ifft);
          
        ASSERT_SNR(outputfft,ref,(float16_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);
        
    } 

  
    void TransformRF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       (void)paramsArgs;

       switch(id)
       {

          case TransformRF16::TEST_RFFT_F16_1:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_32_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_32_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,32);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_17:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_32_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_32_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,32);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_2:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_64_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_64_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,64);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_18:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_64_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_64_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,64);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_3:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_128_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_128_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,128);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_19:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_128_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_128_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,128);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_4:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_256_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_256_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,256);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_20:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_256_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_256_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,256);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_5:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_512_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_512_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,512);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_21:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_512_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_512_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,512);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_6:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_1024_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_1024_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_22:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_1024_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_1024_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_7:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_2048_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_2048_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_23:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_2048_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_2048_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_8:

            input.reload(TransformRF16::INPUTS_RFFT_NOISY_4096_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_NOISY_4096_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_24:

            input.reload(TransformRF16::INPUTS_RIFFT_NOISY_4096_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_NOISY_4096_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          /* STEP FUNCTIONS */

          case TransformRF16::TEST_RFFT_F16_9:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_32_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_32_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,32);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_25:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_32_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_32_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,32);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_10:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_64_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_64_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,64);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_26:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_64_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_64_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,64);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_11:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_128_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_128_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,128);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);
            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_27:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_128_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_128_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,128);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_12:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_256_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_256_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,256);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_28:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_256_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_256_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,256);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_13:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_512_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_512_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,512);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_29:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_512_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_512_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,512);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_14:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_1024_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_1024_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_30:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_1024_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_1024_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_15:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_2048_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_2048_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_31:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_2048_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_2048_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF16::TEST_RFFT_F16_16:

            input.reload(TransformRF16::INPUTS_RFFT_STEP_4096_F16_ID,mgr);
            ref.reload(  TransformRF16::REF_RFFT_STEP_4096_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF16::TEST_RFFT_F16_32:

            input.reload(TransformRF16::INPUTS_RIFFT_STEP_4096_F16_ID,mgr);
            ref.reload(  TransformRF16::INPUTS_RFFT_STEP_4096_F16_ID,mgr);

            arm_rfft_fast_init_f16(&this->instRfftF16 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF16::TEMP_F16_ID,mgr);

            this->ifft=1;

          break;



       }

       
      outputfft.create(ref.nbSamples(),TransformRF16::OUTPUT_RFFT_F16_ID,mgr);
       

    }

    void TransformRF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        outputfft.dump(mgr);
    }
