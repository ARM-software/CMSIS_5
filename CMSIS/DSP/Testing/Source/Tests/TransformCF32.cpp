#include "TransformCF32.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 120

    void TransformCF32::test_cfft_f32()
    {
       const float32_t *inp = input.ptr();

       float32_t *outfftp = outputfft.ptr();

        memcpy(outfftp,inp,sizeof(float32_t)*input.nbSamples());

        ASSERT_TRUE(status == ARM_MATH_SUCCESS);
   
        arm_cfft_f32(
             &(this->varInstCfftF32),
             outfftp,
             this->ifft,
             1);
       

          
        ASSERT_SNR(outputfft,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);


        
    } 

    void TransformCF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       (void)paramsArgs;
       switch(id)
       {
          case TransformCF32::TEST_CFFT_F32_1:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_16_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_16_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,16);

            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_19:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_16_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_16_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,16);

            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_2:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_32_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_32_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,32);

            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_20:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_32_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_32_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,32);

            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_3:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_64_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_64_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,64);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_21:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_64_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_64_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,64);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_4:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_128_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_128_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,128);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_22:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_128_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_128_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,128);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_5:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_256_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_256_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,256);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_23:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_256_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_256_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,256);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_6:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_512_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_512_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,512);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_24:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_512_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_512_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,512);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_7:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_1024_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_1024_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,1024);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_25:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_1024_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_1024_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,1024);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_8:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_2048_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_2048_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,2048);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_26:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_2048_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_2048_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,2048);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_9:

            input.reload(TransformCF32::INPUTS_CFFT_NOISY_4096_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_NOISY_4096_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,4096);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_27:

            input.reload(TransformCF32::INPUTS_CIFFT_NOISY_4096_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_NOISY_4096_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,4096);


            this->ifft=1;

          break;

          /* STEP FUNCTIONS */

          case TransformCF32::TEST_CFFT_F32_10:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_16_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_16_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,16);

            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_28:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_16_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_16_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,16);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_11:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_32_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_32_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,32);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_29:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_32_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_32_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,32);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_12:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_64_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_64_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,64);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_30:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_64_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_64_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,64);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_13:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_128_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_128_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,128);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_31:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_128_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_128_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,128);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_14:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_256_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_256_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,256);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_32:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_256_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_256_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,256);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_15:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_512_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_512_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,512);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_33:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_512_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_512_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,512);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_16:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_1024_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_1024_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,1024);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_34:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_1024_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_1024_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,1024);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_17:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_2048_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_2048_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,2048);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_35:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_2048_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_2048_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,2048);


            this->ifft=1;

          break;

          case TransformCF32::TEST_CFFT_F32_18:

            input.reload(TransformCF32::INPUTS_CFFT_STEP_4096_F32_ID,mgr);
            ref.reload(  TransformCF32::REF_CFFT_STEP_4096_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,4096);


            this->ifft=0;

          break;

          case TransformCF32::TEST_CFFT_F32_36:

            input.reload(TransformCF32::INPUTS_CIFFT_STEP_4096_F32_ID,mgr);
            ref.reload(  TransformCF32::INPUTS_CFFT_STEP_4096_F32_ID,mgr);

            status=arm_cfft_init_f32(&varInstCfftF32,4096);


            this->ifft=1;

          break;



       }
       outputfft.create(ref.nbSamples(),TransformCF32::OUTPUT_CFFT_F32_ID,mgr);
       

    }

    void TransformCF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        outputfft.dump(mgr);
    }
