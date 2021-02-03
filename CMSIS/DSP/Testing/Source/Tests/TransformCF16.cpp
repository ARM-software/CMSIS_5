#include "TransformCF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 58

    void TransformCF16::test_cfft_f16()
    {
       const float16_t *inp = input.ptr();

       float16_t *outfftp = outputfft.ptr();

        memcpy(outfftp,inp,sizeof(float16_t)*input.nbSamples());

        ASSERT_TRUE(status == ARM_MATH_SUCCESS);
   
        arm_cfft_f16(
             &(this->varInstCfftF16),
             outfftp,
             this->ifft,
             1);
       

          
        ASSERT_SNR(outputfft,ref,(float16_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);


        
    } 

    void TransformCF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       (void)paramsArgs;
       
       switch(id)
       {
          case TransformCF16::TEST_CFFT_F16_1:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_16_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_16_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,16);

            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_19:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_16_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_16_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,16);

            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_2:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_32_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_32_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,32);

            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_20:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_32_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_32_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,32);

            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_3:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_64_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_64_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,64);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_21:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_64_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_64_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,64);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_4:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_128_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_128_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,128);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_22:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_128_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_128_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,128);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_5:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_256_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_256_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,256);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_23:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_256_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_256_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,256);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_6:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_512_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_512_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,512);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_24:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_512_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_512_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,512);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_7:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_1024_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_1024_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,1024);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_25:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_1024_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_1024_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,1024);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_8:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_2048_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_2048_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,2048);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_26:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_2048_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_2048_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,2048);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_9:

            input.reload(TransformCF16::INPUTS_CFFT_NOISY_4096_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_NOISY_4096_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,4096);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_27:

            input.reload(TransformCF16::INPUTS_CIFFT_NOISY_4096_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_NOISY_4096_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,4096);


            this->ifft=1;

          break;

          /* STEP FUNCTIONS */

          case TransformCF16::TEST_CFFT_F16_10:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_16_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_16_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,16);

            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_28:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_16_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_16_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,16);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_11:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_32_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_32_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,32);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_29:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_32_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_32_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,32);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_12:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_64_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_64_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,64);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_30:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_64_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_64_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,64);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_13:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_128_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_128_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,128);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_31:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_128_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_128_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,128);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_14:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_256_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_256_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,256);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_32:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_256_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_256_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,256);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_15:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_512_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_512_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,512);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_33:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_512_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_512_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,512);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_16:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_1024_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_1024_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,1024);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_34:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_1024_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_1024_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,1024);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_17:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_2048_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_2048_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,2048);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_35:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_2048_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_2048_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,2048);


            this->ifft=1;

          break;

          case TransformCF16::TEST_CFFT_F16_18:

            input.reload(TransformCF16::INPUTS_CFFT_STEP_4096_F16_ID,mgr);
            ref.reload(  TransformCF16::REF_CFFT_STEP_4096_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,4096);


            this->ifft=0;

          break;

          case TransformCF16::TEST_CFFT_F16_36:

            input.reload(TransformCF16::INPUTS_CIFFT_STEP_4096_F16_ID,mgr);
            ref.reload(  TransformCF16::INPUTS_CFFT_STEP_4096_F16_ID,mgr);

            status=arm_cfft_init_f16(&varInstCfftF16,4096);


            this->ifft=1;

          break;



       }
        outputfft.create(ref.nbSamples(),TransformCF16::OUTPUT_CFFT_F16_ID,mgr);
       

    }

    void TransformCF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        outputfft.dump(mgr);
    }
