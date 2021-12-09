#include "TransformCF64.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define SNR_THRESHOLD 250

    void TransformCF64::test_cfft_f64()
    {
       const float64_t *inp = input.ptr();

       float64_t *outfftp = outputfft.ptr();

        memcpy(outfftp,inp,sizeof(float64_t)*input.nbSamples());
   
        arm_cfft_f64(
             &(this->varInstCfftF64),
             outfftp,
             this->ifft,
             1);

          
        ASSERT_SNR(outputfft,ref,(float64_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);
        
    } 

  
    void TransformCF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       (void)paramsArgs;

       switch(id)
       {
          case TransformCF64::TEST_CFFT_F64_1:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_16_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_16_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,16);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_19:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_16_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_16_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,16);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_2:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_32_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_32_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,32);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_20:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_32_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_32_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,32);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_3:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_64_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_64_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,64);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_21:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_64_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_64_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,64);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_4:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_128_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_128_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,128);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_22:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_128_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_128_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,128);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_5:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_256_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_256_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,256);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_23:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_256_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_256_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,256);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_6:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_512_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_512_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,512);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_24:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_512_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_512_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,512);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_7:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_1024_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_1024_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,1024);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_25:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_1024_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_1024_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,1024);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_8:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_2048_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_2048_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,2048);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_26:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_2048_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_2048_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,2048);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_9:

            input.reload(TransformCF64::INPUTS_CFFT_NOISY_4096_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_NOISY_4096_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,4096);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_27:

            input.reload(TransformCF64::INPUTS_CIFFT_NOISY_4096_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_NOISY_4096_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,4096);

            this->ifft=1;

          break;

          /* STEP FUNCTIONS */

          case TransformCF64::TEST_CFFT_F64_10:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_16_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_16_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,16);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_28:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_16_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_16_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,16);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_11:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_32_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_32_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,32);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_29:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_32_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_32_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,32);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_12:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_64_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_64_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,64);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_30:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_64_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_64_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,64);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_13:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_128_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_128_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,128);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_31:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_128_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_128_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,128);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_14:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_256_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_256_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,256);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_32:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_256_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_256_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,256);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_15:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_512_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_512_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,512);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_33:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_512_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_512_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,512);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_16:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_1024_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_1024_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,1024);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_34:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_1024_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_1024_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,1024);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_17:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_2048_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_2048_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,2048);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_35:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_2048_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_2048_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,2048);

            this->ifft=1;

          break;

          case TransformCF64::TEST_CFFT_F64_18:

            input.reload(TransformCF64::INPUTS_CFFT_STEP_4096_F64_ID,mgr);
            ref.reload(  TransformCF64::REF_CFFT_STEP_4096_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,4096);

            this->ifft=0;

          break;

          case TransformCF64::TEST_CFFT_F64_36:

            input.reload(TransformCF64::INPUTS_CIFFT_STEP_4096_F64_ID,mgr);
            ref.reload(  TransformCF64::INPUTS_CFFT_STEP_4096_F64_ID,mgr);

            status=arm_cfft_init_f64(&varInstCfftF64,4096);

            this->ifft=1;

          break;

       }

       outputfft.create(ref.nbSamples(),TransformCF64::OUTPUT_CFFT_F64_ID,mgr);


    }

    void TransformCF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        outputfft.dump(mgr);
    }
