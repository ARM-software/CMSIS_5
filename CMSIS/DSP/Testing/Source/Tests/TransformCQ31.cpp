#include "TransformCQ31.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "Test.h"

#define SNR_THRESHOLD 90

    void TransformCQ31::test_cfft_q31()
    {
       const q31_t *inp = input.ptr();

       q31_t *outfftp = outputfft.ptr();

        memcpy(outfftp,inp,sizeof(q31_t)*input.nbSamples());

        ASSERT_TRUE(status == ARM_MATH_SUCCESS);
   
        arm_cfft_q31(
             &(this->instCfftQ31),
             outfftp,
             this->ifft,
             1);
          
        ASSERT_SNR(outputfft,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_NEAR_EQ(outputfft,ref,(q31_t)32);
        ASSERT_EMPTY_TAIL(outputfft);
       
        
    } 

    void TransformCQ31::test_cifft_q31()
    {
       const q31_t *inp = input.ptr();

       q31_t *outfftp = outputfft.ptr();
       q31_t *refp = ref.ptr();

        memcpy(outfftp,inp,sizeof(q31_t)*input.nbSamples());

        ASSERT_TRUE(status == ARM_MATH_SUCCESS);
   
        arm_cfft_q31(
             &(this->instCfftQ31),
             outfftp,
             this->ifft,
             1);

        for(int i=0; i < outputfft.nbSamples();i++)
        {
          refp[i] >>= this->scaling;
        }
          
        ASSERT_SNR(outputfft,ref,(float32_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);

       
        
    } 

  
    void TransformCQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       

       switch(id)
       {
          case TransformCQ31::TEST_CFFT_Q31_1:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_16_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_16_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,16);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_19:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_16_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_16_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,16);

            this->ifft=1;
            this->scaling = 4;

          break;

          case TransformCQ31::TEST_CFFT_Q31_2:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_32_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_32_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,32);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_20:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_32_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_32_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,32);

            this->ifft=1;
            this->scaling = 5;

          break;

          case TransformCQ31::TEST_CFFT_Q31_3:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_64_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_64_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,64);

            this->ifft=0;


          break;

          case TransformCQ31::TEST_CIFFT_Q31_21:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_64_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_64_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,64);

            this->ifft=1;
            this->scaling=6;

          break;

          case TransformCQ31::TEST_CFFT_Q31_4:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_128_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_128_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,128);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_22:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_128_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_128_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,128);

            this->ifft=1;
            this->scaling=7;

          break;

          case TransformCQ31::TEST_CFFT_Q31_5:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_256_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_256_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,256);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_23:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_256_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_256_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,256);

            this->ifft=1;
            this->scaling=8;

          break;

          case TransformCQ31::TEST_CFFT_Q31_6:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_512_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_512_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,512);

            this->ifft=0;


          break;

          case TransformCQ31::TEST_CIFFT_Q31_24:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_512_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_512_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,512);

            this->ifft=1;
            this->scaling=9;


          break;

          case TransformCQ31::TEST_CFFT_Q31_7:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_1024_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_1024_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,1024);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_25:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_1024_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_1024_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,1024);

            this->ifft=1;
            this->scaling=10;

          break;

          case TransformCQ31::TEST_CFFT_Q31_8:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_2048_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_2048_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,2048);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_26:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_2048_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_2048_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,2048);

            this->ifft=1;
            this->scaling=11;

          break;

          case TransformCQ31::TEST_CFFT_Q31_9:

            input.reload(TransformCQ31::INPUTS_CFFT_NOISY_4096_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_NOISY_4096_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,4096);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_27:

            input.reload(TransformCQ31::INPUTS_CIFFT_NOISY_4096_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_NOISY_4096_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,4096);

            this->ifft=1;
            this->scaling=12;

          break;

          /* STEP FUNCTIONS */

          case TransformCQ31::TEST_CFFT_Q31_10:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_16_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_16_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,16);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_28:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_16_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_16_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,16);

            this->ifft=1;
            this->scaling=4;

          break;

          case TransformCQ31::TEST_CFFT_Q31_11:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_32_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_32_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,32);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_29:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_32_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_32_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,32);

            this->ifft=1;
            this->scaling=5;

          break;

          case TransformCQ31::TEST_CFFT_Q31_12:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_64_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_64_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,64);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_30:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_64_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_64_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,64);

            this->ifft=1;
            this->scaling=6;

          break;

          case TransformCQ31::TEST_CFFT_Q31_13:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_128_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_128_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,128);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_31:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_128_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_128_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,128);

            this->ifft=1;
            this->scaling=7;

          break;

          case TransformCQ31::TEST_CFFT_Q31_14:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_256_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_256_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,256);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_32:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_256_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_256_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,256);

            this->ifft=1;
            this->scaling=8;

          break;

          case TransformCQ31::TEST_CFFT_Q31_15:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_512_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_512_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,512);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_33:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_512_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_512_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,512);

            this->ifft=1;
            this->scaling=9;

          break;

          case TransformCQ31::TEST_CFFT_Q31_16:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_1024_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_1024_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,1024);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_34:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_1024_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_1024_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,1024);

            this->ifft=1;
            this->scaling=10;

          break;

          case TransformCQ31::TEST_CFFT_Q31_17:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_2048_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_2048_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,2048);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_35:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_2048_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_2048_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,2048);

            this->ifft=1;
            this->scaling=11;

          break;

          case TransformCQ31::TEST_CFFT_Q31_18:

            input.reload(TransformCQ31::INPUTS_CFFT_STEP_4096_Q31_ID,mgr);
            ref.reload(  TransformCQ31::REF_CFFT_STEP_4096_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,4096);

            this->ifft=0;

          break;

          case TransformCQ31::TEST_CIFFT_Q31_36:

            input.reload(TransformCQ31::INPUTS_CIFFT_STEP_4096_Q31_ID,mgr);
            ref.reload(  TransformCQ31::INPUTS_CFFT_STEP_4096_Q31_ID,mgr);

            status=arm_cfft_init_q31(&instCfftQ31,4096);

            this->ifft=1;
            this->scaling=12;

          break;

       }

       outputfft.create(ref.nbSamples(),TransformCQ31::OUTPUT_CFFT_Q31_ID,mgr);


    }

    void TransformCQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        outputfft.dump(mgr);
    }
