#include "TransformQ15.h"
#include "Error.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "Test.h"

#include <cstdio>

#define SNR_THRESHOLD 30

    void TransformQ15::test_cfft_q15()
    {
       const q15_t *inp = input.ptr();

       q15_t *outfftp = outputfft.ptr();

        memcpy(outfftp,inp,sizeof(q15_t)*input.nbSamples());
   
        arm_cfft_q15(
             this->instCfftQ15,
             outfftp,
             this->ifft,
             1);
          
        ASSERT_SNR(outputfft,ref,(float32_t)SNR_THRESHOLD);

       
        
    } 

    void TransformQ15::test_cifft_q15()
    {
       const q15_t *inp = input.ptr();

       q15_t *outfftp = outputfft.ptr();
       q15_t *refp = ref.ptr();

        memcpy(outfftp,inp,sizeof(q15_t)*input.nbSamples());
   
        arm_cfft_q15(
             this->instCfftQ15,
             outfftp,
             this->ifft,
             1);

        for(int i=0; i < outputfft.nbSamples();i++)
        {
          refp[i] >>= this->scaling;
        }
          
        ASSERT_SNR(outputfft,ref,(float32_t)SNR_THRESHOLD);
        

       
        
    } 

  
    void TransformQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {


       

       switch(id)
       {
          case TransformQ15::TEST_CFFT_Q15_1:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_16_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_16_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len16;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_19:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_16_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_16_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len16;

            this->ifft=1;
            this->scaling = 4;

          break;

          case TransformQ15::TEST_CFFT_Q15_2:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_32_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_32_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len32;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_20:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_32_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_32_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len32;

            this->ifft=1;
            this->scaling = 5;

          break;

          case TransformQ15::TEST_CFFT_Q15_3:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_64_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_64_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len64;

            this->ifft=0;


          break;

          case TransformQ15::TEST_CIFFT_Q15_21:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_64_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_64_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len64;

            this->ifft=1;
            this->scaling=6;

          break;

          case TransformQ15::TEST_CFFT_Q15_4:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_128_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_128_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len128;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_22:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_128_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_128_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len128;

            this->ifft=1;
            this->scaling=7;

          break;

          case TransformQ15::TEST_CFFT_Q15_5:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_256_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_256_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len256;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_23:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_256_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_256_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len256;

            this->ifft=1;
            this->scaling=8;

          break;

          case TransformQ15::TEST_CFFT_Q15_6:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_512_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_512_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len512;

            this->ifft=0;


          break;

          case TransformQ15::TEST_CIFFT_Q15_24:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_512_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_512_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len512;

            this->ifft=1;
            this->scaling=9;


          break;

          case TransformQ15::TEST_CFFT_Q15_7:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_1024_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_1024_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len1024;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_25:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_1024_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_1024_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len1024;

            this->ifft=1;
            this->scaling=10;

          break;

          case TransformQ15::TEST_CFFT_Q15_8:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_2048_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_2048_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len2048;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_26:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_2048_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_2048_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len2048;

            this->ifft=1;
            this->scaling=11;

          break;

          case TransformQ15::TEST_CFFT_Q15_9:

            input.reload(TransformQ15::INPUTS_CFFT_NOISY_4096_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_NOISY_4096_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len4096;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_27:

            input.reload(TransformQ15::INPUTS_CIFFT_NOISY_4096_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_NOISY_4096_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len4096;

            this->ifft=1;
            this->scaling=12;

          break;

          /* STEP FUNCTIONS */

          case TransformQ15::TEST_CFFT_Q15_10:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_16_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_16_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len16;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_28:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_16_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_16_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len16;

            this->ifft=1;
            this->scaling=4;

          break;

          case TransformQ15::TEST_CFFT_Q15_11:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_32_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_32_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len32;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_29:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_32_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_32_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len32;

            this->ifft=1;
            this->scaling=5;

          break;

          case TransformQ15::TEST_CFFT_Q15_12:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_64_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_64_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len64;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_30:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_64_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_64_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len64;

            this->ifft=1;
            this->scaling=6;

          break;

          case TransformQ15::TEST_CFFT_Q15_13:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_128_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_128_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len128;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_31:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_128_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_128_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len128;

            this->ifft=1;
            this->scaling=7;

          break;

          case TransformQ15::TEST_CFFT_Q15_14:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_256_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_256_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len256;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_32:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_256_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_256_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len256;

            this->ifft=1;
            this->scaling=8;

          break;

          case TransformQ15::TEST_CFFT_Q15_15:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_512_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_512_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len512;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_33:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_512_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_512_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len512;

            this->ifft=1;
            this->scaling=9;

          break;

          case TransformQ15::TEST_CFFT_Q15_16:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_1024_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_1024_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len1024;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_34:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_1024_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_1024_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len1024;

            this->ifft=1;
            this->scaling=10;

          break;

          case TransformQ15::TEST_CFFT_Q15_17:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_2048_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_2048_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len2048;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_35:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_2048_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_2048_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len2048;

            this->ifft=1;
            this->scaling=11;

          break;

          case TransformQ15::TEST_CFFT_Q15_18:

            input.reload(TransformQ15::INPUTS_CFFT_STEP_4096_Q15_ID,mgr);
            ref.reload(  TransformQ15::REF_CFFT_STEP_4096_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len4096;

            this->ifft=0;

          break;

          case TransformQ15::TEST_CIFFT_Q15_36:

            input.reload(TransformQ15::INPUTS_CIFFT_STEP_4096_Q15_ID,mgr);
            ref.reload(  TransformQ15::INPUTS_CFFT_STEP_4096_Q15_ID,mgr);

            instCfftQ15 = &arm_cfft_sR_q15_len4096;

            this->ifft=1;
            this->scaling=12;

          break;

       }

       outputfft.create(ref.nbSamples(),TransformQ15::OUTPUT_CFFT_Q15_ID,mgr);


    }

    void TransformQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        outputfft.dump(mgr);
    }
