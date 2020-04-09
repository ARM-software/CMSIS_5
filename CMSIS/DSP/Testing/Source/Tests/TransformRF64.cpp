#include "TransformRF64.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "Test.h"


#define SNR_THRESHOLD 250



    void TransformRF64::test_rfft_f64()
    {
       float64_t *inp = input.ptr();

       float64_t *tmp = inputchanged.ptr();

       float64_t *outp = outputfft.ptr();

       memcpy(tmp,inp,sizeof(float64_t)*input.nbSamples());

        arm_rfft_fast_f64(
             &this->instRfftF64,
             tmp,
             outp,
             this->ifft);

        ASSERT_SNR(outputfft,ref,(float64_t)SNR_THRESHOLD);
        ASSERT_EMPTY_TAIL(outputfft);

    }


    void TransformRF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {




       switch(id)
       {

          case TransformRF64::TEST_RFFT_F64_1:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_32_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_32_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,32);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_17:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_64_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_64_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,64);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_2:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_64_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_64_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,64);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_18:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_64_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_64_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,64);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_3:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_128_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_128_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,128);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_19:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_128_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_128_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,128);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_4:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_256_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_256_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,256);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_20:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_256_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_256_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,256);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_5:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_512_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_512_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,512);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_21:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_512_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_512_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,512);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_6:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_1024_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_1024_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_22:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_1024_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_1024_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_7:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_2048_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_2048_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_23:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_2048_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_2048_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_8:

            input.reload(TransformRF64::INPUTS_RFFT_NOISY_4096_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_NOISY_4096_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_24:

            input.reload(TransformRF64::INPUTS_RIFFT_NOISY_4096_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_NOISY_4096_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          /* STEP FUNCTIONS */

          case TransformRF64::TEST_RFFT_F64_9:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_32_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_32_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,32);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_25:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_64_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_64_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,64);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_10:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_64_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_64_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,64);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_26:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_64_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_64_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,64);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_11:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_128_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_128_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,128);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);
            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_27:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_128_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_128_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,128);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_12:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_256_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_256_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,256);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_28:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_256_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_256_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,256);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_13:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_512_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_512_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,512);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_29:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_512_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_512_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,512);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_14:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_1024_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_1024_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_30:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_1024_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_1024_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,1024);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_15:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_2048_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_2048_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_31:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_2048_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_2048_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,2048);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;

          case TransformRF64::TEST_RFFT_F64_16:

            input.reload(TransformRF64::INPUTS_RFFT_STEP_4096_F64_ID,mgr);
            ref.reload(  TransformRF64::REF_RFFT_STEP_4096_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=0;

          break;

          case TransformRF64::TEST_RFFT_F64_32:

            input.reload(TransformRF64::INPUTS_RIFFT_STEP_4096_F64_ID,mgr);
            ref.reload(  TransformRF64::INPUTS_RFFT_STEP_4096_F64_ID,mgr);

            arm_rfft_fast_init_f64(&this->instRfftF64 ,4096);

            inputchanged.create(input.nbSamples(),TransformRF64::TEMP_F64_ID,mgr);

            this->ifft=1;

          break;



       }


      outputfft.create(ref.nbSamples(),TransformRF64::OUTPUT_RFFT_F64_ID,mgr);


    }

    void TransformRF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        outputfft.dump(mgr);
    }
