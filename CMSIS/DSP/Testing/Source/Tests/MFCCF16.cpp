#include "MFCCF16.h"
#include <stdio.h>
#include "Error.h"

#include "mfccdata_f16.h"

#define SNR_THRESHOLD 45

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (2.0e-2)
#define ABS_ERROR (2.0e-2)


    void MFCCF16::test_mfcc_f16()
    {
        const float16_t *inp1=input1.ptr(); 
        float16_t *tmpinp=tmpin.ptr(); 
        float16_t *outp=output.ptr();
        float16_t *tmpp=tmp.ptr();


        memcpy((void*)tmpinp,(void*)inp1,sizeof(float16_t)*this->fftLen);
        arm_mfcc_f16(&mfcc,tmpinp,outp,tmpp);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float16_t)SNR_THRESHOLD);

        ASSERT_CLOSE_ERROR(output,ref,ABS_ERROR,REL_ERROR);

    } 

   
    void MFCCF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case MFCCF16::TEST_MFCC_F16_1:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCF16::REF_MFCC_NOISE_256_F16_ID,mgr,nb);
            input1.reload(MFCCF16::INPUTS_MFCC_NOISE_256_F16_ID,mgr,nb);
            arm_mfcc_init_f16(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_f16,
                    mfcc_filter_pos_config3_f16,mfcc_filter_len_config3_f16,
                    mfcc_filter_coefs_config3_f16,
                    mfcc_window_coefs_config3_f16);
            tmp.create(2*nb,MFCCF16::TMP_MFCC_F16_ID,mgr);
            tmpin.create(nb,MFCCF16::TMPIN_MFCC_F16_ID,mgr);
          }
          break;

        case MFCCF16::TEST_MFCC_F16_2:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCF16::REF_MFCC_NOISE_512_F16_ID,mgr,nb);
            input1.reload(MFCCF16::INPUTS_MFCC_NOISE_512_F16_ID,mgr,nb);
            arm_mfcc_init_f16(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f16,
                      mfcc_filter_pos_config2_f16,mfcc_filter_len_config2_f16,
                      mfcc_filter_coefs_config2_f16,
                      mfcc_window_coefs_config2_f16);
            tmp.create(2*nb,MFCCF16::TMP_MFCC_F16_ID,mgr);
            tmpin.create(nb,MFCCF16::TMPIN_MFCC_F16_ID,mgr);
          }
          break;
        case MFCCF16::TEST_MFCC_F16_3:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCF16::REF_MFCC_NOISE_1024_F16_ID,mgr,nb);
            input1.reload(MFCCF16::INPUTS_MFCC_NOISE_1024_F16_ID,mgr,nb);
            arm_mfcc_init_f16(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f16,
                      mfcc_filter_pos_config1_f16,mfcc_filter_len_config1_f16,
                      mfcc_filter_coefs_config1_f16,
                      mfcc_window_coefs_config1_f16);
            tmp.create(2*nb,MFCCF16::TMP_MFCC_F16_ID,mgr);
            tmpin.create(nb,MFCCF16::TMPIN_MFCC_F16_ID,mgr);

          }
          break;

        case MFCCF16::TEST_MFCC_F16_4:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCF16::REF_MFCC_SINE_256_F16_ID,mgr,nb);
            input1.reload(MFCCF16::INPUTS_MFCC_SINE_256_F16_ID,mgr,nb);
            arm_mfcc_init_f16(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_f16,
                    mfcc_filter_pos_config3_f16,mfcc_filter_len_config3_f16,
                    mfcc_filter_coefs_config3_f16,
                    mfcc_window_coefs_config3_f16);
            tmp.create(2*nb,MFCCF16::TMP_MFCC_F16_ID,mgr);
            tmpin.create(nb,MFCCF16::TMPIN_MFCC_F16_ID,mgr);
          }
          break;

        case MFCCF16::TEST_MFCC_F16_5:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCF16::REF_MFCC_SINE_512_F16_ID,mgr,nb);
            input1.reload(MFCCF16::INPUTS_MFCC_SINE_512_F16_ID,mgr,nb);
            arm_mfcc_init_f16(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f16,
                      mfcc_filter_pos_config2_f16,mfcc_filter_len_config2_f16,
                      mfcc_filter_coefs_config2_f16,
                      mfcc_window_coefs_config2_f16);
            tmp.create(2*nb,MFCCF16::TMP_MFCC_F16_ID,mgr);
            tmpin.create(nb,MFCCF16::TMPIN_MFCC_F16_ID,mgr);
          }
          break;
        case MFCCF16::TEST_MFCC_F16_6:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCF16::REF_MFCC_SINE_1024_F16_ID,mgr,nb);
            input1.reload(MFCCF16::INPUTS_MFCC_SINE_1024_F16_ID,mgr,nb);
            arm_mfcc_init_f16(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f16,
                      mfcc_filter_pos_config1_f16,mfcc_filter_len_config1_f16,
                      mfcc_filter_coefs_config1_f16,
                      mfcc_window_coefs_config1_f16);
            tmp.create(2*nb,MFCCF16::TMP_MFCC_F16_ID,mgr);
            tmpin.create(nb,MFCCF16::TMPIN_MFCC_F16_ID,mgr);

          }
          break;

       }
      

       output.create(ref.nbSamples(),MFCCF16::OUTPUT_MFCC_F16_ID,mgr);

    }

    void MFCCF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        (void)mgr;
        //output.dump(mgr);
    }
