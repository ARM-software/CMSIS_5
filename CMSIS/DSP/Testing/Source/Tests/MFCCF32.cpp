#include "MFCCF32.h"
#include <stdio.h>
#include "Error.h"

#include "mfccdata.h"

#define SNR_THRESHOLD 115

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.2e-3)



    void MFCCF32::test_mfcc_f32()
    {
        const float32_t *inp1=input1.ptr(); 
        float32_t *tmpinp=tmpin.ptr(); 
        float32_t *outp=output.ptr();
        float32_t *tmpp=tmp.ptr();


        memcpy((void*)tmpinp,(void*)inp1,sizeof(float32_t)*this->fftLen);
        arm_mfcc_f32(&mfcc,tmpinp,outp,tmpp);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

   
    void MFCCF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case MFCCF32::TEST_MFCC_F32_1:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCF32::REF_MFCC_NOISE_256_F32_ID,mgr,nb);
            input1.reload(MFCCF32::INPUTS_MFCC_NOISE_256_F32_ID,mgr,nb);
            arm_mfcc_init_f32(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_f32,
                    mfcc_filter_pos_config3_f32,mfcc_filter_len_config3_f32,
                    mfcc_filter_coefs_config3_f32,
                    mfcc_window_coefs_config3_f32);
            tmp.create(2*nb,MFCCF32::TMP_MFCC_F32_ID,mgr);
            tmpin.create(nb,MFCCF32::TMPIN_MFCC_F32_ID,mgr);
          }
          break;

        case MFCCF32::TEST_MFCC_F32_2:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCF32::REF_MFCC_NOISE_512_F32_ID,mgr,nb);
            input1.reload(MFCCF32::INPUTS_MFCC_NOISE_512_F32_ID,mgr,nb);
            arm_mfcc_init_f32(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f32,
                      mfcc_filter_pos_config2_f32,mfcc_filter_len_config2_f32,
                      mfcc_filter_coefs_config2_f32,
                      mfcc_window_coefs_config2_f32);
            tmp.create(2*nb,MFCCF32::TMP_MFCC_F32_ID,mgr);
            tmpin.create(nb,MFCCF32::TMPIN_MFCC_F32_ID,mgr);
          }
          break;
        case MFCCF32::TEST_MFCC_F32_3:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCF32::REF_MFCC_NOISE_1024_F32_ID,mgr,nb);
            input1.reload(MFCCF32::INPUTS_MFCC_NOISE_1024_F32_ID,mgr,nb);
            arm_mfcc_init_f32(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f32,
                      mfcc_filter_pos_config1_f32,mfcc_filter_len_config1_f32,
                      mfcc_filter_coefs_config1_f32,
                      mfcc_window_coefs_config1_f32);
            tmp.create(2*nb,MFCCF32::TMP_MFCC_F32_ID,mgr);
            tmpin.create(nb,MFCCF32::TMPIN_MFCC_F32_ID,mgr);

          }
          break;

        case MFCCF32::TEST_MFCC_F32_4:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCF32::REF_MFCC_SINE_256_F32_ID,mgr,nb);
            input1.reload(MFCCF32::INPUTS_MFCC_SINE_256_F32_ID,mgr,nb);
            arm_mfcc_init_f32(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_f32,
                    mfcc_filter_pos_config3_f32,mfcc_filter_len_config3_f32,
                    mfcc_filter_coefs_config3_f32,
                    mfcc_window_coefs_config3_f32);
            tmp.create(2*nb,MFCCF32::TMP_MFCC_F32_ID,mgr);
            tmpin.create(nb,MFCCF32::TMPIN_MFCC_F32_ID,mgr);
          }
          break;

        case MFCCF32::TEST_MFCC_F32_5:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCF32::REF_MFCC_SINE_512_F32_ID,mgr,nb);
            input1.reload(MFCCF32::INPUTS_MFCC_SINE_512_F32_ID,mgr,nb);
            arm_mfcc_init_f32(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f32,
                      mfcc_filter_pos_config2_f32,mfcc_filter_len_config2_f32,
                      mfcc_filter_coefs_config2_f32,
                      mfcc_window_coefs_config2_f32);
            tmp.create(2*nb,MFCCF32::TMP_MFCC_F32_ID,mgr);
            tmpin.create(nb,MFCCF32::TMPIN_MFCC_F32_ID,mgr);
          }
          break;
        case MFCCF32::TEST_MFCC_F32_6:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCF32::REF_MFCC_SINE_1024_F32_ID,mgr,nb);
            input1.reload(MFCCF32::INPUTS_MFCC_SINE_1024_F32_ID,mgr,nb);
            arm_mfcc_init_f32(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_f32,
                      mfcc_filter_pos_config1_f32,mfcc_filter_len_config1_f32,
                      mfcc_filter_coefs_config1_f32,
                      mfcc_window_coefs_config1_f32);
            tmp.create(2*nb,MFCCF32::TMP_MFCC_F32_ID,mgr);
            tmpin.create(nb,MFCCF32::TMPIN_MFCC_F32_ID,mgr);

          }
          break;

       }
      

       output.create(ref.nbSamples(),MFCCF32::OUTPUT_MFCC_F32_ID,mgr);

    }

    void MFCCF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        (void)mgr;
        //output.dump(mgr);
    }
