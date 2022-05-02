#include "MFCCQ31.h"
#include <stdio.h>
#include "Error.h"

#include "mfccdata.h"

#define SNR_THRESHOLD 70

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR ((q31_t)42000)




    void MFCCQ31::test_mfcc_q31()
    {
        const q31_t *inp1=input1.ptr(); 
        q31_t *tmpinp=tmpin.ptr(); 
        q31_t *outp=output.ptr();
        q31_t *tmpp=tmp.ptr();


        memcpy((void*)tmpinp,(void*)inp1,sizeof(q31_t)*this->fftLen);
        arm_mfcc_q31(&mfcc,tmpinp,outp,tmpp);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(q31_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR);

    } 

   
    void MFCCQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case MFCCQ31::TEST_MFCC_Q31_1:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCQ31::REF_MFCC_NOISE_256_Q31_ID,mgr,nb);
            input1.reload(MFCCQ31::INPUTS_MFCC_NOISE_256_Q31_ID,mgr,nb);
            arm_mfcc_init_q31(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_q31,
                    mfcc_filter_pos_config3_q31,mfcc_filter_len_config3_q31,
                    mfcc_filter_coefs_config3_q31,
                    mfcc_window_coefs_config3_q31);
            tmp.create(2*nb,MFCCQ31::TMP_MFCC_Q31_ID,mgr);
            tmpin.create(nb,MFCCQ31::TMPIN_MFCC_Q31_ID,mgr);
          }
          break;

        case MFCCQ31::TEST_MFCC_Q31_2:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCQ31::REF_MFCC_NOISE_512_Q31_ID,mgr,nb);
            input1.reload(MFCCQ31::INPUTS_MFCC_NOISE_512_Q31_ID,mgr,nb);
            arm_mfcc_init_q31(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q31,
                      mfcc_filter_pos_config2_q31,mfcc_filter_len_config2_q31,
                      mfcc_filter_coefs_config2_q31,
                      mfcc_window_coefs_config2_q31);
            tmp.create(2*nb,MFCCQ31::TMP_MFCC_Q31_ID,mgr);
            tmpin.create(nb,MFCCQ31::TMPIN_MFCC_Q31_ID,mgr);
          }
          break;
        case MFCCQ31::TEST_MFCC_Q31_3:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCQ31::REF_MFCC_NOISE_1024_Q31_ID,mgr,nb);
            input1.reload(MFCCQ31::INPUTS_MFCC_NOISE_1024_Q31_ID,mgr,nb);
            arm_mfcc_init_q31(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q31,
                      mfcc_filter_pos_config1_q31,mfcc_filter_len_config1_q31,
                      mfcc_filter_coefs_config1_q31,
                      mfcc_window_coefs_config1_q31);
            tmp.create(2*nb,MFCCQ31::TMP_MFCC_Q31_ID,mgr);
            tmpin.create(nb,MFCCQ31::TMPIN_MFCC_Q31_ID,mgr);

          }
          break;

        case MFCCQ31::TEST_MFCC_Q31_4:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCQ31::REF_MFCC_SINE_256_Q31_ID,mgr,nb);
            input1.reload(MFCCQ31::INPUTS_MFCC_SINE_256_Q31_ID,mgr,nb);
            arm_mfcc_init_q31(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_q31,
                    mfcc_filter_pos_config3_q31,mfcc_filter_len_config3_q31,
                    mfcc_filter_coefs_config3_q31,
                    mfcc_window_coefs_config3_q31);
            tmp.create(2*nb,MFCCQ31::TMP_MFCC_Q31_ID,mgr);
            tmpin.create(nb,MFCCQ31::TMPIN_MFCC_Q31_ID,mgr);
          }
          break;

        case MFCCQ31::TEST_MFCC_Q31_5:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCQ31::REF_MFCC_SINE_512_Q31_ID,mgr,nb);
            input1.reload(MFCCQ31::INPUTS_MFCC_SINE_512_Q31_ID,mgr,nb);
            arm_mfcc_init_q31(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q31,
                      mfcc_filter_pos_config2_q31,mfcc_filter_len_config2_q31,
                      mfcc_filter_coefs_config2_q31,
                      mfcc_window_coefs_config2_q31);
            tmp.create(2*nb,MFCCQ31::TMP_MFCC_Q31_ID,mgr);
            tmpin.create(nb,MFCCQ31::TMPIN_MFCC_Q31_ID,mgr);
          }
          break;
        case MFCCQ31::TEST_MFCC_Q31_6:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCQ31::REF_MFCC_SINE_1024_Q31_ID,mgr,nb);
            input1.reload(MFCCQ31::INPUTS_MFCC_SINE_1024_Q31_ID,mgr,nb);
            arm_mfcc_init_q31(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q31,
                      mfcc_filter_pos_config1_q31,mfcc_filter_len_config1_q31,
                      mfcc_filter_coefs_config1_q31,
                      mfcc_window_coefs_config1_q31);
            tmp.create(2*nb,MFCCQ31::TMP_MFCC_Q31_ID,mgr);
            tmpin.create(nb,MFCCQ31::TMPIN_MFCC_Q31_ID,mgr);

          }
          break;

       }
      

       output.create(ref.nbSamples(),MFCCQ31::OUTPUT_MFCC_Q31_ID,mgr);

    }

    void MFCCQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        (void)mgr;
        //output.dump(mgr);
    }
