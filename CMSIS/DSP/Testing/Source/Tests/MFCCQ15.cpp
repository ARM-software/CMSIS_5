#include "MFCCQ15.h"
#include <stdio.h>
#include "Error.h"

#include "mfccdata.h"

#define SNR_THRESHOLD 34

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR ((q15_t)30)



    void MFCCQ15::test_mfcc_q15()
    {
        const q15_t *inp1=input1.ptr(); 
        q15_t *tmpinp=tmpin.ptr(); 
        q15_t *outp=output.ptr();
        q31_t *tmpp=tmp.ptr();


        memcpy((void*)tmpinp,(void*)inp1,sizeof(q15_t)*this->fftLen);
        arm_mfcc_q15(&mfcc,tmpinp,outp,tmpp);

        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(q15_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR);

    } 

   
    void MFCCQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;

       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 

       
       switch(id)
       {
        case MFCCQ15::TEST_MFCC_Q15_1:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCQ15::REF_MFCC_NOISE_256_Q15_ID,mgr,nb);
            input1.reload(MFCCQ15::INPUTS_MFCC_NOISE_256_Q15_ID,mgr,nb);
            arm_mfcc_init_q15(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_q15,
                    mfcc_filter_pos_config3_q15,mfcc_filter_len_config3_q15,
                    mfcc_filter_coefs_config3_q15,
                    mfcc_window_coefs_config3_q15);
            tmp.create(2*nb,MFCCQ15::TMP_MFCC_Q15_ID,mgr);
            tmpin.create(nb,MFCCQ15::TMPIN_MFCC_Q15_ID,mgr);
          }
          break;

        case MFCCQ15::TEST_MFCC_Q15_2:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCQ15::REF_MFCC_NOISE_512_Q15_ID,mgr,nb);
            input1.reload(MFCCQ15::INPUTS_MFCC_NOISE_512_Q15_ID,mgr,nb);
            arm_mfcc_init_q15(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q15,
                      mfcc_filter_pos_config2_q15,mfcc_filter_len_config2_q15,
                      mfcc_filter_coefs_config2_q15,
                      mfcc_window_coefs_config2_q15);
            tmp.create(2*nb,MFCCQ15::TMP_MFCC_Q15_ID,mgr);
            tmpin.create(nb,MFCCQ15::TMPIN_MFCC_Q15_ID,mgr);
          }
          break;
        case MFCCQ15::TEST_MFCC_Q15_3:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCQ15::REF_MFCC_NOISE_1024_Q15_ID,mgr,nb);
            input1.reload(MFCCQ15::INPUTS_MFCC_NOISE_1024_Q15_ID,mgr,nb);
            arm_mfcc_init_q15(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q15,
                      mfcc_filter_pos_config1_q15,mfcc_filter_len_config1_q15,
                      mfcc_filter_coefs_config1_q15,
                      mfcc_window_coefs_config1_q15);
            tmp.create(2*nb,MFCCQ15::TMP_MFCC_Q15_ID,mgr);
            tmpin.create(nb,MFCCQ15::TMPIN_MFCC_Q15_ID,mgr);

          }
          break;

        case MFCCQ15::TEST_MFCC_Q15_4:
        {  
            nb = 256;
            this->fftLen = nb;
            ref.reload(MFCCQ15::REF_MFCC_SINE_256_Q15_ID,mgr,nb);
            input1.reload(MFCCQ15::INPUTS_MFCC_SINE_256_Q15_ID,mgr,nb);
            arm_mfcc_init_q15(&mfcc,
                    nb,20,13,mfcc_dct_coefs_config1_q15,
                    mfcc_filter_pos_config3_q15,mfcc_filter_len_config3_q15,
                    mfcc_filter_coefs_config3_q15,
                    mfcc_window_coefs_config3_q15);
            tmp.create(2*nb,MFCCQ15::TMP_MFCC_Q15_ID,mgr);
            tmpin.create(nb,MFCCQ15::TMPIN_MFCC_Q15_ID,mgr);
          }
          break;

        case MFCCQ15::TEST_MFCC_Q15_5:
          {
            nb = 512;
            this->fftLen = nb;
            ref.reload(MFCCQ15::REF_MFCC_SINE_512_Q15_ID,mgr,nb);
            input1.reload(MFCCQ15::INPUTS_MFCC_SINE_512_Q15_ID,mgr,nb);
            arm_mfcc_init_q15(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q15,
                      mfcc_filter_pos_config2_q15,mfcc_filter_len_config2_q15,
                      mfcc_filter_coefs_config2_q15,
                      mfcc_window_coefs_config2_q15);
            tmp.create(2*nb,MFCCQ15::TMP_MFCC_Q15_ID,mgr);
            tmpin.create(nb,MFCCQ15::TMPIN_MFCC_Q15_ID,mgr);
          }
          break;
        case MFCCQ15::TEST_MFCC_Q15_6:
          {
            nb = 1024;
            this->fftLen = nb;
            ref.reload(MFCCQ15::REF_MFCC_SINE_1024_Q15_ID,mgr,nb);
            input1.reload(MFCCQ15::INPUTS_MFCC_SINE_1024_Q15_ID,mgr,nb);
            arm_mfcc_init_q15(&mfcc,
                      nb,20,13,mfcc_dct_coefs_config1_q15,
                      mfcc_filter_pos_config1_q15,mfcc_filter_len_config1_q15,
                      mfcc_filter_coefs_config1_q15,
                      mfcc_window_coefs_config1_q15);
            tmp.create(2*nb,MFCCQ15::TMP_MFCC_Q15_ID,mgr);
            tmpin.create(nb,MFCCQ15::TMPIN_MFCC_Q15_ID,mgr);

          }
          break;

       }
      

       output.create(ref.nbSamples(),MFCCQ15::OUTPUT_MFCC_Q15_ID,mgr);

    }

    void MFCCQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        (void)mgr;
        //output.dump(mgr);
    }
