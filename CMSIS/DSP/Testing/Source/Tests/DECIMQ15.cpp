#include "DECIMQ15.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 70

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q15 ((q15_t)5)
#define ABS_ERROR_Q63 ((q63_t)(1<<17))

#define ONEHALF 0x40000000


    void DECIMQ15::test_fir_decimate_q15()
    {
        int nbTests;
        int nb;
        uint32_t *pConfig = config.ptr();

        const q15_t * pSrc = input.ptr();
        q15_t * pDst = output.ptr();
        q15_t * pCoefs = coefs.ptr();

        nbTests=config.nbSamples() / 4;

        for(nb=0;nb < nbTests; nb++)
        {

            this->q = pConfig[0];
            this->numTaps = pConfig[1];
            this->blocksize = pConfig[2];
            this->refsize = pConfig[3];

            pConfig += 4;

            this->status=arm_fir_decimate_init_q15(&(this->S),
               this->numTaps,
               this->q,
               pCoefs,
               state.ptr(),
               this->blocksize);

            ASSERT_TRUE(this->status == ARM_MATH_SUCCESS);

            arm_fir_decimate_q15(
                 &(this->S),
                 pSrc,
                 pDst,
                 this->blocksize);

            pSrc += this->blocksize;
            pDst += this->refsize;

            pCoefs += this->numTaps;
        }


        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 

    void DECIMQ15::test_fir_interpolate_q15()
    {
        int nbTests;
        int nb;
        uint32_t *pConfig = config.ptr();

        const q15_t * pSrc = input.ptr();
        q15_t * pDst = output.ptr();
        q15_t * pCoefs = coefs.ptr();

        nbTests=config.nbSamples() / 4;

        for(nb=0;nb < nbTests; nb++)
        {

            this->q = pConfig[0];
            this->numTaps = pConfig[1];
            this->blocksize = pConfig[2];
            this->refsize = pConfig[3];


            pConfig += 4;

            this->status=arm_fir_interpolate_init_q15(&(this->SI),
               this->q,
               this->numTaps,
               pCoefs,
               state.ptr(),
               this->blocksize);



            ASSERT_TRUE(this->status == ARM_MATH_SUCCESS);

            arm_fir_interpolate_q15(
                 &(this->SI),
                 pSrc,
                 pDst,
                 this->blocksize);

            pSrc += this->blocksize;
            pDst += this->refsize;

            pCoefs += this->numTaps;
        }


        ASSERT_EMPTY_TAIL(output);

        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q15);

    } 
   
    void DECIMQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;
       config.reload(DECIMQ15::CONFIGSDECIMQ15_ID,mgr);
       
       
       switch(id)
       {
        case DECIMQ15::TEST_FIR_DECIMATE_Q15_1:
          config.reload(DECIMQ15::CONFIGSDECIMQ15_ID,mgr);
          input.reload(DECIMQ15::INPUT1_Q15_ID,mgr);
          coefs.reload(DECIMQ15::COEFS1_Q15_ID,mgr);

          ref.reload(DECIMQ15::REF1_DECIM_Q15_ID,mgr);
          state.create(16 + 768 - 1,DECIMQ15::STATE_Q15_ID,mgr);

          break;

        case DECIMQ15::TEST_FIR_INTERPOLATE_Q15_2:
          config.reload(DECIMQ15::CONFIGSINTERPQ15_ID,mgr);

          input.reload(DECIMQ15::INPUT2_Q15_ID,mgr);
          coefs.reload(DECIMQ15::COEFS2_Q15_ID,mgr);

          ref.reload(DECIMQ15::REF2_INTERP_Q15_ID,mgr);
          state.create(16 + 768 - 1,DECIMQ15::STATE_Q15_ID,mgr);

          break;

       }
      

       

       output.create(ref.nbSamples(),DECIMQ15::OUT_Q15_ID,mgr);
    }

    void DECIMQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
    }
