#include "DECIMQ31.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 100

/* 

Reference patterns are generated with
a double precision computation.

*/
#define ABS_ERROR_Q31 ((q31_t)2)
#define ABS_ERROR_Q63 ((q63_t)(1<<17))

#define ONEHALF 0x40000000


    void DECIMQ31::test_fir_decimate_q31()
    {
        int nbTests;
        int nb;
        uint32_t *pConfig = config.ptr();

        const q31_t * pSrc = input.ptr();
        q31_t * pDst = output.ptr();
        q31_t * pCoefs = coefs.ptr();

        nbTests=config.nbSamples() / 4;

        for(nb=0;nb < nbTests; nb++)
        {

            this->q = pConfig[0];
            this->numTaps = pConfig[1];
            this->blocksize = pConfig[2];
            this->refsize = pConfig[3];


            pConfig += 4;

            this->status=arm_fir_decimate_init_q31(&(this->S),
               this->numTaps,
               this->q,
               pCoefs,
               state.ptr(),
               this->blocksize);



            ASSERT_TRUE(this->status == ARM_MATH_SUCCESS);

            arm_fir_decimate_q31(
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

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

    void DECIMQ31::test_fir_interpolate_q31()
    {
        int nbTests;
        int nb;
        uint32_t *pConfig = config.ptr();

        const q31_t * pSrc = input.ptr();
        q31_t * pDst = output.ptr();
        q31_t * pCoefs = coefs.ptr();

        nbTests=config.nbSamples() / 4;

        for(nb=0;nb < nbTests; nb++)
        {

            this->q = pConfig[0];
            this->numTaps = pConfig[1];
            this->blocksize = pConfig[2];
            this->refsize = pConfig[3];


            pConfig += 4;

            this->status=arm_fir_interpolate_init_q31(&(this->SI),
               this->q,
               this->numTaps,
               pCoefs,
               state.ptr(),
               this->blocksize);



            ASSERT_TRUE(this->status == ARM_MATH_SUCCESS);

            arm_fir_interpolate_q31(
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

        ASSERT_NEAR_EQ(output,ref,ABS_ERROR_Q31);

    } 

   
    void DECIMQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       
       
       switch(id)
       {
        case DECIMQ31::TEST_FIR_DECIMATE_Q31_1:
          config.reload(DECIMQ31::CONFIGSDECIMQ31_ID,mgr);

          input.reload(DECIMQ31::INPUT1_Q31_ID,mgr);
          coefs.reload(DECIMQ31::COEFS1_Q31_ID,mgr);

          ref.reload(DECIMQ31::REF1_DECIM_Q31_ID,mgr);
          state.create(16 + 768 - 1,DECIMQ31::STATE_Q31_ID,mgr);

          break;

        case DECIMQ31::TEST_FIR_INTERPOLATE_Q31_2:
          config.reload(DECIMQ31::CONFIGSINTERPQ31_ID,mgr);

          input.reload(DECIMQ31::INPUT2_Q31_ID,mgr);
          coefs.reload(DECIMQ31::COEFS2_Q31_ID,mgr);

          ref.reload(DECIMQ31::REF2_INTERP_Q31_ID,mgr);
          state.create(16 + 768 - 1,DECIMQ31::STATE_Q31_ID,mgr);

          break;

       }
      

       

       output.create(ref.nbSamples(),DECIMQ31::OUT_Q31_ID,mgr);
    }

    void DECIMQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
