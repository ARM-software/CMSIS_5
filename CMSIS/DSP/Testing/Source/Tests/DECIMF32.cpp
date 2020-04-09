#include "DECIMF32.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 120

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (8.0e-4)


    void DECIMF32::test_fir_decimate_f32()
    {
        int nbTests;
        int nb;
        uint32_t *pConfig = config.ptr();

        const float32_t * pSrc = input.ptr();
        float32_t * pDst = output.ptr();
        float32_t * pCoefs = coefs.ptr();

        nbTests=config.nbSamples() / 4;

        for(nb=0;nb < nbTests; nb++)
        {

            this->q = pConfig[0];
            this->numTaps = pConfig[1];
            this->blocksize = pConfig[2];
            this->refsize = pConfig[3];


            pConfig += 4;

            this->status=arm_fir_decimate_init_f32(&(this->S),
               this->numTaps,
               this->q,
               pCoefs,
               state.ptr(),
               this->blocksize);



            ASSERT_TRUE(this->status == ARM_MATH_SUCCESS);

            arm_fir_decimate_f32(
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

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

    void DECIMF32::test_fir_interpolate_f32()
    {
        int nbTests;
        int nb;
        uint32_t *pConfig = config.ptr();

        const float32_t * pSrc = input.ptr();
        float32_t * pDst = output.ptr();
        float32_t * pCoefs = coefs.ptr();

        nbTests=config.nbSamples() / 4;

        for(nb=0;nb < nbTests; nb++)
        {

            this->q = pConfig[0];
            this->numTaps = pConfig[1];
            this->blocksize = pConfig[2];
            this->refsize = pConfig[3];



            pConfig += 4;

            this->status=arm_fir_interpolate_init_f32(&(this->SI),
               this->q,
               this->numTaps,
               pCoefs,
               state.ptr(),
               this->blocksize);



            ASSERT_TRUE(this->status == ARM_MATH_SUCCESS);

            arm_fir_interpolate_f32(
                 &(this->SI),
                 pSrc,
                 pDst,
                 this->blocksize);

            pSrc += this->blocksize;
            pDst += this->refsize;

            pCoefs += this->numTaps;
        }


        ASSERT_EMPTY_TAIL(output);

        //ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 

   
    void DECIMF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       
       
       switch(id)
       {
        case DECIMF32::TEST_FIR_DECIMATE_F32_1:
          config.reload(DECIMF32::CONFIGSDECIMF32_ID,mgr);
         
          input.reload(DECIMF32::INPUT1_F32_ID,mgr);
          coefs.reload(DECIMF32::COEFS1_F32_ID,mgr);

          ref.reload(DECIMF32::REF1_DECIM_F32_ID,mgr);
          state.create(16 + 768 - 1,DECIMF32::STATE_F32_ID,mgr);

          break;

        case DECIMF32::TEST_FIR_INTERPOLATE_F32_2:
          config.reload(DECIMF32::CONFIGSINTERPF32_ID,mgr);
         
          input.reload(DECIMF32::INPUT2_F32_ID,mgr);
          coefs.reload(DECIMF32::COEFS2_F32_ID,mgr);

          ref.reload(DECIMF32::REF2_INTERP_F32_ID,mgr);
          state.create(16 + 768 - 1,DECIMF32::STATE_F32_ID,mgr);

          break;


       }
      

       

       output.create(ref.nbSamples(),DECIMF32::OUT_F32_ID,mgr);
    }

    void DECIMF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
