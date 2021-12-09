#include "BIQUADF64.h"
#include <stdio.h>
#include "Error.h"

#define SNR_THRESHOLD 98

/* 

Reference patterns are generated with
a double precision computation.

*/
#define REL_ERROR (1.2e-3)

    void BIQUADF64::test_biquad_cascade_df2T_ref()
    {


        float64_t *statep = state.ptr();


        float64_t *coefsp = coefs.ptr();

        
        float64_t *inputp = inputs.ptr();
        float64_t *outp = output.ptr();

        int blockSize;

        

        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           blockSize = inputs.nbSamples() >> 1;

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
           arm_biquad_cascade_df2T_init_f64(&this->Sdf2T,3,coefsp,statep);


           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df2T_f64(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;

           
           inputp += blockSize;
           arm_biquad_cascade_df2T_f64(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;


           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

 

    void BIQUADF64::test_biquad_cascade_df2T_rand()
    {


        float64_t *statep = state.ptr();
        const int16_t *configsp = configs.ptr();

        float64_t *coefsp = coefs.ptr();

        
        float64_t *inputp = inputs.ptr();
        float64_t *outp = output.ptr();

        int blockSize;
        int numStages;

        unsigned long i;

        

        for(i=0;i < configs.nbSamples(); i+=2)
        {

        /*

        Python script is generating different tests with
        different blockSize and numTaps.

        We loop on those configs.

        */
        
           numStages = configsp[0];
           blockSize = configsp[1];

           configsp += 2;

          

           /*

           The filter is initialized with the coefs, blockSize and numTaps.

           */
           arm_biquad_cascade_df2T_init_f64(&this->Sdf2T,numStages,coefsp,statep);

           coefsp += numStages * 5;

           /*
           
           Python script is filtering a 2*blockSize number of samples.
           We do the same filtering in two pass to check (indirectly that
           the state management of the fir is working.)

           */

           arm_biquad_cascade_df2T_f64(&this->Sdf2T,inputp,outp,blockSize);
           outp += blockSize;
           inputp += blockSize;
           
        }

           ASSERT_EMPTY_TAIL(output);

           ASSERT_SNR(output,ref,(float64_t)SNR_THRESHOLD);

           ASSERT_REL_ERROR(output,ref,REL_ERROR);
  

    } 

    void BIQUADF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       (void)params;
       switch(id)
       {
        case BIQUADF64::TEST_BIQUAD_CASCADE_DF2T_REF_1:

           inputs.reload(BIQUADF64::BIQUADINPUTS_F64_ID,mgr);
           coefs.reload(BIQUADF64::BIQUADCOEFS_F64_ID,mgr);
           ref.reload(BIQUADF64::BIQUADREFS_F64_ID,mgr);
        break;

        case BIQUADF64::TEST_BIQUAD_CASCADE_DF2T_RAND_2:

           inputs.reload(BIQUADF64::ALLBIQUADINPUTS_F64_ID,mgr);
           coefs.reload(BIQUADF64::ALLBIQUADCOEFS_F64_ID,mgr);
           ref.reload(BIQUADF64::ALLBIQUADREFS_F64_ID,mgr);
           configs.reload(BIQUADF64::ALLBIQUADCONFIGS_S16_ID,mgr);
        break;

       }
      

       

       output.create(ref.nbSamples(),BIQUADF64::OUT_F64_ID,mgr);
      
       state.create(128,BIQUADF64::STATE_F64_ID,mgr);

       
    }

    void BIQUADF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        (void)id;
        output.dump(mgr);
       
    }
