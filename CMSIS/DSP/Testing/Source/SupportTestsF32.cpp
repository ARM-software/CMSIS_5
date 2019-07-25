#include "SupportTestsF32.h"
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#include <cstdio>


    void SupportTestsF32::test_barycenter_f32()
    {
       const float32_t *inp = input.ptr();
       const float32_t *coefsp = coefs.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
        arm_barycenter_f32(inp, coefsp,
            outp, 
            this->nbVectors, 
            this->vecDim);
         
          inp += this->vecDim * this->nbVectors;
          coefsp += this->nbVectors;
          outp += this->vecDim;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void SupportTestsF32::test_weighted_sum_f32()
    {
       const float32_t *inp = input.ptr();
       const float32_t *coefsp = coefs.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp=arm_weighted_sum_f32(inp, coefsp,
            this->vecDim);
         
          inp += this->vecDim;
          coefsp += this->vecDim;
          outp++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

  
    void SupportTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        switch(id)
        {
            case TEST_BARYCENTER_F32_1:
            {
              input.reload(SupportTestsF32::INPUTS1_F32_ID,mgr);
              coefs.reload(SupportTestsF32::WEIGHTS1_F32_ID,mgr);
              dims.reload(SupportTestsF32::DIMS1_S16_ID,mgr);
              ref.reload(SupportTestsF32::REF1_F32_ID,mgr);

              const int16_t   *dimsp = dims.ptr();

              this->nbPatterns=dimsp[0];
              this->nbVectors=dimsp[1];
              this->vecDim=dimsp[2];
              output.create(this->nbPatterns*this->vecDim,SupportTestsF32::OUT_F32_ID,mgr);
            }
            break;

            case TEST_WEIGHTED_SUM_F32_2:
              input.reload(SupportTestsF32::INPUTS2_F32_ID,mgr);
              coefs.reload(SupportTestsF32::WEIGHTS2_F32_ID,mgr);
              dims.reload(SupportTestsF32::DIMS2_S16_ID,mgr);
              ref.reload(SupportTestsF32::REF2_F32_ID,mgr);

              const int16_t   *dimsp = dims.ptr();

              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,SupportTestsF32::OUT_F32_ID,mgr);
            break;
        }

       

    }

    void SupportTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
