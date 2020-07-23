#include "DistanceTestsF32.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void DistanceTestsF32::test_braycurtis_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_braycurtis_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 
 
    void DistanceTestsF32::test_canberra_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_canberra_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_chebyshev_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_chebyshev_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_cityblock_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_cityblock_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_correlation_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *tmpap = tmpA.ptr();
       float32_t *tmpbp = tmpB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          memcpy(tmpap, inpA, sizeof(float32_t) * this->vecDim);
          memcpy(tmpbp, inpB, sizeof(float32_t) * this->vecDim);
          
          *outp = arm_correlation_distance_f32(tmpap, tmpbp, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_cosine_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_cosine_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_euclidean_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_euclidean_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_jensenshannon_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();

      
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_jensenshannon_distance_f32(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 

    void DistanceTestsF32::test_minkowski_distance_f32()
    {
       const float32_t *inpA = inputA.ptr();
       const float32_t *inpB = inputB.ptr();
       const int16_t   *dimsp= dims.ptr();
       dimsp += 2;

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_minkowski_distance_f32(inpA, inpB, *dimsp,this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
          dimsp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)1e-3);
    } 
  
  
    void DistanceTestsF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        (void)paramsArgs;
        if ((id != DistanceTestsF32::TEST_MINKOWSKI_DISTANCE_F32_9) && (id != DistanceTestsF32::TEST_JENSENSHANNON_DISTANCE_F32_8))
        {
            inputA.reload(DistanceTestsF32::INPUTA_F32_ID,mgr);
            inputB.reload(DistanceTestsF32::INPUTB_F32_ID,mgr);
            dims.reload(DistanceTestsF32::DIMS_S16_ID,mgr);
            
            const int16_t   *dimsp = dims.ptr();
            
            this->nbPatterns=dimsp[0];
            this->vecDim=dimsp[1];
            output.create(this->nbPatterns,DistanceTestsF32::OUT_F32_ID,mgr);
        }

        switch(id)
        {
            case DistanceTestsF32::TEST_BRAYCURTIS_DISTANCE_F32_1:
            {
              ref.reload(DistanceTestsF32::REF1_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_CANBERRA_DISTANCE_F32_2:
            {
              ref.reload(DistanceTestsF32::REF2_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_CHEBYSHEV_DISTANCE_F32_3:
            {
              ref.reload(DistanceTestsF32::REF3_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_CITYBLOCK_DISTANCE_F32_4:
            {
              ref.reload(DistanceTestsF32::REF4_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_CORRELATION_DISTANCE_F32_5:
            {
              ref.reload(DistanceTestsF32::REF5_F32_ID,mgr);
              tmpA.create(this->vecDim,DistanceTestsF32::TMPA_F32_ID,mgr);
              tmpB.create(this->vecDim,DistanceTestsF32::TMPB_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_COSINE_DISTANCE_F32_6:
            {
              ref.reload(DistanceTestsF32::REF6_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_EUCLIDEAN_DISTANCE_F32_7:
            {
              ref.reload(DistanceTestsF32::REF7_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_JENSENSHANNON_DISTANCE_F32_8:
            {
              inputA.reload(DistanceTestsF32::INPUTA_JEN_F32_ID,mgr);
              inputB.reload(DistanceTestsF32::INPUTB_JEN_F32_ID,mgr);
              dims.reload(DistanceTestsF32::DIMS_S16_ID,mgr);
              
              const int16_t   *dimsp = dims.ptr();
              
              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,DistanceTestsF32::OUT_F32_ID,mgr);

              ref.reload(DistanceTestsF32::REF8_F32_ID,mgr);
            }
            break;

            case DistanceTestsF32::TEST_MINKOWSKI_DISTANCE_F32_9:
            {
              inputA.reload(DistanceTestsF32::INPUTA_F32_ID,mgr);
              inputB.reload(DistanceTestsF32::INPUTB_F32_ID,mgr);
              dims.reload(DistanceTestsF32::DIMS_MINKOWSKI_S16_ID,mgr);
              
              const int16_t   *dimsp = dims.ptr();
              
              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,DistanceTestsF32::OUT_F32_ID,mgr);

              ref.reload(DistanceTestsF32::REF9_F32_ID,mgr);
            }
            break;

        }

       

       

    }

    void DistanceTestsF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
