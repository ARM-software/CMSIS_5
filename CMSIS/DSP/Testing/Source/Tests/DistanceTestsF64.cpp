#include "DistanceTestsF64.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"


#define REL_ERROR (2.0e-14)

/*
    void DistanceTestsF64::test_braycurtis_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_braycurtis_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)1e-3);
    } 
 
    void DistanceTestsF64::test_canberra_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_canberra_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)1e-3);
    } 
*/
    void DistanceTestsF64::test_chebyshev_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_chebyshev_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)REL_ERROR);
    } 

    void DistanceTestsF64::test_cityblock_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_cityblock_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)REL_ERROR);
    } 

/*
    void DistanceTestsF64::test_correlation_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *tmpap = tmpA.ptr();
       float64_t *tmpbp = tmpB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          memcpy(tmpap, inpA, sizeof(float64_t) * this->vecDim);
          memcpy(tmpbp, inpB, sizeof(float64_t) * this->vecDim);
          
          *outp = arm_correlation_distance_f64(tmpap, tmpbp, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)1e-3);
    } 
*/
    void DistanceTestsF64::test_cosine_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_cosine_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)REL_ERROR);
    } 

    void DistanceTestsF64::test_euclidean_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_euclidean_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)REL_ERROR);
    } 
/*
    void DistanceTestsF64::test_jensenshannon_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();

       float64_t *outp = output.ptr();

      
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_jensenshannon_distance_f64(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)1e-3);
    } 

    void DistanceTestsF64::test_minkowski_distance_f64()
    {
       const float64_t *inpA = inputA.ptr();
       const float64_t *inpB = inputB.ptr();
       const int16_t   *dimsp= dims.ptr();
       dimsp += 2;

       float64_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_minkowski_distance_f64(inpA, inpB, *dimsp,this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
          dimsp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float64_t)1e-3);
    } 
  
 */ 
    void DistanceTestsF64::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        (void)paramsArgs;
        if ((id != DistanceTestsF64::TEST_MINKOWSKI_DISTANCE_F64_9) && (id != DistanceTestsF64::TEST_JENSENSHANNON_DISTANCE_F64_8))
        {
            inputA.reload(DistanceTestsF64::INPUTA_F64_ID,mgr);
            inputB.reload(DistanceTestsF64::INPUTB_F64_ID,mgr);
            dims.reload(DistanceTestsF64::DIMS_S16_ID,mgr);
            
            const int16_t   *dimsp = dims.ptr();
            
            this->nbPatterns=dimsp[0];
            this->vecDim=dimsp[1];
            output.create(this->nbPatterns,DistanceTestsF64::OUT_F64_ID,mgr);
        }

        switch(id)
        {
            case DistanceTestsF64::TEST_BRAYCURTIS_DISTANCE_F64_1:
            {
              ref.reload(DistanceTestsF64::REF1_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_CANBERRA_DISTANCE_F64_2:
            {
              ref.reload(DistanceTestsF64::REF2_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_CHEBYSHEV_DISTANCE_F64_3:
            {
              ref.reload(DistanceTestsF64::REF3_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_CITYBLOCK_DISTANCE_F64_4:
            {
              ref.reload(DistanceTestsF64::REF4_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_CORRELATION_DISTANCE_F64_5:
            {
              ref.reload(DistanceTestsF64::REF5_F64_ID,mgr);
              tmpA.create(this->vecDim,DistanceTestsF64::TMPA_F64_ID,mgr);
              tmpB.create(this->vecDim,DistanceTestsF64::TMPB_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_COSINE_DISTANCE_F64_6:
            {
              ref.reload(DistanceTestsF64::REF6_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_EUCLIDEAN_DISTANCE_F64_7:
            {
              ref.reload(DistanceTestsF64::REF7_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_JENSENSHANNON_DISTANCE_F64_8:
            {
              inputA.reload(DistanceTestsF64::INPUTA_JEN_F64_ID,mgr);
              inputB.reload(DistanceTestsF64::INPUTB_JEN_F64_ID,mgr);
              dims.reload(DistanceTestsF64::DIMS_S16_ID,mgr);
              
              const int16_t   *dimsp = dims.ptr();
              
              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,DistanceTestsF64::OUT_F64_ID,mgr);

              ref.reload(DistanceTestsF64::REF8_F64_ID,mgr);
            }
            break;

            case DistanceTestsF64::TEST_MINKOWSKI_DISTANCE_F64_9:
            {
              inputA.reload(DistanceTestsF64::INPUTA_F64_ID,mgr);
              inputB.reload(DistanceTestsF64::INPUTB_F64_ID,mgr);
              dims.reload(DistanceTestsF64::DIMS_MINKOWSKI_S16_ID,mgr);
              
              const int16_t   *dimsp = dims.ptr();
              
              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,DistanceTestsF64::OUT_F64_ID,mgr);

              ref.reload(DistanceTestsF64::REF9_F64_ID,mgr);
            }
            break;

        }

       

       

    }

    void DistanceTestsF64::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
