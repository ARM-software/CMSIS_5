#include "DistanceTestsF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"

#define REL_ERROR (5e-3)

#define REL_JS_ERROR (3e-2)

#define REL_MK_ERROR (1e-2)


    void DistanceTestsF16::test_braycurtis_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_braycurtis_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 
 
    void DistanceTestsF16::test_canberra_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_canberra_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void DistanceTestsF16::test_chebyshev_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_chebyshev_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void DistanceTestsF16::test_cityblock_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_cityblock_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void DistanceTestsF16::test_correlation_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *tmpap = tmpA.ptr();
       float16_t *tmpbp = tmpB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          memcpy(tmpap, inpA, sizeof(float16_t) * this->vecDim);
          memcpy(tmpbp, inpB, sizeof(float16_t) * this->vecDim);
          
          *outp = arm_correlation_distance_f16(tmpap, tmpbp, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void DistanceTestsF16::test_cosine_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_cosine_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void DistanceTestsF16::test_euclidean_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_euclidean_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_ERROR);
    } 

    void DistanceTestsF16::test_jensenshannon_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();

       float16_t *outp = output.ptr();

      
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_jensenshannon_distance_f16(inpA, inpB, this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_JS_ERROR);
    } 

    void DistanceTestsF16::test_minkowski_distance_f16()
    {
       const float16_t *inpA = inputA.ptr();
       const float16_t *inpB = inputB.ptr();
       const int16_t   *dimsp= dims.ptr();
       dimsp += 2;

       float16_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_minkowski_distance_f16(inpA, inpB, *dimsp,this->vecDim);
         
          inpA += this->vecDim;
          inpB += this->vecDim;
          outp ++;
          dimsp ++;
       }

        ASSERT_REL_ERROR(output,ref,REL_MK_ERROR);
    } 
  
  
    void DistanceTestsF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        (void)paramsArgs;
        if ((id != DistanceTestsF16::TEST_MINKOWSKI_DISTANCE_F16_9) && (id != DistanceTestsF16::TEST_JENSENSHANNON_DISTANCE_F16_8))
        {
            inputA.reload(DistanceTestsF16::INPUTA_F16_ID,mgr);
            inputB.reload(DistanceTestsF16::INPUTB_F16_ID,mgr);
            dims.reload(DistanceTestsF16::DIMS_S16_ID,mgr);
            
            const int16_t   *dimsp = dims.ptr();
            
            this->nbPatterns=dimsp[0];
            this->vecDim=dimsp[1];
            output.create(this->nbPatterns,DistanceTestsF16::OUT_F16_ID,mgr);
        }

        switch(id)
        {
            case DistanceTestsF16::TEST_BRAYCURTIS_DISTANCE_F16_1:
            {
              ref.reload(DistanceTestsF16::REF1_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_CANBERRA_DISTANCE_F16_2:
            {
              ref.reload(DistanceTestsF16::REF2_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_CHEBYSHEV_DISTANCE_F16_3:
            {
              ref.reload(DistanceTestsF16::REF3_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_CITYBLOCK_DISTANCE_F16_4:
            {
              ref.reload(DistanceTestsF16::REF4_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_CORRELATION_DISTANCE_F16_5:
            {
              ref.reload(DistanceTestsF16::REF5_F16_ID,mgr);
              tmpA.create(this->vecDim,DistanceTestsF16::TMPA_F16_ID,mgr);
              tmpB.create(this->vecDim,DistanceTestsF16::TMPB_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_COSINE_DISTANCE_F16_6:
            {
              ref.reload(DistanceTestsF16::REF6_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_EUCLIDEAN_DISTANCE_F16_7:
            {
              ref.reload(DistanceTestsF16::REF7_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_JENSENSHANNON_DISTANCE_F16_8:
            {
              inputA.reload(DistanceTestsF16::INPUTA_JEN_F16_ID,mgr);
              inputB.reload(DistanceTestsF16::INPUTB_JEN_F16_ID,mgr);
              dims.reload(DistanceTestsF16::DIMS_S16_ID,mgr);
              
              const int16_t   *dimsp = dims.ptr();
              
              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,DistanceTestsF16::OUT_F16_ID,mgr);

              ref.reload(DistanceTestsF16::REF8_F16_ID,mgr);
            }
            break;

            case DistanceTestsF16::TEST_MINKOWSKI_DISTANCE_F16_9:
            {
              inputA.reload(DistanceTestsF16::INPUTA_F16_ID,mgr);
              inputB.reload(DistanceTestsF16::INPUTB_F16_ID,mgr);
              dims.reload(DistanceTestsF16::DIMS_MINKOWSKI_S16_ID,mgr);
              
              const int16_t   *dimsp = dims.ptr();
              
              this->nbPatterns=dimsp[0];
              this->vecDim=dimsp[1];
              output.create(this->nbPatterns,DistanceTestsF16::OUT_F16_ID,mgr);

              ref.reload(DistanceTestsF16::REF9_F16_ID,mgr);
            }
            break;

        }

       

       

    }

    void DistanceTestsF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       output.dump(mgr);
    }
