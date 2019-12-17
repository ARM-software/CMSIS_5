#include "DistanceTestsU32.h"
#include <stdio.h>
#include "Error.h"
#include "arm_math.h"
#include "Test.h"

#define ERROR_THRESHOLD 1e-8

    void DistanceTestsU32::test_dice_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_dice_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_REL_ERROR(output,ref,(float32_t)ERROR_THRESHOLD);
    } 

    void DistanceTestsU32::test_hamming_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_hamming_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    } 

    void DistanceTestsU32::test_jaccard_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_jaccard_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    } 

    void DistanceTestsU32::test_kulsinski_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_kulsinski_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    }

    void DistanceTestsU32::test_rogerstanimoto_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_rogerstanimoto_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    }

    void DistanceTestsU32::test_russellrao_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_russellrao_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    }

    void DistanceTestsU32::test_sokalmichener_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_sokalmichener_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    }

    void DistanceTestsU32::test_sokalsneath_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_sokalsneath_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    }

    void DistanceTestsU32::test_yule_distance()
    {
       const uint32_t *inpA = inputA.ptr();
       const uint32_t *inpB = inputB.ptr();

       float32_t *outp = output.ptr();
       
       for(int i=0; i < this->nbPatterns ; i ++)
       {
          *outp = arm_yule_distance(inpA, inpB,this->vecDim);
         
          inpA += this->bitVecDim ;
          inpB += this->bitVecDim ;
          outp ++;
       }

        ASSERT_NEAR_EQ(output,ref,(float32_t)ERROR_THRESHOLD);
    }


  
  
    void DistanceTestsU32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

        inputA.reload(DistanceTestsU32::INPUTA_U32_ID,mgr);
        inputB.reload(DistanceTestsU32::INPUTB_U32_ID,mgr);
        dims.reload(DistanceTestsU32::DIMS_S16_ID,mgr);

        const int16_t   *dimsp = dims.ptr();

        this->nbPatterns=dimsp[0];
        this->vecDim=dimsp[1];
        this->bitVecDim=dimsp[2];
        //printf("%d %d %d\n",dimsp[0],dimsp[1],dimsp[2]);
        output.create(this->nbPatterns,DistanceTestsU32::OUT_F32_ID,mgr);

        switch(id)
        {
            case DistanceTestsU32::TEST_DICE_DISTANCE_1:
            {
              ref.reload(DistanceTestsU32::REF1_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_HAMMING_DISTANCE_2:
            {
              ref.reload(DistanceTestsU32::REF2_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_JACCARD_DISTANCE_3:
            {
              ref.reload(DistanceTestsU32::REF3_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_KULSINSKI_DISTANCE_4:
            {
              ref.reload(DistanceTestsU32::REF4_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_ROGERSTANIMOTO_DISTANCE_5:
            {
              ref.reload(DistanceTestsU32::REF5_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_RUSSELLRAO_DISTANCE_6:
            {
              ref.reload(DistanceTestsU32::REF6_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_SOKALMICHENER_DISTANCE_7:
            {
              ref.reload(DistanceTestsU32::REF7_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_SOKALSNEATH_DISTANCE_8:
            {
              ref.reload(DistanceTestsU32::REF8_F32_ID,mgr);
            }
            break;

            case DistanceTestsU32::TEST_YULE_DISTANCE_9:
            {
              ref.reload(DistanceTestsU32::REF9_F32_ID,mgr);
            }
            break;

        }


    }

    void DistanceTestsU32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       output.dump(mgr);
    }
