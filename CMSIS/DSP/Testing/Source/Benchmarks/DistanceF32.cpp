#include "DistanceF32.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void DistanceF32::test_braycurtis_distance_f32()
    {
       float32_t outp;
       
       outp = arm_braycurtis_distance_f32(inpA, inpB, this->vecDim);
         
      
    } 
 
    void DistanceF32::test_canberra_distance_f32()
    {
       float32_t outp;
       
       outp = arm_canberra_distance_f32(inpA, inpB, this->vecDim);
        
    } 

    void DistanceF32::test_chebyshev_distance_f32()
    {
       float32_t outp;
       
       outp = arm_chebyshev_distance_f32(inpA, inpB, this->vecDim);
         
        
    } 

    void DistanceF32::test_cityblock_distance_f32()
    {
       float32_t outp;
       
       outp = arm_cityblock_distance_f32(inpA, inpB, this->vecDim);
         

    } 

    void DistanceF32::test_correlation_distance_f32()
    {
        float32_t outp;
       
        memcpy(tmpAp, inpA, sizeof(float32_t) * this->vecDim);
        memcpy(tmpBp, inpB, sizeof(float32_t) * this->vecDim);
          
        outp = arm_correlation_distance_f32(tmpAp, tmpBp, this->vecDim);
     
    } 

    void DistanceF32::test_cosine_distance_f32()
    {
       float32_t outp;
       
       outp = arm_cosine_distance_f32(inpA, inpB, this->vecDim);
         
    } 

    void DistanceF32::test_euclidean_distance_f32()
    {
       float32_t outp;
       
       outp = arm_euclidean_distance_f32(inpA, inpB, this->vecDim);
         
    } 

    void DistanceF32::test_jensenshannon_distance_f32()
    {
       float32_t outp;

       outp = arm_jensenshannon_distance_f32(inpA, inpB, this->vecDim);
         
    } 

    void DistanceF32::test_minkowski_distance_f32()
    {
       float32_t outp;
       
       outp = arm_minkowski_distance_f32(inpA, inpB, 2,this->vecDim);
  
    } 
  
  
    void DistanceF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->vecDim = *it++;
 
        if ((id != DistanceF32::TEST_MINKOWSKI_DISTANCE_F32_9) && (id != DistanceF32::TEST_JENSENSHANNON_DISTANCE_F32_8))
        {
            inputA.reload(DistanceF32::INPUTA_PROBA_F32_ID,mgr);
            inputB.reload(DistanceF32::INPUTB_PROBA_F32_ID,mgr);
            
        }
        else
        {
           inputA.reload(DistanceF32::INPUTA_F32_ID,mgr);
           inputB.reload(DistanceF32::INPUTB_F32_ID,mgr);
        }

        if (id == DistanceF32::TEST_CORRELATION_DISTANCE_F32_5)
        {
              tmpA.create(this->vecDim,DistanceF32::TMPA_F32_ID,mgr);
              tmpB.create(this->vecDim,DistanceF32::TMPB_F32_ID,mgr);

              tmpAp = tmpA.ptr();
              tmpBp = tmpB.ptr();
        }

       inpA=inputA.ptr();
       inpB=inputB.ptr();
       

    }

    void DistanceF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
    }
