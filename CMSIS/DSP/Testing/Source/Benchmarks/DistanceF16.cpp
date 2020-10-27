#include "DistanceF16.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



    void DistanceF16::test_braycurtis_distance_f16()
    {
       float16_t outp;
       
       outp = arm_braycurtis_distance_f16(inpA, inpB, this->vecDim);
         
      
    } 
 
    void DistanceF16::test_canberra_distance_f16()
    {
       float16_t outp;
       
       outp = arm_canberra_distance_f16(inpA, inpB, this->vecDim);
        
    } 

    void DistanceF16::test_chebyshev_distance_f16()
    {
       float16_t outp;
       
       outp = arm_chebyshev_distance_f16(inpA, inpB, this->vecDim);
         
        
    } 

    void DistanceF16::test_cityblock_distance_f16()
    {
       float16_t outp;
       
       outp = arm_cityblock_distance_f16(inpA, inpB, this->vecDim);
         

    } 

    void DistanceF16::test_correlation_distance_f16()
    {
        float16_t outp;
       
        memcpy(tmpAp, inpA, sizeof(float16_t) * this->vecDim);
        memcpy(tmpBp, inpB, sizeof(float16_t) * this->vecDim);
          
        outp = arm_correlation_distance_f16(tmpAp, tmpBp, this->vecDim);
     
    } 

    void DistanceF16::test_cosine_distance_f16()
    {
       float16_t outp;
       
       outp = arm_cosine_distance_f16(inpA, inpB, this->vecDim);
         
    } 

    void DistanceF16::test_euclidean_distance_f16()
    {
       float16_t outp;
       
       outp = arm_euclidean_distance_f16(inpA, inpB, this->vecDim);
         
    } 

    void DistanceF16::test_jensenshannon_distance_f16()
    {
       float16_t outp;

       outp = arm_jensenshannon_distance_f16(inpA, inpB, this->vecDim);
         
    } 

    void DistanceF16::test_minkowski_distance_f16()
    {
       float16_t outp;
       
       outp = arm_minkowski_distance_f16(inpA, inpB, 2,this->vecDim);
  
    } 
  
  
    void DistanceF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->vecDim = *it++;
 
        if ((id != DistanceF16::TEST_MINKOWSKI_DISTANCE_F16_9) && (id != DistanceF16::TEST_JENSENSHANNON_DISTANCE_F16_8))
        {
            inputA.reload(DistanceF16::INPUTA_PROBA_F16_ID,mgr);
            inputB.reload(DistanceF16::INPUTB_PROBA_F16_ID,mgr);
            
        }
        else
        {
           inputA.reload(DistanceF16::INPUTA_F16_ID,mgr);
           inputB.reload(DistanceF16::INPUTB_F16_ID,mgr);
        }

        if (id == DistanceF16::TEST_CORRELATION_DISTANCE_F16_5)
        {
              tmpA.create(this->vecDim,DistanceF16::TMPA_F16_ID,mgr);
              tmpB.create(this->vecDim,DistanceF16::TMPB_F16_ID,mgr);

              tmpAp = tmpA.ptr();
              tmpBp = tmpB.ptr();
        }

       inpA=inputA.ptr();
       inpB=inputB.ptr();
       

    }

    void DistanceF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       (void)mgr;
    }
