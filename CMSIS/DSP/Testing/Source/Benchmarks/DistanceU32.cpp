#include "DistanceU32.h"
#include <stdio.h>
#include "Error.h"
#include "Test.h"



  void DistanceU32::test_dice_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_dice_distance(inpA, inpB,this->vecDim);
       
    } 

    void DistanceU32::test_hamming_distance_u32()
    {
       uint32_t outp;

       outp = arm_hamming_distance(inpA, inpB,this->vecDim);

    } 

    void DistanceU32::test_jaccard_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_jaccard_distance(inpA, inpB,this->vecDim);
    
    } 

    void DistanceU32::test_kulsinski_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_kulsinski_distance(inpA, inpB,this->vecDim);
         

    }

    void DistanceU32::test_rogerstanimoto_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_rogerstanimoto_distance(inpA, inpB,this->vecDim);
      
    }

    void DistanceU32::test_russellrao_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_russellrao_distance(inpA, inpB,this->vecDim);
     
    }

    void DistanceU32::test_sokalmichener_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_sokalmichener_distance(inpA, inpB,this->vecDim);
         
    
    }

    void DistanceU32::test_sokalsneath_distance_u32()
    {
       uint32_t outp;
       outp = arm_sokalsneath_distance(inpA, inpB,this->vecDim);
      
    }

    void DistanceU32::test_yule_distance_u32()
    {
       uint32_t outp;
       
       outp = arm_yule_distance(inpA, inpB,this->vecDim);
      
    }


  
  
    void DistanceU32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {
        std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
        this->vecDim = *it++;
 
       
        inputA.reload(DistanceU32::INPUTA_U32_ID,mgr);
        inputB.reload(DistanceU32::INPUTB_U32_ID,mgr);
      
       
        inpA=inputA.ptr();
        inpB=inputB.ptr();
       

    }

    void DistanceU32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
    }
