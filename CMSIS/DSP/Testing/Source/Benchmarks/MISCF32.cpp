#include "MISCF32.h"
#include "Error.h"

#define MAX(A,B) (A) > (B) ? (A) : (B)
   
    void MISCF32::test_conv_f32()
    {
       arm_conv_f32(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

    void MISCF32::test_correlate_f32()
    {
       arm_correlate_f32(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

   
    
    void MISCF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nba = *it++;
       this->nbb = *it;

       input1.reload(MISCF32::INPUTSA1_F32_ID,mgr,this->nba);
       input2.reload(MISCF32::INPUTSB1_F32_ID,mgr,this->nbb);

       switch(id)
       {
          case TEST_CONV_F32_1:
             output.create(this->nba + this->nbb - 1 ,MISCF32::OUT_SAMPLES_F32_ID,mgr);
          break;

          case TEST_CORRELATE_F32_2:
             output.create(2*MAX(this->nba , this->nbb) - 1 ,MISCF32::OUT_SAMPLES_F32_ID,mgr);
          break;
       }

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();
       
    }

    void MISCF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
