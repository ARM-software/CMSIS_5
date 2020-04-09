#include "MISCQ7.h"
#include "Error.h"

#define MAX(A,B) (A) > (B) ? (A) : (B)
   
    void MISCQ7::test_conv_q7()
    {
       arm_conv_q7(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

    void MISCQ7::test_correlate_q7()
    {
       arm_correlate_q7(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

   
    
    void MISCQ7::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nba = *it++;
       this->nbb = *it;

       input1.reload(MISCQ7::INPUTSA1_Q7_ID,mgr,this->nba);
       input2.reload(MISCQ7::INPUTSB1_Q7_ID,mgr,this->nbb);

       switch(id)
       {
          case TEST_CONV_Q7_1:
             output.create(this->nba + this->nbb - 1 ,MISCQ7::OUT_SAMPLES_Q7_ID,mgr);
          break;

          case TEST_CORRELATE_Q7_2:
             output.create(2*MAX(this->nba , this->nbb) - 1 ,MISCQ7::OUT_SAMPLES_Q7_ID,mgr);
          break;
       }

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();
       
    }

    void MISCQ7::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
