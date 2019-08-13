#include "MISCQ15.h"
#include "Error.h"

#define MAX(A,B) (A) > (B) ? (A) : (B)
   
    void MISCQ15::test_conv_q15()
    {
       arm_conv_q15(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

    void MISCQ15::test_correlate_q15()
    {
       arm_correlate_q15(this->inp1, this->nba,this->inp2, this->nbb, this->outp);  
    } 

   
    
    void MISCQ15::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nba = *it++;
       this->nbb = *it;

       input1.reload(MISCQ15::INPUTSA1_Q15_ID,mgr,this->nba);
       input2.reload(MISCQ15::INPUTSB1_Q15_ID,mgr,this->nbb);

       switch(id)
       {
          case TEST_CONV_Q15_1:
             output.create(this->nba + this->nbb - 1 ,MISCQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;

          case TEST_CORRELATE_Q15_2:
             output.create(2*MAX(this->nba , this->nbb) - 1 ,MISCQ15::OUT_SAMPLES_Q15_ID,mgr);
          break;
       }

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();
       
    }

    void MISCQ15::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
