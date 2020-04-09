#include "MISCQ31.h"
#include "Error.h"

#define MAX(A,B) (A) > (B) ? (A) : (B)
   
    void MISCQ31::test_conv_q31()
    {
       arm_conv_q31(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

    void MISCQ31::test_correlate_q31()
    {
       arm_correlate_q31(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

    
    void MISCQ31::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nba = *it++;
       this->nbb = *it;

       input1.reload(MISCQ31::INPUTSA1_Q31_ID,mgr,this->nba);
       input2.reload(MISCQ31::INPUTSB1_Q31_ID,mgr,this->nbb);

       switch(id)
       {
          case TEST_CONV_Q31_1:
             output.create(this->nba + this->nbb - 1 ,MISCQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;

          case TEST_CORRELATE_Q31_2:
             output.create(2*MAX(this->nba , this->nbb) - 1 ,MISCQ31::OUT_SAMPLES_Q31_ID,mgr);
          break;
       }

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();
       
    }

    void MISCQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
