#include "MISCQ31.h"
#include "Error.h"

#define MAX(A,B) (A) > (B) ? (A) : (B)
   
    void MISCQ31::test_conv_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       const q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();

       arm_conv_q31(inp1, this->nba,inp2, this->nbb, outp);
        
    } 

    void MISCQ31::test_correlate_q31()
    {
       
       const q31_t *inp1=input1.ptr();
       const q31_t *inp2=input2.ptr();
       q31_t *outp=output.ptr();

       arm_correlate_q31(inp1, this->nba,inp2, this->nbb, outp);
        
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
       
    }

    void MISCQ31::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
    }
