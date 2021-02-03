#include "MISCF16.h"
#include "Error.h"

#define MAX(A,B) (A) > (B) ? (A) : (B)
   
#if 0
    void MISCF16::test_conv_f16()
    {
       arm_conv_f16(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 
#endif 

    void MISCF16::test_correlate_f16()
    {
       arm_correlate_f16(this->inp1, this->nba,this->inp2, this->nbb, this->outp);
    } 

   
    
    void MISCF16::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       std::vector<Testing::param_t>::iterator it = params.begin();
       this->nba = *it++;
       this->nbb = *it;

       input1.reload(MISCF16::INPUTSA1_F16_ID,mgr,this->nba);
       input2.reload(MISCF16::INPUTSB1_F16_ID,mgr,this->nbb);

       switch(id)
       {
  #if 0
          case TEST_CONV_F16_1:
             output.create(this->nba + this->nbb - 1 ,MISCF16::OUT_SAMPLES_F16_ID,mgr);
          break;
  #endif

          case TEST_CORRELATE_F16_2:
             output.create(2*MAX(this->nba , this->nbb) - 1 ,MISCF16::OUT_SAMPLES_F16_ID,mgr);
          break;
       }

       this->inp1=input1.ptr();
       this->inp2=input2.ptr();
       this->outp=output.ptr();
       
    }

    void MISCF16::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
       (void)mgr;
    }
