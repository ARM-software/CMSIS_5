#include "PoolingBench.h"
#include "Error.h"
#include "arm_nnfunctions.h"
#include "Test.h"

#include <cstdio>


    void PoolingBench::test_avgpool_s8()
    {

      //for(int i=0; i < this->repeatNb; i++)
      {
       arm_avgpool_s8(
           DIM_IN_Y,
           DIM_IN_X,
           DIM_OUT_Y,
           DIM_OUT_X,
           STRIDE_Y,
           STRIDE_X,
           DIM_FILTER_Y,
           DIM_FILTER_X,
           PAD_HEIGHT,
           PAD_WIDTH,
           ACT_MIN,
           ACT_MAX,
           IN_CHANNEL,
           tmpin,
           tempp,
           outp);
       
       }

       //ASSERT_EQ(this->ref,this->output);

    } 

  
    void PoolingBench::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       std::vector<Testing::param_t>::iterator it = paramsArgs.begin();
       this->repeatNb = *it;

       switch(id)
       {
          case PoolingBench::TEST_AVGPOOL_S8_1:
            input.reload(PoolingBench::INPUT1_S8_ID,mgr);
            ref.reload(PoolingBench::REF1_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 2;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 101;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 2;
            this->STRIDE_Y= 2;
            this->ACT_MIN= -128;
            this->ACT_MAX= 127;

          break; 


       }
       temp.create(this->DIM_OUT_X * this->IN_CHANNEL,PoolingBench::TEMP_S8_ID,mgr);

       output.create(ref.nbSamples(),PoolingBench::OUTPUT_S8_ID,mgr);
       tmpInput.create(input.nbSamples(),PoolingBench::TEMPINPUT_S8_ID,mgr);

       const q7_t *inp = input.ptr();


       this->tmpin = tmpInput.ptr();
       this->outp = output.ptr();
       this->tempp = temp.ptr();

       memcpy(tmpin,inp,input.nbSamples());

    }

    void PoolingBench::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        
    }
