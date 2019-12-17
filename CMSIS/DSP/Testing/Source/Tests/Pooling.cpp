#include "Pooling.h"
#include <stdio.h>
#include "Error.h"
#include "arm_nnfunctions.h"
#include "Test.h"


    void Pooling::test_avgpool_s8()
    {
       const q7_t *inp = input.ptr();
       q7_t *tmpin = tmpInput.ptr();
       q7_t *outp = output.ptr();
       q15_t *tempp = temp.ptr();

       memcpy(tmpin,inp,input.nbSamples());

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
       
       ASSERT_EQ(ref,output);
    } 

  
    void Pooling::setUp(Testing::testID_t id,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr)
    {

       switch(id)
       {
          case Pooling::TEST_AVGPOOL_S8_1:
            input.reload(Pooling::INPUT1_S8_ID,mgr);
            ref.reload(Pooling::REF1_S8_ID,mgr);

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

          case Pooling::TEST_AVGPOOL_S8_2:
            input.reload(Pooling::INPUT2_S8_ID,mgr);
            ref.reload(Pooling::REF2_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 2;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 2;
            this->STRIDE_Y= 2;
            this->ACT_MIN= -128;
            this->ACT_MAX= 127;
          break; 

          case Pooling::TEST_AVGPOOL_S8_3:
            input.reload(Pooling::INPUT3_S8_ID,mgr);
            ref.reload(Pooling::REF3_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 2;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 2;
            this->STRIDE_Y= 2;
            this->ACT_MIN= -8;
            this->ACT_MAX= 8;
          break; 

          case Pooling::TEST_AVGPOOL_S8_4:
            input.reload(Pooling::INPUT4_S8_ID,mgr);
            ref.reload(Pooling::REF4_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 2;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 2;
            this->STRIDE_Y= 2;
            this->ACT_MIN= 0;
            this->ACT_MAX= 48;
          break; 

          case Pooling::TEST_AVGPOOL_S8_5:
            input.reload(Pooling::INPUT5_S8_ID,mgr);
            ref.reload(Pooling::REF5_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 2;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 2;
            this->STRIDE_Y= 2;
            this->ACT_MIN= 0;
            this->ACT_MAX= 48;
          break; 

          case Pooling::TEST_AVGPOOL_S8_6:
            input.reload(Pooling::INPUT6_S8_ID,mgr);
            ref.reload(Pooling::REF6_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 4;
            this->DIM_OUT_Y= 2;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 1;
            this->STRIDE_Y= 1;
            this->ACT_MIN= -128;
            this->ACT_MAX= 127;
          break; 

          case Pooling::TEST_AVGPOOL_S8_7:
            input.reload(Pooling::INPUT7_S8_ID,mgr);
            ref.reload(Pooling::REF7_S8_ID,mgr);

            this->DIM_IN_X= 4;
            this->DIM_IN_Y= 2;
            this->DIM_OUT_X= 3;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 2;
            this->DIM_FILTER_Y= 2;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 1;
            this->STRIDE_Y= 1;
            this->ACT_MIN= -128;
            this->ACT_MAX= 127;
          break; 

          case Pooling::TEST_AVGPOOL_S8_8:
            input.reload(Pooling::INPUT8_S8_ID,mgr);
            ref.reload(Pooling::REF8_S8_ID,mgr);

            this->DIM_IN_X= 63;
            this->DIM_IN_Y= 63;
            this->DIM_OUT_X= 1;
            this->DIM_OUT_Y= 1;
            this->IN_CHANNEL= 1;
            this->DIM_FILTER_X= 63;
            this->DIM_FILTER_Y= 63;
            this->PAD_WIDTH=  0;
            this->PAD_HEIGHT= 0;
            this->STRIDE_X= 2;
            this->STRIDE_Y= 2;
            this->ACT_MIN= -128;
            this->ACT_MAX= 127;
          break; 




       }
       temp.create(this->DIM_OUT_X * this->IN_CHANNEL,Pooling::TEMP_S8_ID,mgr);

       output.create(ref.nbSamples(),Pooling::OUTPUT_S8_ID,mgr);
       tmpInput.create(input.nbSamples(),Pooling::TEMPINPUT_S8_ID,mgr);




    }

    void Pooling::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
