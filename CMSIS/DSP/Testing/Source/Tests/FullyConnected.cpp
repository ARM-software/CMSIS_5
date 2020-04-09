#include "FullyConnected.h"
#include <stdio.h>
#include "Error.h"
#include "arm_nnfunctions.h"
#include "Test.h"
#include "stdio.h"



void printPattern(char *s,Client::AnyPattern<q7_t> pat)
{
   q7_t *p=pat.ptr();
   printf("%s\n",s);
   for(int i=0;i < pat.nbSamples(); i++)
   {
      printf("0x%02x\n",p[i]);
   }
   printf("----\n");
}
   
    void FullyConnected::test_fully_connected_tflite_s8()
    {
       
       q7_t *inp=input.ptr();
       q31_t *biasp=bias.ptr();
       q7_t *weightp=weight.ptr();
       q7_t *outp=output.ptr();
       q7_t *refp=ref.ptr();
       q15_t *tempp=temp.ptr();

       arm_fully_connected_s8((int8_t*)inp
        ,(const int8_t*)weightp
        ,colDim
        ,rowDim
        ,nb_batches
        ,input_offset
        ,filter_offset
        ,output_mult
        ,output_shift
        ,output_offset
        ,(const int32_t*)biasp
        ,(int8_t*)outp 
        ,act_min 
        ,act_max
        ,tempp
        );
      
        ASSERT_EQ(ref,output);
    } 

  
    void FullyConnected::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {


       nb_batches = 1;
       

       switch(id)
       {
          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_1:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

             input.reload(FullyConnected::INPUT1_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS1_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT1_S8_ID,mgr);

             ref.reload(FullyConnected::REF1_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(input.nbSamples(),FullyConnected::TEMP_S16_ID,mgr);

             colDim=input.nbSamples();
             rowDim=output.nbSamples();
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_2:
             output_mult = 1073741824;
             output_shift = 1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

             input.reload(FullyConnected::INPUT2_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS2_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT2_S8_ID,mgr);

             ref.reload(FullyConnected::REF2_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(input.nbSamples(),FullyConnected::TEMP_S16_ID,mgr);

             colDim=input.nbSamples();
             rowDim=output.nbSamples();
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_3:
             output_mult = 1073741824;
             output_shift = 2;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-1;
             act_max= 127;

             input.reload(FullyConnected::INPUT3_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS3_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT3_S8_ID,mgr);

             ref.reload(FullyConnected::REF3_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(input.nbSamples(),FullyConnected::TEMP_S16_ID,mgr);

             colDim=input.nbSamples();
             rowDim=output.nbSamples();
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_4:

             output_mult = 1073741824;
             output_shift = 1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;


             input.reload(FullyConnected::INPUT4_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS4_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT4_S8_ID,mgr);

             ref.reload(FullyConnected::REF4_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(input.nbSamples(),FullyConnected::TEMP_S16_ID,mgr);

             colDim=input.nbSamples();
             rowDim=output.nbSamples();
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_5:

             output_mult = 1073741824;
             output_shift = 1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;


             input.reload(FullyConnected::INPUT5_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS5_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT5_S8_ID,mgr);

             ref.reload(FullyConnected::REF5_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(input.nbSamples(),FullyConnected::TEMP_S16_ID,mgr);

             colDim=input.nbSamples();
             rowDim=output.nbSamples();
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_6:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=9;

             colDim=6;
             rowDim=1;

             input.reload(FullyConnected::INPUT6_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS6_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT6_S8_ID,mgr);

             ref.reload(FullyConnected::REF6_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_7:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=8;

             colDim=8;
             rowDim=1;


             input.reload(FullyConnected::INPUT7_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS7_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT7_S8_ID,mgr);

             ref.reload(FullyConnected::REF7_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_8:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=4;

             colDim=10;
             rowDim=1;


             input.reload(FullyConnected::INPUT8_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS8_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT8_S8_ID,mgr);

             ref.reload(FullyConnected::REF8_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_9:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=9;

             colDim=6;
             rowDim=1;

             input.reload(FullyConnected::INPUT9_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS9_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT9_S8_ID,mgr);

             ref.reload(FullyConnected::REF9_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_10:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=4;

             colDim=10;
             rowDim=1;

             input.reload(FullyConnected::INPUT10_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS10_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT10_S8_ID,mgr);

             ref.reload(FullyConnected::REF10_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_11:
             output_mult = 1073741824;
             output_shift = -1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=8;

             colDim=8;
             rowDim=1;

             input.reload(FullyConnected::INPUT11_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS11_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT11_S8_ID,mgr);

             ref.reload(FullyConnected::REF11_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_12:
             output_mult = 1073741824;
             output_shift = 1;
             filter_offset = 0;
             input_offset = 0;
             output_offset = 0;
             act_min =-128;
             act_max= 127;

          
             nb_batches=9;

             colDim=8;
             rowDim=4;


             input.reload(FullyConnected::INPUT12_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS12_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT12_S8_ID,mgr);

             ref.reload(FullyConnected::REF12_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_13:
             output_mult = 1077969154;
             output_shift = 2;
             filter_offset = 0;
             input_offset = 0;
             output_offset = 1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=8;

             colDim=8;
             rowDim=5;

             input.reload(FullyConnected::INPUT13_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS13_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT13_S8_ID,mgr);

             ref.reload(FullyConnected::REF13_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_14:
             output_mult = 1073741824;
             output_shift = 1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=4;

             colDim=7;
             rowDim=3;


             input.reload(FullyConnected::INPUT14_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS14_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT14_S8_ID,mgr);

             ref.reload(FullyConnected::REF14_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;

          case FullyConnected::TEST_FULLY_CONNECTED_TFLITE_S8_15:
             output_mult = 1073741824;
             output_shift = 1;
             filter_offset = 1;
             input_offset = 1;
             output_offset = -1;
             act_min =-128;
             act_max= 127;

          
             nb_batches=8;

             colDim=7;
             rowDim=4;


             input.reload(FullyConnected::INPUT15_S8_ID,mgr);
             bias.reload(FullyConnected::BIAS15_S8_ID,mgr);
             weight.reload(FullyConnected::WEIGHT15_S8_ID,mgr);

             ref.reload(FullyConnected::REF15_S8_ID,mgr);

             output.create(ref.nbSamples(),FullyConnected::OUTPUT_S8_ID,mgr);
             temp.create(colDim,FullyConnected::TEMP_S16_ID,mgr);
          break;
       }
       


    }

    void FullyConnected::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
        output.dump(mgr);
    }
